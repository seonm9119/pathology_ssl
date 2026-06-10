from argparse import ArgumentParser

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_default_pretrain_config
from config import get_supported_model_names
from datasets import build_pathology_pretrain_dataset
from datasets import collate_pathology_images
from losses import InpaintingReconstructionLoss
from masking import generate_pretrain_masks
from models import build_pathology_cvt13_ssl
from models import build_pathology_cvt21_ssl
from utils import AverageMeter
from utils import create_training_dirs
from utils import get_training_device
from utils import load_training_checkpoint
from utils import save_encoder_checkpoint
from utils import save_training_checkpoint
from utils import set_random_seed


PRETRAIN_CONFIG = get_default_pretrain_config()


def parse_args():
    parser = ArgumentParser(description="Pre-train the pathology SSL backbone with masked reconstruction.")
    parser.add_argument("--data-dir", default=PRETRAIN_CONFIG["data_dir"])
    parser.add_argument("--sources", default=PRETRAIN_CONFIG["sources"])
    parser.add_argument("--split", default=PRETRAIN_CONFIG["split"], choices=["train", "val", "test"])
    parser.add_argument("--validation-split", default=PRETRAIN_CONFIG["validation_split"], choices=["train", "val", "test"])
    parser.add_argument("--image-size", default=PRETRAIN_CONFIG["image_size"], type=int)
    parser.add_argument("--batch-size", default=PRETRAIN_CONFIG["batch_size"], type=int)
    parser.add_argument("--min-epochs", default=PRETRAIN_CONFIG["min_epochs"], type=int)
    parser.add_argument("--early-stop-patience", default=PRETRAIN_CONFIG["early_stop_patience"], type=int)
    parser.add_argument("--early-stop-min-delta", default=PRETRAIN_CONFIG["early_stop_min_delta"], type=float)
    parser.add_argument("--validation-every", default=PRETRAIN_CONFIG["validation_every"], type=int)
    parser.add_argument("--learning-rate", default=PRETRAIN_CONFIG["learning_rate"], type=float)
    parser.add_argument("--min-learning-rate", default=PRETRAIN_CONFIG["min_learning_rate"], type=float)
    parser.add_argument("--warmup-epochs", default=PRETRAIN_CONFIG["warmup_epochs"], type=int)
    parser.add_argument("--lr-decay-factor", default=PRETRAIN_CONFIG["lr_decay_factor"], type=float)
    parser.add_argument("--lr-decay-patience", default=PRETRAIN_CONFIG["lr_decay_patience"], type=int)
    parser.add_argument("--weight-decay", default=PRETRAIN_CONFIG["weight_decay"], type=float)
    parser.add_argument("--num-workers", default=PRETRAIN_CONFIG["num_workers"], type=int)
    parser.add_argument("--variant", default=PRETRAIN_CONFIG["variant"], choices=get_supported_model_names())
    parser.add_argument("--mask-style", default=PRETRAIN_CONFIG["mask_style"], choices=["patch", "block", "mixed"])
    parser.add_argument("--mask-ratio", default=PRETRAIN_CONFIG["mask_ratio"], type=float)
    parser.add_argument("--mask-patch-size", default=PRETRAIN_CONFIG["mask_patch_size"], type=int)
    parser.add_argument("--max-samples-per-source", default=PRETRAIN_CONFIG["max_samples_per_source"], type=int)
    parser.add_argument("--checkpoint-dir", default=PRETRAIN_CONFIG["checkpoint_dir"])
    parser.add_argument("--log-dir", default=PRETRAIN_CONFIG["log_dir"])
    parser.add_argument("--resume", default=PRETRAIN_CONFIG["resume"])
    parser.add_argument("--save-every", default=PRETRAIN_CONFIG["save_every"], type=int)
    parser.add_argument("--log-every", default=PRETRAIN_CONFIG["log_every"], type=int)
    parser.add_argument("--max-steps", default=PRETRAIN_CONFIG["max_steps"], type=int)
    parser.add_argument("--max-validation-steps", default=PRETRAIN_CONFIG["max_validation_steps"], type=int)
    parser.add_argument("--grad-clip", default=PRETRAIN_CONFIG["grad_clip"], type=float)
    parser.add_argument("--seed", default=PRETRAIN_CONFIG["seed"], type=int)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def build_ssl_model(variant):
    if variant == "cvt13":
        return build_pathology_cvt13_ssl(input_channels=3)
    if variant == "cvt21":
        return build_pathology_cvt21_ssl(input_channels=3)
    raise ValueError(f"Unsupported model variant: {variant}")


def get_warmup_learning_rate(args, epoch):
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        return args.learning_rate * float(epoch + 1) / float(args.warmup_epochs)
    return None


def set_optimizer_learning_rate(optimizer, learning_rate):
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def create_pretrain_loader(args, device, split, shuffle, drop_last, max_samples_per_source):
    dataset = build_pathology_pretrain_dataset(
        args.data_dir,
        args.sources,
        split,
        args.image_size,
        max_samples_per_source,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        collate_fn=collate_pathology_images,
    )
    return dataset, loader


def train_one_epoch(model, loader, loss_function, optimizer, scaler, writer, device, epoch, global_step, args):
    model.train()
    loss_meter = AverageMeter()
    hole_loss_meter = AverageMeter()
    valid_loss_meter = AverageMeter()
    tv_loss_meter = AverageMeter()
    use_amp = device.type == "cuda" and args.amp

    progress = tqdm(loader, desc=f"epoch {epoch + 1}", leave=False)
    for step_index, images in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks = generate_pretrain_masks(images, args.mask_style, args.mask_patch_size, args.mask_ratio)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            reconstructed_images = model(images, masks)
            loss_parts = loss_function(images, reconstructed_images, masks, return_parts=True)
            loss = loss_parts["loss"]

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        loss_meter.update(loss.detach().item(), batch_size)
        hole_loss_meter.update(loss_parts["hole_loss"].item(), batch_size)
        valid_loss_meter.update(loss_parts["valid_loss"].item(), batch_size)
        tv_loss_meter.update(loss_parts["tv_loss"].item(), batch_size)

        writer.add_scalar("train/loss", loss.detach().item(), global_step)
        writer.add_scalar("train/hole_loss", loss_parts["hole_loss"].item(), global_step)
        writer.add_scalar("train/valid_loss", loss_parts["valid_loss"].item(), global_step)
        writer.add_scalar("train/tv_loss", loss_parts["tv_loss"].item(), global_step)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)

        if args.log_every and (step_index + 1) % args.log_every == 0:
            progress.set_postfix(loss=f"{loss_meter.average:.4f}", hole=f"{hole_loss_meter.average:.4f}")

        global_step += 1
        if args.max_steps and step_index + 1 >= args.max_steps:
            break

    epoch_metrics = {
        "loss": loss_meter.average,
        "hole_loss": hole_loss_meter.average,
        "valid_loss": valid_loss_meter.average,
        "tv_loss": tv_loss_meter.average,
    }
    return epoch_metrics, global_step


def validate_one_epoch(model, loader, loss_function, writer, device, epoch, args):
    model.eval()
    loss_meter = AverageMeter()
    hole_loss_meter = AverageMeter()
    valid_loss_meter = AverageMeter()
    tv_loss_meter = AverageMeter()
    use_amp = device.type == "cuda" and args.amp

    with torch.no_grad():
        progress = tqdm(loader, desc=f"validation {epoch + 1}", leave=False)
        for step_index, images in enumerate(progress):
            images = images.to(device, non_blocking=True)
            masks = generate_pretrain_masks(images, args.mask_style, args.mask_patch_size, args.mask_ratio)

            with torch.amp.autocast(device.type, enabled=use_amp):
                reconstructed_images = model(images, masks)
                loss_parts = loss_function(images, reconstructed_images, masks, return_parts=True)
                loss = loss_parts["loss"]

            batch_size = images.shape[0]
            loss_meter.update(loss.detach().item(), batch_size)
            hole_loss_meter.update(loss_parts["hole_loss"].item(), batch_size)
            valid_loss_meter.update(loss_parts["valid_loss"].item(), batch_size)
            tv_loss_meter.update(loss_parts["tv_loss"].item(), batch_size)

            if args.max_validation_steps and step_index + 1 >= args.max_validation_steps:
                break

    validation_metrics = {
        "loss": loss_meter.average,
        "hole_loss": hole_loss_meter.average,
        "valid_loss": valid_loss_meter.average,
        "tv_loss": tv_loss_meter.average,
    }
    writer.add_scalar("validation/loss", validation_metrics["loss"], epoch + 1)
    writer.add_scalar("validation/hole_loss", validation_metrics["hole_loss"], epoch + 1)
    writer.add_scalar("validation/valid_loss", validation_metrics["valid_loss"], epoch + 1)
    writer.add_scalar("validation/tv_loss", validation_metrics["tv_loss"], epoch + 1)
    return validation_metrics


def save_epoch_checkpoints(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    args,
    checkpoint_dir,
    validation_loss,
    best_validation_loss,
    epochs_without_improvement,
):
    last_checkpoint_path = checkpoint_dir / "last.pt"
    last_encoder_path = checkpoint_dir / "encoder_last.pt"
    save_training_checkpoint(
        last_checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        global_step,
        args,
        validation_loss,
        best_validation_loss,
        epochs_without_improvement,
    )
    save_encoder_checkpoint(
        last_encoder_path,
        model,
        epoch,
        global_step,
        args,
        validation_loss,
        best_validation_loss,
        epochs_without_improvement,
    )

    should_save_epoch_checkpoint = args.save_every and (epoch + 1) % args.save_every == 0
    if should_save_epoch_checkpoint:
        epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
        epoch_encoder_path = checkpoint_dir / f"encoder_epoch_{epoch + 1:04d}.pt"
        save_training_checkpoint(
            epoch_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            args,
            validation_loss,
            best_validation_loss,
            epochs_without_improvement,
        )
        save_encoder_checkpoint(
            epoch_encoder_path,
            model,
            epoch,
            global_step,
            args,
            validation_loss,
            best_validation_loss,
            epochs_without_improvement,
        )


def save_best_checkpoints(model, optimizer, scheduler, scaler, epoch, global_step, args, checkpoint_dir, validation_loss):
    best_checkpoint_path = checkpoint_dir / "best.pt"
    best_encoder_path = checkpoint_dir / "encoder_best.pt"
    save_training_checkpoint(
        best_checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        global_step,
        args,
        validation_loss,
        validation_loss,
        0,
    )
    save_encoder_checkpoint(best_encoder_path, model, epoch, global_step, args, validation_loss, validation_loss, 0)


def main():
    args = parse_args()
    if args.validation_every <= 0:
        raise ValueError("validation_every must be greater than 0 for early stopping")

    set_random_seed(args.seed)
    device = get_training_device()
    checkpoint_dir, log_dir = create_training_dirs(args.checkpoint_dir, args.log_dir)

    train_dataset, train_loader = create_pretrain_loader(
        args,
        device,
        args.split,
        shuffle=True,
        drop_last=True,
        max_samples_per_source=args.max_samples_per_source,
    )
    validation_dataset, validation_loader = create_pretrain_loader(
        args,
        device,
        args.validation_split,
        shuffle=False,
        drop_last=False,
        max_samples_per_source=args.max_samples_per_source,
    )
    model = build_ssl_model(args.variant).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        min_lr=args.min_learning_rate,
    )
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = InpaintingReconstructionLoss()
    writer = SummaryWriter(log_dir)

    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = load_training_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        set_optimizer_learning_rate(optimizer, args.learning_rate)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_validation_loss = checkpoint.get("best_validation_loss") or float("inf")
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"validation samples: {len(validation_dataset)}")
    print(f"checkpoints: {checkpoint_dir}")
    print(f"tensorboard: {log_dir}")
    print(f"learning rate: {optimizer.param_groups[0]['lr']}")

    epoch = start_epoch
    while True:
        warmup_learning_rate = get_warmup_learning_rate(args, epoch)
        if warmup_learning_rate is not None:
            set_optimizer_learning_rate(optimizer, warmup_learning_rate)

        epoch_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            scaler,
            writer,
            device,
            epoch,
            global_step,
            args,
        )
        writer.add_scalar("epoch/loss", epoch_metrics["loss"], epoch + 1)
        writer.add_scalar("epoch/hole_loss", epoch_metrics["hole_loss"], epoch + 1)
        writer.add_scalar("epoch/valid_loss", epoch_metrics["valid_loss"], epoch + 1)
        writer.add_scalar("epoch/tv_loss", epoch_metrics["tv_loss"], epoch + 1)

        should_validate = (epoch + 1) % args.validation_every == 0
        validation_loss = None
        if should_validate:
            validation_metrics = validate_one_epoch(model, validation_loader, loss_function, writer, device, epoch, args)
            validation_loss = validation_metrics["loss"]
            if warmup_learning_rate is None:
                scheduler.step(validation_loss)

            improved = validation_loss < best_validation_loss - args.early_stop_min_delta
            if improved:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
                save_best_checkpoints(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    args,
                    checkpoint_dir,
                    validation_loss,
                )
            else:
                epochs_without_improvement += 1

            writer.add_scalar("early_stop/best_validation_loss", best_validation_loss, epoch + 1)
            writer.add_scalar("early_stop/epochs_without_improvement", epochs_without_improvement, epoch + 1)

        validation_text = "skipped"
        if validation_loss is not None:
            validation_text = f"{validation_loss:.4f}"
        print(
            f"epoch {epoch + 1}: "
            f"loss={epoch_metrics['loss']:.4f}, "
            f"hole={epoch_metrics['hole_loss']:.4f}, "
            f"valid={epoch_metrics['valid_loss']:.4f}, "
            f"tv={epoch_metrics['tv_loss']:.4f}, "
            f"val={validation_text}, "
            f"best_val={best_validation_loss:.4f}, "
            f"stale={epochs_without_improvement}/{args.early_stop_patience}"
        )
        save_epoch_checkpoints(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            args,
            checkpoint_dir,
            validation_loss,
            best_validation_loss,
            epochs_without_improvement,
        )

        should_stop = (
            should_validate
            and not improved
            and epoch + 1 >= args.min_epochs
            and epochs_without_improvement >= args.early_stop_patience
        )
        if should_stop:
            print(f"early stopping: validation loss did not improve for {epochs_without_improvement} validations")
            print(f"best checkpoint: {checkpoint_dir / 'best.pt'}")
            print(f"best encoder: {checkpoint_dir / 'encoder_best.pt'}")
            break

        epoch += 1

    writer.close()


if __name__ == "__main__":
    main()
