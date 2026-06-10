from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_default_segmentation_config
from config import get_supported_model_names
from datasets import PanNukeSegmentationDataset
from datasets import collate_segmentation_samples
from losses import SegmentationLoss
from metrics import segmentation_mean_iou
from metrics import segmentation_pixel_accuracy
from models import build_pathology_cvt13_segmentation
from models import build_pathology_cvt21_segmentation
from utils import AverageMeter
from utils import create_training_dirs
from utils import get_training_device
from utils import load_encoder_checkpoint
from utils import set_random_seed


SEGMENTATION_CONFIG = get_default_segmentation_config()


def parse_args():
    parser = ArgumentParser(description="Train a PanNuke segmenter with the pathology SSL backbone.")
    parser.add_argument("--data-dir", default=SEGMENTATION_CONFIG["data_dir"])
    parser.add_argument("--image-size", default=SEGMENTATION_CONFIG["image_size"], type=int)
    parser.add_argument("--batch-size", default=SEGMENTATION_CONFIG["batch_size"], type=int)
    parser.add_argument("--min-epochs", default=SEGMENTATION_CONFIG["min_epochs"], type=int)
    parser.add_argument("--early-stop-patience", default=SEGMENTATION_CONFIG["early_stop_patience"], type=int)
    parser.add_argument("--early-stop-min-delta", default=SEGMENTATION_CONFIG["early_stop_min_delta"], type=float)
    parser.add_argument("--learning-rate", default=SEGMENTATION_CONFIG["learning_rate"], type=float)
    parser.add_argument("--encoder-learning-rate", default=SEGMENTATION_CONFIG["encoder_learning_rate"], type=float)
    parser.add_argument("--min-learning-rate", default=SEGMENTATION_CONFIG["min_learning_rate"], type=float)
    parser.add_argument("--weight-decay", default=SEGMENTATION_CONFIG["weight_decay"], type=float)
    parser.add_argument("--num-workers", default=SEGMENTATION_CONFIG["num_workers"], type=int)
    parser.add_argument("--variant", default=SEGMENTATION_CONFIG["variant"], choices=get_supported_model_names())
    parser.add_argument("--num-classes", default=SEGMENTATION_CONFIG["num_classes"], type=int)
    parser.add_argument("--dice-weight", default=SEGMENTATION_CONFIG["dice_weight"], type=float)
    parser.add_argument("--freeze-encoder", default=SEGMENTATION_CONFIG["freeze_encoder"], action="store_true")
    parser.add_argument("--encoder-checkpoint", default=SEGMENTATION_CONFIG["encoder_checkpoint"])
    parser.add_argument("--checkpoint-dir", default=SEGMENTATION_CONFIG["checkpoint_dir"])
    parser.add_argument("--log-dir", default=SEGMENTATION_CONFIG["log_dir"])
    parser.add_argument("--log-every", default=SEGMENTATION_CONFIG["log_every"], type=int)
    parser.add_argument("--max-steps", default=SEGMENTATION_CONFIG["max_steps"], type=int)
    parser.add_argument("--max-validation-steps", default=SEGMENTATION_CONFIG["max_validation_steps"], type=int)
    parser.add_argument("--max-samples", default=SEGMENTATION_CONFIG["max_samples"], type=int)
    parser.add_argument("--grad-clip", default=SEGMENTATION_CONFIG["grad_clip"], type=float)
    parser.add_argument("--seed", default=SEGMENTATION_CONFIG["seed"], type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def get_device(device_name):
    if device_name == "auto":
        return get_training_device()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def build_segmenter(variant, num_classes):
    if variant == "cvt13":
        return build_pathology_cvt13_segmentation(num_classes)
    if variant == "cvt21":
        return build_pathology_cvt21_segmentation(num_classes)
    raise ValueError(f"Unsupported model variant: {variant}")


def create_segmentation_loader(args, split, shuffle):
    dataset = PanNukeSegmentationDataset(
        Path(args.data_dir) / "pannuke",
        split,
        args.image_size,
        args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
        collate_fn=collate_segmentation_samples,
    )
    return dataset, loader


def load_encoder_if_available(model, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"encoder checkpoint not found, training from random init: {checkpoint_path}", flush=True)
        return
    load_encoder_checkpoint(model, checkpoint_path)
    print(f"loaded encoder checkpoint: {checkpoint_path}", flush=True)


def build_optimizer(model, args):
    if args.freeze_encoder:
        model.freeze_encoder()
        return AdamW(model.decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_learning_rate},
            {"params": model.decoder.parameters(), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )


def train_one_epoch(model, loader, loss_function, optimizer, writer, device, epoch, global_step, args):
    model.train()
    loss_meter = AverageMeter()
    cross_entropy_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    progress = tqdm(loader, desc=f"segmentation epoch {epoch + 1}", leave=False)
    for step_index, (images, masks) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss_parts = loss_function(logits, masks, return_parts=True)
        loss = loss_parts["loss"]
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch_size = images.shape[0]
        mean_iou = segmentation_mean_iou(logits.detach(), masks, args.num_classes)
        loss_meter.update(loss.detach().item(), batch_size)
        cross_entropy_meter.update(loss_parts["cross_entropy_loss"].item(), batch_size)
        dice_meter.update(loss_parts["dice_loss"].item(), batch_size)
        iou_meter.update(mean_iou, batch_size)
        writer.add_scalar("train/loss", loss.detach().item(), global_step)
        writer.add_scalar("train/cross_entropy_loss", loss_parts["cross_entropy_loss"].item(), global_step)
        writer.add_scalar("train/dice_loss", loss_parts["dice_loss"].item(), global_step)
        writer.add_scalar("train/mean_iou", mean_iou, global_step)

        if args.log_every and (step_index + 1) % args.log_every == 0:
            progress.set_postfix(loss=f"{loss_meter.average:.4f}", iou=f"{iou_meter.average:.4f}")

        global_step += 1
        if args.max_steps and step_index + 1 >= args.max_steps:
            break

    return {
        "loss": loss_meter.average,
        "cross_entropy_loss": cross_entropy_meter.average,
        "dice_loss": dice_meter.average,
        "mean_iou": iou_meter.average,
    }, global_step


def validate_one_epoch(model, loader, loss_function, writer, device, epoch, args):
    model.eval()
    loss_meter = AverageMeter()
    cross_entropy_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    pixel_accuracy_meter = AverageMeter()

    with torch.no_grad():
        progress = tqdm(loader, desc=f"segmentation validation {epoch + 1}", leave=False)
        for step_index, (images, masks) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss_parts = loss_function(logits, masks, return_parts=True)
            loss = loss_parts["loss"]

            batch_size = images.shape[0]
            mean_iou = segmentation_mean_iou(logits, masks, args.num_classes)
            pixel_accuracy = segmentation_pixel_accuracy(logits, masks)
            loss_meter.update(loss.item(), batch_size)
            cross_entropy_meter.update(loss_parts["cross_entropy_loss"].item(), batch_size)
            dice_meter.update(loss_parts["dice_loss"].item(), batch_size)
            iou_meter.update(mean_iou, batch_size)
            pixel_accuracy_meter.update(pixel_accuracy, batch_size)

            if args.max_validation_steps and step_index + 1 >= args.max_validation_steps:
                break

    writer.add_scalar("validation/loss", loss_meter.average, epoch + 1)
    writer.add_scalar("validation/cross_entropy_loss", cross_entropy_meter.average, epoch + 1)
    writer.add_scalar("validation/dice_loss", dice_meter.average, epoch + 1)
    writer.add_scalar("validation/mean_iou", iou_meter.average, epoch + 1)
    writer.add_scalar("validation/pixel_accuracy", pixel_accuracy_meter.average, epoch + 1)
    return {
        "loss": loss_meter.average,
        "cross_entropy_loss": cross_entropy_meter.average,
        "dice_loss": dice_meter.average,
        "mean_iou": iou_meter.average,
        "pixel_accuracy": pixel_accuracy_meter.average,
    }


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, global_step, metrics, args):
    checkpoint = {
        "model_state": model.state_dict(),
        "encoder_state": model.encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = get_device(args.device)
    checkpoint_dir, log_dir = create_training_dirs(args.checkpoint_dir, args.log_dir)

    train_dataset, train_loader = create_segmentation_loader(args, "train", True)
    validation_dataset, validation_loader = create_segmentation_loader(args, "val", False)
    model = build_segmenter(args.variant, args.num_classes).to(device)
    load_encoder_if_available(model, args.encoder_checkpoint)
    optimizer = build_optimizer(model, args)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.early_stop_patience // 2),
        min_lr=args.min_learning_rate,
    )
    loss_function = SegmentationLoss(args.num_classes, args.dice_weight)
    writer = SummaryWriter(log_dir)

    print(f"device: {device}", flush=True)
    print(f"train samples: {len(train_dataset)}", flush=True)
    print(f"validation samples: {len(validation_dataset)}", flush=True)
    print(f"checkpoints: {checkpoint_dir}", flush=True)
    print(f"tensorboard: {log_dir}", flush=True)

    global_step = 0
    epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    while True:
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            writer,
            device,
            epoch,
            global_step,
            args,
        )
        validation_metrics = validate_one_epoch(model, validation_loader, loss_function, writer, device, epoch, args)
        scheduler.step(validation_metrics["loss"])

        improved = validation_metrics["loss"] < best_validation_loss - args.early_stop_min_delta
        if improved:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, scheduler, epoch, global_step, validation_metrics, args)
        else:
            epochs_without_improvement += 1
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, global_step, validation_metrics, args)

        print(
            f"epoch {epoch + 1}: "
            f"loss={train_metrics['loss']:.4f}, "
            f"iou={train_metrics['mean_iou']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_iou={validation_metrics['mean_iou']:.4f}, "
            f"val_pixel_acc={validation_metrics['pixel_accuracy']:.4f}, "
            f"best_val={best_validation_loss:.4f}, "
            f"stale={epochs_without_improvement}/{args.early_stop_patience}",
            flush=True,
        )

        if epoch + 1 >= args.min_epochs and epochs_without_improvement >= args.early_stop_patience:
            print(f"early stopping: validation loss did not improve for {epochs_without_improvement} validations", flush=True)
            break

        epoch += 1

    writer.close()


if __name__ == "__main__":
    main()
