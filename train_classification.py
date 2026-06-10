from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_default_classification_config
from config import get_medmnist_classification_config
from config import get_supported_classification_sources
from config import get_supported_model_names
from datasets import MedMnistClassificationDataset
from datasets import collate_classification_samples
from metrics import classification_accuracy
from models import build_pathology_cvt13_classifier
from models import build_pathology_cvt21_classifier
from utils import AverageMeter
from utils import create_training_dirs
from utils import get_training_device
from utils import load_encoder_checkpoint
from utils import set_random_seed


CLASSIFICATION_CONFIG = get_default_classification_config()


def parse_args():
    parser = ArgumentParser(description="Train a MedMNIST classifier with the pathology SSL backbone.")
    parser.add_argument("--data-dir", default=CLASSIFICATION_CONFIG["data_dir"])
    parser.add_argument("--source", default=CLASSIFICATION_CONFIG["source"], choices=get_supported_classification_sources())
    parser.add_argument("--image-size", default=CLASSIFICATION_CONFIG["image_size"], type=int)
    parser.add_argument("--batch-size", default=CLASSIFICATION_CONFIG["batch_size"], type=int)
    parser.add_argument("--min-epochs", default=CLASSIFICATION_CONFIG["min_epochs"], type=int)
    parser.add_argument("--early-stop-patience", default=CLASSIFICATION_CONFIG["early_stop_patience"], type=int)
    parser.add_argument("--early-stop-min-delta", default=CLASSIFICATION_CONFIG["early_stop_min_delta"], type=float)
    parser.add_argument("--learning-rate", default=CLASSIFICATION_CONFIG["learning_rate"], type=float)
    parser.add_argument("--encoder-learning-rate", default=CLASSIFICATION_CONFIG["encoder_learning_rate"], type=float)
    parser.add_argument("--min-learning-rate", default=CLASSIFICATION_CONFIG["min_learning_rate"], type=float)
    parser.add_argument("--weight-decay", default=CLASSIFICATION_CONFIG["weight_decay"], type=float)
    parser.add_argument("--num-workers", default=CLASSIFICATION_CONFIG["num_workers"], type=int)
    parser.add_argument("--variant", default=CLASSIFICATION_CONFIG["variant"], choices=get_supported_model_names())
    parser.add_argument("--freeze-encoder", default=CLASSIFICATION_CONFIG["freeze_encoder"], action="store_true")
    parser.add_argument("--encoder-checkpoint", default=CLASSIFICATION_CONFIG["encoder_checkpoint"])
    parser.add_argument("--checkpoint-dir", default=CLASSIFICATION_CONFIG["checkpoint_dir"])
    parser.add_argument("--log-dir", default=CLASSIFICATION_CONFIG["log_dir"])
    parser.add_argument("--log-every", default=CLASSIFICATION_CONFIG["log_every"], type=int)
    parser.add_argument("--max-steps", default=CLASSIFICATION_CONFIG["max_steps"], type=int)
    parser.add_argument("--max-validation-steps", default=CLASSIFICATION_CONFIG["max_validation_steps"], type=int)
    parser.add_argument("--max-samples", default=CLASSIFICATION_CONFIG["max_samples"], type=int)
    parser.add_argument("--grad-clip", default=CLASSIFICATION_CONFIG["grad_clip"], type=float)
    parser.add_argument("--seed", default=CLASSIFICATION_CONFIG["seed"], type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def get_device(device_name):
    if device_name == "auto":
        return get_training_device()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def build_classifier(variant, num_classes):
    if variant == "cvt13":
        return build_pathology_cvt13_classifier(num_classes)
    if variant == "cvt21":
        return build_pathology_cvt21_classifier(num_classes)
    raise ValueError(f"Unsupported model variant: {variant}")


def create_classification_loader(args, split, shuffle):
    source_config = get_medmnist_classification_config(args.source)
    dataset_path = Path(args.data_dir) / "medmnist_pathology_128" / source_config["file_name"]
    cache_dir = Path(args.data_dir) / "cache" / "medmnist_pathology_128"
    dataset = MedMnistClassificationDataset(
        dataset_path,
        split,
        args.image_size,
        args.source,
        cache_dir,
        args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=split == "train",
        collate_fn=collate_classification_samples,
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
        return AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_learning_rate},
            {"params": list(model.norm.parameters()) + list(model.head.parameters()), "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )


def train_one_epoch(model, loader, loss_function, optimizer, writer, device, epoch, global_step, args):
    model.train()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    progress = tqdm(loader, desc=f"classification epoch {epoch + 1}", leave=False)
    for step_index, (images, labels) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch_size = images.shape[0]
        accuracy = classification_accuracy(logits.detach(), labels)
        loss_meter.update(loss.detach().item(), batch_size)
        accuracy_meter.update(accuracy, batch_size)
        writer.add_scalar("train/loss", loss.detach().item(), global_step)
        writer.add_scalar("train/accuracy", accuracy, global_step)

        if args.log_every and (step_index + 1) % args.log_every == 0:
            progress.set_postfix(loss=f"{loss_meter.average:.4f}", acc=f"{accuracy_meter.average:.4f}")

        global_step += 1
        if args.max_steps and step_index + 1 >= args.max_steps:
            break

    return {"loss": loss_meter.average, "accuracy": accuracy_meter.average}, global_step


def validate_one_epoch(model, loader, loss_function, writer, device, epoch, args):
    model.eval()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()

    with torch.no_grad():
        progress = tqdm(loader, desc=f"classification validation {epoch + 1}", leave=False)
        for step_index, (images, labels) in enumerate(progress):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            accuracy = classification_accuracy(logits, labels)

            batch_size = images.shape[0]
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(accuracy, batch_size)

            if args.max_validation_steps and step_index + 1 >= args.max_validation_steps:
                break

    writer.add_scalar("validation/loss", loss_meter.average, epoch + 1)
    writer.add_scalar("validation/accuracy", accuracy_meter.average, epoch + 1)
    return {"loss": loss_meter.average, "accuracy": accuracy_meter.average}


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
    source_config = get_medmnist_classification_config(args.source)

    train_dataset, train_loader = create_classification_loader(args, "train", True)
    validation_dataset, validation_loader = create_classification_loader(args, "val", False)
    model = build_classifier(args.variant, source_config["num_classes"]).to(device)
    load_encoder_if_available(model, args.encoder_checkpoint)
    optimizer = build_optimizer(model, args)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(1, args.early_stop_patience // 2),
        min_lr=args.min_learning_rate,
    )
    loss_function = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir)

    print(f"device: {device}", flush=True)
    print(f"source: {args.source}", flush=True)
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
            f"acc={train_metrics['accuracy']:.4f}, "
            f"val_loss={validation_metrics['loss']:.4f}, "
            f"val_acc={validation_metrics['accuracy']:.4f}, "
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
