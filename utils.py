from pathlib import Path
import random

import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_training_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_training_dirs(checkpoint_dir, log_dir):
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, log_dir


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, metric_value, sample_count=1):
        self.total += float(metric_value) * sample_count
        self.count += sample_count

    @property
    def average(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


def save_training_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    args,
    validation_loss=None,
    best_validation_loss=None,
    epochs_without_improvement=0,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = {
        "model_state": model.state_dict(),
        "encoder_state": model.encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    torch.save(checkpoint, checkpoint_path)


def save_encoder_checkpoint(
    checkpoint_path,
    model,
    epoch,
    global_step,
    args,
    validation_loss=None,
    best_validation_loss=None,
    epochs_without_improvement=0,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = {
        "encoder_state": model.encoder.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
    }
    torch.save(checkpoint, checkpoint_path)


def load_training_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])

    return checkpoint


def load_encoder_checkpoint(model, checkpoint_path, strict=False):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder_state = checkpoint.get("encoder_state", checkpoint)
    load_result = model.encoder.load_state_dict(encoder_state, strict=strict)
    return load_result
