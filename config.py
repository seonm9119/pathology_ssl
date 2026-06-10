from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

MODEL_CONFIGS = {
    "cvt13": {
        "stage_depths": [1, 2, 10],
        "stage_channels": [64, 192, 384],
        "stage_heads": [1, 3, 6],
        "patch_sizes": [7, 3, 3],
        "patch_strides": [4, 2, 2],
        "patch_paddings": [2, 1, 1],
        "mlp_ratios": [4.0, 4.0, 4.0],
        "drop_rates": [0.0, 0.0, 0.0],
        "drop_path_rate": 0.1,
        "pretrain_image_size": 64,
        "pretrain_mask_patch_size": 8,
    },
    "cvt21": {
        "stage_depths": [1, 4, 16],
        "stage_channels": [64, 192, 384],
        "stage_heads": [1, 3, 6],
        "patch_sizes": [7, 3, 3],
        "patch_strides": [4, 2, 2],
        "patch_paddings": [2, 1, 1],
        "mlp_ratios": [4.0, 4.0, 4.0],
        "drop_rates": [0.0, 0.0, 0.0],
        "drop_path_rate": 0.1,
        "pretrain_image_size": 28,
        "pretrain_mask_patch_size": 4,
    },
}

DEFAULT_PRETRAIN_CONFIG = {
    "data_dir": str(PROJECT_DIR / "data"),
    "sources": "pathmnist_128,bloodmnist_128,tissuemnist_128,pannuke",
    "split": "train",
    "validation_split": "val",
    "image_size": 64,
    "batch_size": 32,
    "min_epochs": 20,
    "early_stop_patience": 8,
    "early_stop_min_delta": 1e-4,
    "validation_every": 1,
    "learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
    "warmup_epochs": 5,
    "lr_decay_factor": 0.5,
    "lr_decay_patience": 2,
    "weight_decay": 0.05,
    "num_workers": 4,
    "variant": "cvt13",
    "mask_style": "patch",
    "mask_ratio": 0.5,
    "mask_patch_size": 8,
    "max_samples_per_source": 0,
    "checkpoint_dir": str(PROJECT_DIR / "checkpoints" / "ssl"),
    "log_dir": str(PROJECT_DIR / "runs" / "ssl"),
    "resume": "",
    "save_every": 0,
    "log_every": 50,
    "max_steps": 0,
    "max_validation_steps": 0,
    "grad_clip": 1.0,
    "seed": 42,
}

DEFAULT_CLASSIFICATION_CONFIG = {
    "data_dir": str(PROJECT_DIR / "data"),
    "source": "pathmnist_128",
    "image_size": 64,
    "batch_size": 64,
    "min_epochs": 5,
    "early_stop_patience": 8,
    "early_stop_min_delta": 1e-4,
    "learning_rate": 1e-3,
    "encoder_learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
    "weight_decay": 0.05,
    "num_workers": 4,
    "variant": "cvt13",
    "freeze_encoder": False,
    "encoder_checkpoint": str(PROJECT_DIR / "checkpoints" / "ssl" / "encoder_best.pt"),
    "checkpoint_dir": str(PROJECT_DIR / "checkpoints" / "classification"),
    "log_dir": str(PROJECT_DIR / "runs" / "classification"),
    "log_every": 50,
    "max_steps": 0,
    "max_validation_steps": 0,
    "max_samples": 0,
    "grad_clip": 1.0,
    "seed": 42,
}

DEFAULT_SEGMENTATION_CONFIG = {
    "data_dir": str(PROJECT_DIR / "data"),
    "image_size": 128,
    "batch_size": 8,
    "min_epochs": 5,
    "early_stop_patience": 8,
    "early_stop_min_delta": 1e-4,
    "learning_rate": 5e-4,
    "encoder_learning_rate": 5e-5,
    "min_learning_rate": 1e-6,
    "weight_decay": 0.05,
    "num_workers": 4,
    "variant": "cvt13",
    "num_classes": 6,
    "dice_weight": 0.5,
    "freeze_encoder": False,
    "encoder_checkpoint": str(PROJECT_DIR / "checkpoints" / "ssl" / "encoder_best.pt"),
    "checkpoint_dir": str(PROJECT_DIR / "checkpoints" / "segmentation"),
    "log_dir": str(PROJECT_DIR / "runs" / "segmentation"),
    "log_every": 25,
    "max_steps": 0,
    "max_validation_steps": 0,
    "max_samples": 0,
    "grad_clip": 1.0,
    "seed": 42,
}

MEDMNIST_CLASSIFICATION_CONFIGS = {
    "pathmnist_128": {
        "file_name": "pathmnist_128.npz",
        "num_classes": 9,
    },
    "bloodmnist_128": {
        "file_name": "bloodmnist_128.npz",
        "num_classes": 8,
    },
    "tissuemnist_128": {
        "file_name": "tissuemnist_128.npz",
        "num_classes": 8,
    },
}


def get_supported_model_names():
    return list(MODEL_CONFIGS.keys())


def get_model_config(variant):
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported CvT variant: {variant}")
    return dict(MODEL_CONFIGS[variant])


def get_default_pretrain_config():
    return dict(DEFAULT_PRETRAIN_CONFIG)


def get_default_classification_config():
    return dict(DEFAULT_CLASSIFICATION_CONFIG)


def get_default_segmentation_config():
    return dict(DEFAULT_SEGMENTATION_CONFIG)


def get_medmnist_classification_config(source):
    if source not in MEDMNIST_CLASSIFICATION_CONFIGS:
        raise ValueError(f"Unsupported classification source: {source}")
    return dict(MEDMNIST_CLASSIFICATION_CONFIGS[source])


def get_supported_classification_sources():
    return list(MEDMNIST_CLASSIFICATION_CONFIGS.keys())
