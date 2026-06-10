from io import BytesIO
from pathlib import Path
import gc
import random

import numpy as np
import pyarrow.parquet as parquet
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torchvision.transforms import functional as transform_functional
from PIL import Image


MEDMNIST_PATHOLOGY_SOURCES = {
    "pathmnist_128": "pathmnist_128.npz",
    "bloodmnist_128": "bloodmnist_128.npz",
    "tissuemnist_128": "tissuemnist_128.npz",
}

PATHOLOGY_SOURCE_ALIASES = {
    "medmnist_pathology_128": ["pathmnist_128", "bloodmnist_128", "tissuemnist_128"],
    "all": ["pathmnist_128", "bloodmnist_128", "tissuemnist_128", "pannuke"],
}


def parse_pretrain_sources(source_text):
    requested_sources = []
    for source_name in source_text.split(","):
        source_name = source_name.strip()
        if not source_name:
            continue
        requested_sources.extend(PATHOLOGY_SOURCE_ALIASES.get(source_name, [source_name]))
    return requested_sources


def build_pathology_pretrain_dataset(data_dir, source_text, split, image_size, max_samples_per_source=0):
    data_dir = Path(data_dir)
    requested_sources = parse_pretrain_sources(source_text)
    datasets = []

    for source_name in requested_sources:
        if source_name in MEDMNIST_PATHOLOGY_SOURCES:
            dataset_path = data_dir / "medmnist_pathology_128" / MEDMNIST_PATHOLOGY_SOURCES[source_name]
            cache_dir = data_dir / "cache" / "medmnist_pathology_128"
            datasets.append(MedMnistNpzImageDataset(
                dataset_path,
                split,
                image_size,
                source_name,
                cache_dir,
                max_samples_per_source,
            ))
            continue
        if source_name == "pannuke":
            dataset_dir = data_dir / "pannuke"
            datasets.append(PanNukeImageDataset(dataset_dir, split, image_size, max_samples_per_source))
            continue
        raise ValueError(f"Unsupported pathology source: {source_name}")

    if not datasets:
        raise ValueError("At least one pathology source is required")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


class PathologyImageTransform:
    def __init__(self, image_size, training=True):
        self.image_size = image_size
        self.training = training

    def __call__(self, image):
        image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        if self.training:
            if random.random() < 0.5:
                image = transform_functional.hflip(image)
            if random.random() < 0.5:
                image = transform_functional.vflip(image)
            rotation_count = random.randint(0, 3)
            if rotation_count:
                image = image.rotate(90 * rotation_count)

        return transform_functional.to_tensor(image)


class MedMnistNpzImageDataset(Dataset):
    def __init__(self, dataset_path, split, image_size, source_name, cache_dir, max_samples=0):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.source_name = source_name
        self.transform = PathologyImageTransform(image_size, training=split == "train")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Missing MedMNIST dataset: {self.dataset_path}")

        images = load_or_create_medmnist_image_cache(self.dataset_path, split, source_name, cache_dir)
        if max_samples:
            images = images[:max_samples]
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, sample_index):
        image = Image.fromarray(np.asarray(self.images[sample_index]))
        return self.transform(image)


class MedMnistClassificationDataset(Dataset):
    def __init__(self, dataset_path, split, image_size, source_name, cache_dir, max_samples=0):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.source_name = source_name
        self.transform = PathologyImageTransform(image_size, training=split == "train")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Missing MedMNIST dataset: {self.dataset_path}")

        images = load_or_create_medmnist_image_cache(self.dataset_path, split, source_name, cache_dir)
        labels = load_medmnist_labels(self.dataset_path, split)
        if max_samples:
            images = images[:max_samples]
            labels = labels[:max_samples]

        self.images = images
        self.labels = labels.reshape(-1).astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, sample_index):
        image = Image.fromarray(np.asarray(self.images[sample_index]))
        image = self.transform(image)
        label = torch.tensor(self.labels[sample_index], dtype=torch.long)
        return image, label


def load_or_create_medmnist_image_cache(dataset_path, split, source_name, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{source_name}_{split}_images.npy"

    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r")

    print(f"creating MedMNIST image cache: {cache_path}", flush=True)
    image_key = f"{split}_images"
    archive = np.load(dataset_path)
    try:
        if image_key not in archive.files:
            raise ValueError(f"{dataset_path} does not contain {image_key}")
        images = archive[image_key]
        np.save(cache_path, images)
    finally:
        archive.close()

    del images
    gc.collect()
    return np.load(cache_path, mmap_mode="r")


def load_medmnist_labels(dataset_path, split):
    label_key = f"{split}_labels"
    archive = np.load(dataset_path)
    try:
        if label_key not in archive.files:
            raise ValueError(f"{dataset_path} does not contain {label_key}")
        return archive[label_key]
    finally:
        archive.close()


def split_pannuke_fold_records(records, split):
    if split == "val":
        return records[:len(records) // 2]
    if split == "test":
        return records[len(records) // 2:]
    return records


class PanNukeImageDataset(Dataset):
    def __init__(self, dataset_dir, split, image_size, max_samples=0):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = PathologyImageTransform(image_size, training=split == "train")

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Missing PanNuke dataset directory: {self.dataset_dir}")

        parquet_paths = get_pannuke_parquet_paths(self.dataset_dir, split)
        if not parquet_paths:
            raise FileNotFoundError(f"No PanNuke parquet files found for {split} in {self.dataset_dir}")

        image_records = []
        for parquet_path in parquet_paths:
            table = parquet.read_table(parquet_path, columns=["image"])
            image_records.extend(table.column("image").to_pylist())

        image_records = split_pannuke_fold_records(image_records, split)
        if max_samples:
            image_records = image_records[:max_samples]
        self.image_records = image_records

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, sample_index):
        image_bytes = self.image_records[sample_index]["bytes"]
        image = Image.open(BytesIO(image_bytes))
        return self.transform(image)


class PanNukeSegmentationDataset(Dataset):
    def __init__(self, dataset_dir, split, image_size, max_samples=0):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_size = image_size
        self.training = split == "train"

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Missing PanNuke dataset directory: {self.dataset_dir}")

        parquet_paths = get_pannuke_parquet_paths(self.dataset_dir, split)
        if not parquet_paths:
            raise FileNotFoundError(f"No PanNuke parquet files found for {split} in {self.dataset_dir}")

        records = []
        for parquet_path in parquet_paths:
            table = parquet.read_table(parquet_path, columns=["image", "type_map"])
            images = table.column("image").to_pylist()
            masks = table.column("type_map").to_pylist()
            records.extend(zip(images, masks))

        records = split_pannuke_fold_records(records, split)
        if max_samples:
            records = records[:max_samples]
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, sample_index):
        image_record, mask_record = self.records[sample_index]
        image = Image.open(BytesIO(image_record["bytes"])).convert("RGB")
        mask = Image.open(BytesIO(mask_record["bytes"]))

        image, mask = transform_segmentation_pair(image, mask, self.image_size, self.training)
        return image, mask


def transform_segmentation_pair(image, mask, image_size, training):
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
        mask = mask.resize((image_size, image_size), Image.NEAREST)

    if training:
        if random.random() < 0.5:
            image = transform_functional.hflip(image)
            mask = transform_functional.hflip(mask)
        if random.random() < 0.5:
            image = transform_functional.vflip(image)
            mask = transform_functional.vflip(mask)
        rotation_count = random.randint(0, 3)
        if rotation_count:
            image = image.rotate(90 * rotation_count)
            mask = mask.rotate(90 * rotation_count)

    image = transform_functional.to_tensor(image)
    mask = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
    return image, mask


def collate_pathology_images(images):
    return torch.stack(images, dim=0)


def collate_classification_samples(samples):
    images, labels = zip(*samples)
    return torch.stack(images, dim=0), torch.stack(labels, dim=0)


def collate_segmentation_samples(samples):
    images, masks = zip(*samples)
    return torch.stack(images, dim=0), torch.stack(masks, dim=0)


def get_pannuke_parquet_paths(dataset_dir, split):
    if split == "train":
        return sorted(dataset_dir.glob("fold1-*.parquet")) + sorted(dataset_dir.glob("fold2-*.parquet"))
    if split in ["val", "test"]:
        return sorted(dataset_dir.glob("fold3-*.parquet"))
    raise ValueError(f"Unsupported PanNuke split: {split}")
