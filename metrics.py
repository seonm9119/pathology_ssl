import torch


def classification_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    correct_count = (predictions == labels).sum().item()
    return correct_count / max(1, labels.numel())


def segmentation_pixel_accuracy(logits, masks):
    predictions = logits.argmax(dim=1)
    correct_count = (predictions == masks).sum().item()
    return correct_count / max(1, masks.numel())


def segmentation_mean_iou(logits, masks, num_classes):
    predictions = logits.argmax(dim=1)
    class_scores = []

    for class_index in range(num_classes):
        predicted_class = predictions == class_index
        target_class = masks == class_index
        union = (predicted_class | target_class).sum().item()
        if union == 0:
            continue
        intersection = (predicted_class & target_class).sum().item()
        class_scores.append(intersection / union)

    if not class_scores:
        return 0.0
    return sum(class_scores) / len(class_scores)
