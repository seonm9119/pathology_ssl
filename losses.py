import torch
from torch import nn
import torch.nn.functional as F


def masked_l1_loss(predicted_images, target_images, masks):
    loss = torch.abs(predicted_images - target_images) * masks
    return loss.sum() / masks.sum().clamp_min(1.0)


def total_variation_loss(images):
    height_loss = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]).mean()
    width_loss = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]).mean()
    return height_loss + width_loss


class InpaintingReconstructionLoss(nn.Module):
    def __init__(self, hole_weight=6.0, tv_weight=0.1):
        super().__init__()
        self.hole_weight = hole_weight
        self.tv_weight = tv_weight

    def forward(self, target_images, predicted_images, masks, return_parts=False):
        if masks.shape[1] != 1:
            masks = masks[:, :1]

        valid_masks = masks
        hole_masks = 1 - masks
        valid_loss = masked_l1_loss(predicted_images, target_images, valid_masks)
        hole_loss = masked_l1_loss(predicted_images, target_images, hole_masks)
        composite_images = target_images * valid_masks + predicted_images * hole_masks
        tv_loss = total_variation_loss(composite_images)
        total_loss = valid_loss + self.hole_weight * hole_loss + self.tv_weight * tv_loss

        if not return_parts:
            return total_loss

        return {
            "loss": total_loss,
            "valid_loss": valid_loss.detach(),
            "hole_loss": hole_loss.detach(),
            "tv_loss": tv_loss.detach(),
        }


def multiclass_dice_loss(logits, targets, num_classes, ignore_index=None, epsilon=1e-6):
    probabilities = logits.softmax(dim=1)
    valid_masks = torch.ones_like(targets, dtype=torch.bool)
    if ignore_index is not None:
        valid_masks = targets != ignore_index
        targets = targets.clamp_min(0)

    target_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    valid_masks = valid_masks.unsqueeze(1)
    probabilities = probabilities * valid_masks
    target_one_hot = target_one_hot * valid_masks

    intersection = (probabilities * target_one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    dice_score = (2 * intersection + epsilon) / (denominator + epsilon)
    return 1 - dice_score.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets, return_parts=False):
        cross_entropy_loss = self.cross_entropy(logits, targets)
        dice_loss = multiclass_dice_loss(logits, targets, self.num_classes)
        total_loss = cross_entropy_loss + self.dice_weight * dice_loss

        if not return_parts:
            return total_loss

        return {
            "loss": total_loss,
            "cross_entropy_loss": cross_entropy_loss.detach(),
            "dice_loss": dice_loss.detach(),
        }
