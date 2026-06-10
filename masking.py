import torch
import torch.nn.functional as F


def generate_random_patch_masks(images, patch_size=16, mask_ratio=0.45):
    batch_size, _, image_height, image_width = images.shape
    grid_height = max(1, image_height // patch_size)
    grid_width = max(1, image_width // patch_size)
    token_count = grid_height * grid_width
    masked_token_count = max(1, int(token_count * mask_ratio))

    patch_masks = images.new_ones(batch_size, 1, token_count)
    for sample_index in range(batch_size):
        masked_indices = torch.randperm(token_count, device=images.device)[:masked_token_count]
        patch_masks[sample_index, 0, masked_indices] = 0

    patch_masks = patch_masks.reshape(batch_size, 1, grid_height, grid_width)
    return F.interpolate(patch_masks, size=(image_height, image_width), mode="nearest")


def generate_random_block_masks(images, min_block_ratio=0.15, max_block_ratio=0.35):
    batch_size, _, image_height, image_width = images.shape
    masks = images.new_ones(batch_size, 1, image_height, image_width)

    for sample_index in range(batch_size):
        block_height_ratio = torch.empty(1, device=images.device).uniform_(min_block_ratio, max_block_ratio).item()
        block_width_ratio = torch.empty(1, device=images.device).uniform_(min_block_ratio, max_block_ratio).item()
        block_height = max(1, int(image_height * block_height_ratio))
        block_width = max(1, int(image_width * block_width_ratio))
        top = torch.randint(0, image_height - block_height + 1, (1,), device=images.device).item()
        left = torch.randint(0, image_width - block_width + 1, (1,), device=images.device).item()
        masks[sample_index, :, top:top + block_height, left:left + block_width] = 0

    return masks


def generate_pretrain_masks(images, mask_style="patch", patch_size=16, mask_ratio=0.45):
    if mask_style == "patch":
        return generate_random_patch_masks(images, patch_size, mask_ratio)
    if mask_style == "block":
        return generate_random_block_masks(images)
    if mask_style == "mixed":
        patch_masks = generate_random_patch_masks(images, patch_size, mask_ratio)
        block_masks = generate_random_block_masks(images)
        return patch_masks * block_masks
    raise ValueError(f"Unsupported mask style: {mask_style}")
