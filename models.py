import torch
from torch import nn
import torch.nn.functional as F

try:
    from .config import get_model_config
except ImportError:
    from config import get_model_config


def to_pair(size):
    if isinstance(size, tuple):
        return size
    return (size, size)


def get_cvt_stage_depths(variant):
    return get_model_config(variant)["stage_depths"]


def drop_path(features, drop_probability, training):
    if drop_probability == 0.0 or not training:
        return features

    keep_probability = 1 - drop_probability
    noise_shape = (features.shape[0],) + (1,) * (features.ndim - 1)
    random_tensor = features.new_empty(noise_shape).bernoulli_(keep_probability)
    return features.div(keep_probability) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_probability):
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, features):
        return drop_path(features, self.drop_probability, self.training)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, features):
        normalized_features = features.permute(0, 2, 3, 1)
        normalized_features = self.norm(normalized_features)
        return normalized_features.permute(0, 3, 1, 2).contiguous()


class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        kernel_size = to_pair(kernel_size)
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        self.slide_window_size = kernel_size[0] * kernel_size[1]

        nn.init.constant_(self.mask_conv.weight, 1.0)
        for mask_parameter in self.mask_conv.parameters():
            mask_parameter.requires_grad = False

    def forward(self, images, masks=None):
        if masks is None:
            masks = images.new_ones(images.shape[0], 1, images.shape[2], images.shape[3])
        if masks.shape[1] != 1:
            masks = masks[:, :1]

        masked_images = images * masks
        raw_output = self.input_conv(masked_images)

        with torch.no_grad():
            valid_pixel_count = self.mask_conv(masks)
            output_masks = (valid_pixel_count > 0).to(images.dtype)
            valid_ratio = self.slide_window_size / (valid_pixel_count + 1e-8)
            valid_ratio = valid_ratio * output_masks

        if self.input_conv.bias is None:
            output = raw_output * valid_ratio
        else:
            bias = self.input_conv.bias.view(1, -1, 1, 1)
            output = (raw_output - bias) * valid_ratio + bias
            output = output * output_masks

        return output, output_masks


class PartialConvTokenEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, patch_stride, patch_padding):
        super().__init__()
        self.partial_conv = PartialConv2d(in_channels, out_channels, patch_size, patch_stride, patch_padding)
        self.norm = ChannelLayerNorm(out_channels)

    def forward(self, features, masks):
        embedded_features, embedded_masks = self.partial_conv(features, masks)
        embedded_features = self.norm(embedded_features)
        return embedded_features, embedded_masks


class DepthwiseProjection(nn.Module):
    def __init__(self, channels, kernel_size, padding, stride):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, features):
        return self.projection(features)


class ConvAttention(nn.Module):
    def __init__(self, channels, num_heads, kernel_size=3, padding=1, stride_q=1, stride_kv=2, dropout=0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.scale = self.head_channels ** -0.5
        self.q_projection = DepthwiseProjection(channels, kernel_size, padding, stride_q)
        self.k_projection = DepthwiseProjection(channels, kernel_size, padding, stride_kv)
        self.v_projection = DepthwiseProjection(channels, kernel_size, padding, stride_kv)
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def reshape_to_heads(self, features):
        batch_size, channels, height, width = features.shape
        features = features.flatten(2).transpose(1, 2)
        features = features.reshape(batch_size, height * width, self.num_heads, self.head_channels)
        return features.permute(0, 2, 1, 3)

    def restore_from_heads(self, features, height, width):
        batch_size = features.shape[0]
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.reshape(batch_size, height * width, self.num_heads * self.head_channels)
        return features.transpose(1, 2).reshape(batch_size, self.num_heads * self.head_channels, height, width)

    def forward(self, features):
        query_features = self.q_projection(features)
        key_features = self.k_projection(features)
        value_features = self.v_projection(features)

        query_height, query_width = query_features.shape[-2:]
        query = self.reshape_to_heads(query_features)
        key = self.reshape_to_heads(key_features)
        value = self.reshape_to_heads(value_features)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        attended_features = attention @ value
        attended_features = self.restore_from_heads(attended_features, query_height, query_width)
        attended_features = self.output_projection(attended_features)
        return self.output_dropout(attended_features)


class ConvMlp(nn.Module):
    def __init__(self, channels, hidden_channels, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, features):
        return self.layers(features)


class CvtBlock(nn.Module):
    def __init__(self, channels, num_heads, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.attention_norm = ChannelLayerNorm(channels)
        self.attention = ConvAttention(channels, num_heads, dropout=drop_rate)
        self.drop_path = DropPath(drop_path_rate)
        self.mlp_norm = ChannelLayerNorm(channels)
        self.mlp = ConvMlp(channels, hidden_channels, dropout=drop_rate)

    def forward(self, features):
        features = features + self.drop_path(self.attention(self.attention_norm(features)))
        features = features + self.drop_path(self.mlp(self.mlp_norm(features)))
        return features


class CvtStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        num_heads,
        patch_size,
        patch_stride,
        patch_padding,
        mlp_ratio,
        drop_rate,
        drop_path_rates,
    ):
        super().__init__()
        self.embedding = PartialConvTokenEmbedding(in_channels, out_channels, patch_size, patch_stride, patch_padding)
        self.blocks = nn.ModuleList([
            CvtBlock(out_channels, num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
            for drop_path_rate in drop_path_rates
        ])

    def forward(self, features, masks):
        features, masks = self.embedding(features, masks)
        for block in self.blocks:
            features = block(features)
        return features, masks


class PathologyCvtEncoder(nn.Module):
    def __init__(self, input_channels=3, variant="cvt13", drop_path_rate=0.1):
        super().__init__()
        model_config = get_model_config(variant)
        stage_depths = model_config["stage_depths"]
        stage_channels = model_config["stage_channels"]
        stage_heads = model_config["stage_heads"]
        patch_sizes = model_config["patch_sizes"]
        patch_strides = model_config["patch_strides"]
        patch_paddings = model_config["patch_paddings"]
        mlp_ratios = model_config["mlp_ratios"]
        drop_rates = model_config["drop_rates"]
        drop_path_rate = model_config.get("drop_path_rate", drop_path_rate)
        total_depth = sum(stage_depths)
        drop_path_rates = torch.linspace(0, drop_path_rate, total_depth).tolist()

        stages = []
        input_stage_channels = [input_channels] + stage_channels[:-1]
        drop_path_index = 0
        for stage_index, stage_depth in enumerate(stage_depths):
            stage_drop_path_rates = drop_path_rates[drop_path_index:drop_path_index + stage_depth]
            drop_path_index += stage_depth
            stages.append(CvtStage(
                input_stage_channels[stage_index],
                stage_channels[stage_index],
                stage_depth,
                stage_heads[stage_index],
                patch_sizes[stage_index],
                patch_strides[stage_index],
                patch_paddings[stage_index],
                mlp_ratios[stage_index],
                drop_rates[stage_index],
                stage_drop_path_rates,
            ))

        self.stages = nn.ModuleList(stages)
        self.output_channels = stage_channels
        self.variant = variant

    def forward(self, images, masks=None):
        if masks is None:
            masks = images.new_ones(images.shape[0], 1, images.shape[2], images.shape[3])

        features = images
        stage_features = []
        stage_masks = []
        for stage in self.stages:
            features, masks = stage(features, masks)
            stage_features.append(features)
            stage_masks.append(masks)

        return stage_features, stage_masks


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, features, skip_features):
        features = F.interpolate(features, size=skip_features.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat([features, skip_features], dim=1)
        return self.fuse(features)


class CvtFeatureDecoder(nn.Module):
    def __init__(self, output_channels, input_channels=384):
        super().__init__()
        self.decoder_stage1 = DecoderBlock(input_channels, 192, 192)
        self.decoder_stage2 = DecoderBlock(192, 64, 64)
        self.output_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, output_channels, 3, padding=1),
        )

    def forward(self, stage_features, output_size):
        stage1_features, stage2_features, stage3_features = stage_features
        decoded_features = self.decoder_stage1(stage3_features, stage2_features)
        decoded_features = self.decoder_stage2(decoded_features, stage1_features)
        decoded_features = F.interpolate(decoded_features, size=output_size, mode="bilinear", align_corners=False)
        return self.output_head(decoded_features)


class PathologySslReconstructionModel(nn.Module):
    def __init__(self, input_channels=3, variant="cvt13"):
        super().__init__()
        self.encoder = PathologyCvtEncoder(input_channels=input_channels, variant=variant)
        self.decoder = CvtFeatureDecoder(output_channels=input_channels)

    def forward(self, images, masks=None, return_features=False):
        if masks is None:
            masks = images.new_ones(images.shape[0], 1, images.shape[2], images.shape[3])

        masked_images = images * masks
        stage_features, stage_masks = self.encoder(masked_images, masks)
        reconstructed_images = self.decoder(stage_features, images.shape[-2:])

        if return_features:
            return reconstructed_images, stage_features, stage_masks
        return reconstructed_images


class PathologyClassificationModel(nn.Module):
    def __init__(self, num_classes, input_channels=3, variant="cvt13"):
        super().__init__()
        self.encoder = PathologyCvtEncoder(input_channels=input_channels, variant=variant)
        self.norm = ChannelLayerNorm(384)
        self.head = nn.Linear(384, num_classes)

    def forward(self, images):
        stage_features, _ = self.encoder(images)
        final_features = self.norm(stage_features[-1])
        pooled_features = final_features.mean(dim=(2, 3))
        return self.head(pooled_features)

    def freeze_encoder(self):
        for encoder_parameter in self.encoder.parameters():
            encoder_parameter.requires_grad = False


class PathologySegmentationModel(nn.Module):
    def __init__(self, num_classes, input_channels=3, variant="cvt13"):
        super().__init__()
        self.encoder = PathologyCvtEncoder(input_channels=input_channels, variant=variant)
        self.decoder = CvtFeatureDecoder(output_channels=num_classes)

    def forward(self, images):
        stage_features, _ = self.encoder(images)
        return self.decoder(stage_features, images.shape[-2:])

    def freeze_encoder(self):
        for encoder_parameter in self.encoder.parameters():
            encoder_parameter.requires_grad = False


def build_pathology_cvt13_ssl(input_channels=3):
    return PathologySslReconstructionModel(input_channels=input_channels, variant="cvt13")


def build_pathology_cvt21_ssl(input_channels=3):
    return PathologySslReconstructionModel(input_channels=input_channels, variant="cvt21")


def build_pathology_cvt13_classifier(num_classes, input_channels=3):
    return PathologyClassificationModel(num_classes, input_channels=input_channels, variant="cvt13")


def build_pathology_cvt21_classifier(num_classes, input_channels=3):
    return PathologyClassificationModel(num_classes, input_channels=input_channels, variant="cvt21")


def build_pathology_cvt13_segmentation(num_classes, input_channels=3):
    return PathologySegmentationModel(num_classes, input_channels=input_channels, variant="cvt13")


def build_pathology_cvt21_segmentation(num_classes, input_channels=3):
    return PathologySegmentationModel(num_classes, input_channels=input_channels, variant="cvt21")
