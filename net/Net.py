#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def rgb_to_ycbcr(image):
    if image.shape[1] == 1:
        return image
    elif image.shape[1] == 3:
        r, g, b = torch.chunk(image, 3, dim=1)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
        return torch.cat([y, cb, cr], dim=1)
    else:
        raise ValueError(f"Unexpected number of channels: {image.shape[1]}")


def ycbcr_to_rgb(image):
    if image.shape[1] == 1:
        return image
    elif image.shape[1] == 3:
        y, cb, cr = torch.chunk(image, 3, dim=1)
        r = y + 1.402 * (cr - 0.5)
        g = y - 0.34414 * (cb - 0.5) - 0.71414 * (cr - 0.5)
        b = y + 1.772 * (cb - 0.5)
        return torch.cat([r, g, b], dim=1)
    else:
        raise ValueError(f"Unexpected number of channels: {image.shape[1]}")


def enhance_low_variance_regions(image, window_size=65, variance_threshold=0.2, detail_scale=10.0,
                                 color_balance_strength=0.5):
    image = image.float() / 255.0

    if image.dim() == 3:
        image = image.unsqueeze(0)

    num_channels = image.shape[1]

    if num_channels == 1:
        enhanced = enhance_channel(image.squeeze(1), window_size, variance_threshold, detail_scale)
        enhanced = enhanced.unsqueeze(1)
    elif num_channels == 3:
        ycbcr = rgb_to_ycbcr(image)
        y, cb, cr = torch.chunk(ycbcr, 3, dim=1)
        y_enhanced = enhance_channel(y.squeeze(1), window_size, variance_threshold, detail_scale)

        cb = (cb - 0.5) * 0.9 + 0.5
        cr = (cr - 0.5) * 0.9 + 0.5

        ycbcr_enhanced = torch.cat([y_enhanced.unsqueeze(1), cb, cr], dim=1)
        enhanced = ycbcr_to_rgb(ycbcr_enhanced)
    else:
        raise ValueError(f"Unexpected number of channels: {num_channels}")

    enhanced = color_balance(enhanced, strength=color_balance_strength)

    enhanced = adjust_contrast(enhanced, factor=1.5)

    return enhanced


def enhance_channel(channel, window_size, variance_threshold, detail_scale):
    mean = F.avg_pool2d(channel.unsqueeze(1), window_size, stride=1, padding=window_size // 2).squeeze(1)
    mean_sq = F.avg_pool2d(channel.unsqueeze(1) ** 2, window_size, stride=1, padding=window_size // 2).squeeze(1)
    local_variance = mean_sq - mean ** 2

    max_variance = torch.max(local_variance)
    high_var_mask = local_variance > variance_threshold * max_variance
    low_var_mask = ~high_var_mask

    kernel = torch.tensor([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=torch.float32, device=channel.device).unsqueeze(0).unsqueeze(0)

    high_freq = F.conv2d(channel.unsqueeze(1), kernel, padding=1).squeeze(1)

    adaptive_scale = detail_scale * (1 - local_variance / max_variance)
    enhanced_high_freq = channel + adaptive_scale * (high_freq - channel)

    result = torch.where(low_var_mask, enhanced_high_freq, channel)

    result = (result - result.min()) / (result.max() - result.min())

    return result


def color_balance(image, strength=0.5):
    mean = torch.mean(image, dim=[2, 3], keepdim=True)
    balanced = image * (1 - strength) + (image / (mean + 1e-6)) * strength
    return torch.clamp(balanced, 0, 1)


def adjust_contrast(image, factor=1.5):
    mean = torch.mean(image, dim=[2, 3], keepdim=True)
    return torch.clamp((image - mean) * factor + mean, 0, 1)


class Net(torch.nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, out_channel, 7, 1, 3),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 7, 1, 3),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 7, 1, 3),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 7, 1, 3),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(inplace=True)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel, out_channel, 7, 1, 3),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        data = self.conv1(x)
        x = enhance_low_variance_regions(data)

        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        data = self.conv2(x)
        x = enhance_low_variance_regions(data)


        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        data = self.conv3(x)
        x = enhance_low_variance_regions(data)

        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        data = self.conv4(x)
        x = enhance_low_variance_regions(data)

        b,c,h,w = x.shape
        x_sort, idx_h = x[:,:c//2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:,:c//2] = x_sort
        data = self.final(x)
        x = enhance_low_variance_regions(data)
        return x
