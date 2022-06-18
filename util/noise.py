import numpy as np
import torch
import torchvision.transforms.functional as tvisf
from PIL import Image


def apply_noise(x_crop, noise, noise_mag, max_val=1., frame_num=None):
    if noise == "uniform":
        image_shape = x_crop.shape
        image_max, image_min = x_crop.max(), x_crop.min()
        noise = (np.random.uniform(size=image_shape) * noise_mag).astype(np.float32) - (noise_mag // 2)
        x_crop = (torch.from_numpy(noise).to(x_crop.device) + x_crop).clamp(image_min, image_max)
    elif noise == "rand_uniform":
        if np.random.rand() > 0.5:
            image_shape = x_crop.shape
            image_max, image_min = x_crop.max(), x_crop.min()
            noise = (np.random.uniform(size=image_shape) * noise_mag).astype(np.float32) - (noise_mag // 2)
            x_crop = (torch.from_numpy(noise).to(x_crop.device) + x_crop).clamp(image_min, image_max)

    elif noise == "gaussian":
        image_shape = x_crop.shape
        image_max, image_min = x_crop.max(), x_crop.min()
        noise = torch.randn(size=image_shape, dtype=x_crop.dtype, device=x_crop.device) * noise_mag
        x_crop = (noise + x_crop).clamp(image_min, image_max)
    elif noise == "rand_gaussian":
        if np.random.rand() > 0.5:
            image_shape = x_crop.shape
            image_max, image_min = x_crop.max(), x_crop.min()
            noise = torch.randn(size=image_shape, dtype=x_crop.dtype, device=x_crop.device) * noise_mag
            x_crop = (noise + x_crop).clamp(image_min, image_max)
    elif noise == "gamma":
        image_shape = image.shape
        noise = (np.random.uniform() / 2) - 0.5
        image = np.asarray(tvisf.adjust_hue(Image.fromarray(image), 0))
    elif noise == "invert_color":
        x_crop = max_val - x_crop
    elif noise == "rand_invert_color":
        if np.random.rand() > 0.5:
             x_crop = max_val - x_crop
    elif noise == "flipv":
        x_crop = torch.flip(x_crop, (3,))
    elif noise == "fix_occlusion":
        # if frame_num > 1 and frame_num % 10 != 0 and frame_num % 10 != 1:
        # # if frame_num > 0 and frame_num % 10 != 0:
        #     return x_crop
        # else:
        #     x_crop = torch.zeros_like(x_crop)
        image_shape = x_crop.shape
        min_width = 256 / 2
        r_or_c = np.random.rand() > 0.5
        if r_or_c:
            coord = np.random.randint(low=0, high=image_shape[3] - min_width)
            width = np.random.randint(low=2, high=image_shape[3] - coord)
            width = width // 2
            patch = x_crop[:, :, :, coord: coord + width]
            patch_shape = patch.shape
            patch = patch.flatten()
            patch = patch[torch.randperm(patch.size()[0])]
            patch = patch.view(patch_shape)
            x_crop[:, :, :, coord: coord + width] = patch
        else:
            coord = np.random.randint(low=0, high=image_shape[2] - min_width)
            width = np.random.randint(low=2, high=image_shape[2] - coord)
            width = width // 2
            patch = x_crop[:, :, coord: coord + width]
            patch_shape = patch.shape
            patch = patch.flatten()
            patch = patch[torch.randperm(patch.size()[0])]
            patch = patch.view(patch_shape)
            x_crop[:, :, coord: coord + width] = patch
    elif noise == "occlusion":
        image_shape = x_crop.shape
        min_width = 10
        r_or_c = np.random.rand() > 0.5
        if r_or_c:
            coord = np.random.randint(low=0, high=image_shape[3] - min_width)
            width = np.random.randint(low=2, high=image_shape[3] - coord)
            width = width // 2
            patch = x_crop[:, :, :, coord: coord + width]
            patch_shape = patch.shape
            patch = patch.flatten()
            patch = patch[torch.randperm(patch.size()[0])]
            patch = patch.view(patch_shape)
            x_crop[:, :, :, coord: coord + width] = patch
        else:
            coord = np.random.randint(low=0, high=image_shape[2] - min_width)
            width = np.random.randint(low=2, high=image_shape[2] - coord)
            width = width // 2  
            patch = x_crop[:, :, coord: coord + width]
            patch_shape = patch.shape
            patch = patch.flatten()
            patch = patch[torch.randperm(patch.size()[0])]
            patch = patch.view(patch_shape)
            x_crop[:, :, coord: coord + width] = patch
    elif noise == "occlusion_inf":
        image_shape = x_crop.shape
        min_width = 10
        r_or_c = np.random.rand() > 0.5
        if r_or_c:
            coord = np.random.randint(low=0, high=image_shape[1] - min_width)
            width = np.random.randint(low=2, high=image_shape[1] - coord)
            width = width // 2
            patch = x_crop[:, coord: coord + width]
            patch_shape = patch.shape
            patch = patch.flatten()
            patch = patch[np.random.permutation(patch.size)]
            patch = patch.reshape(patch_shape)
            x_crop[:, coord: coord + width] = patch
        else:
            coord = np.random.randint(low=0, high=image_shape[0] - min_width)
            width = np.random.randint(low=2, high=image_shape[0] - coord)
            width = width // 2
            patch = x_crop[coord: coord + width]
            patch_shape = patch.shape
            patch = patch.flatten()
            patch = patch[np.random.permutation(patch.size)]
            patch = patch.reshape(patch_shape)
            x_crop[coord: coord + width] = patch
    else:
        raise NotImplementedError(noise)
    return x_crop