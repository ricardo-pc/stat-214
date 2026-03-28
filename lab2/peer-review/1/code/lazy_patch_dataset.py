import numpy as np
import torch
from torch.utils.data import Dataset

class LazyPatchDataset(Dataset):
    def __init__(self, filepaths, patch_size=9, means=None, stds=None, max_patches=None):
        self.filepaths = sorted(filepaths)
        self.patch_size = patch_size
        self.pad = patch_size // 2

        self.images_long = []
        for fp in self.filepaths:
            npz_data = np.load(fp)
            key = list(npz_data.files)[0]
            data = npz_data[key]

            if data.shape[1] == 11:
                data = data[:, :-1]

            self.images_long.append(data)

        all_y = np.concatenate([img[:, 0] for img in self.images_long]).astype(int)
        all_x = np.concatenate([img[:, 1] for img in self.images_long]).astype(int)

        self.global_miny = all_y.min()
        self.global_maxy = all_y.max()
        self.global_minx = all_x.min()
        self.global_maxx = all_x.max()

        self.height = int(self.global_maxy - self.global_miny + 1)
        self.width = int(self.global_maxx - self.global_minx + 1)

        self.nchannels = self.images_long[0].shape[1] - 2

        images = []
        coords_per_image = []

        for img in self.images_long:
            y = img[:, 0].astype(int)
            x = img[:, 1].astype(int)

            y_rel = y - self.global_miny
            x_rel = x - self.global_minx

            image = np.zeros((self.nchannels, self.height, self.width), dtype=np.float32)

            for c in range(self.nchannels):
                image[c, y_rel, x_rel] = img[:, c + 2]

            images.append(image)
            coords_per_image.append(np.column_stack([y_rel, x_rel]))

        images = np.stack(images, axis=0)

        if means is None or stds is None:
            means = np.mean(images, axis=(0, 2, 3), keepdims=True)
            stds = np.std(images, axis=(0, 2, 3), keepdims=True)
            stds = np.where(stds < 1e-8, 1.0, stds)

        self.means = means.astype(np.float32)
        self.stds = stds.astype(np.float32)

        images = (images - self.means) / self.stds

        self.images = np.pad(
            images,
            ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)),
            mode="reflect",
        ).astype(np.float32)

        self.index_map = []
        for img_idx, coords in enumerate(coords_per_image):
            for y_rel, x_rel in coords:
                self.index_map.append((img_idx, int(y_rel), int(x_rel)))

        if max_patches is not None and len(self.index_map) > max_patches:
            rng = np.random.default_rng(214)
            keep_idx = rng.choice(len(self.index_map), size=max_patches, replace=False)
            self.index_map = [self.index_map[i] for i in keep_idx]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        return self.index_map[idx]


class PatchCollator:
    def __init__(self, images, patch_size=9):
        self.images = torch.from_numpy(images)
        self.patch_size = patch_size
        self.pad = patch_size // 2

    def __call__(self, batch):
        batch_size = len(batch)
        c = self.images.shape[1]
        patches = torch.empty(
            (batch_size, c, self.patch_size, self.patch_size),
            dtype=self.images.dtype,
        )

        for i, (img_idx, y_rel, x_rel) in enumerate(batch):
            y_pad = y_rel + self.pad
            x_pad = x_rel + self.pad
            patches[i] = self.images[
                img_idx,
                :,
                y_pad - self.pad : y_pad + self.pad + 1,
                x_pad - self.pad : x_pad + self.pad + 1,
            ]

        return patches