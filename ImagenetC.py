from .utils import thousandK_wnids as all_wnids

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

import os
import h5py
import torch


wnid_to_class = {wnid: i for i, wnid in enumerate(all_wnids)}


class ImFolder(Dataset):
    def __init__(self, path, transform=None):
        self.file = h5py.File(f"{path}.hdf5", "r")
        self.labels = sorted(self.get_folders())
        self.transform = transform

    def get_folders(self):
        classes = []
        for name in self.file:
            item = self.file[name]
            if isinstance(item, h5py.Dataset):
                classes.append(name)
        return classes

    def __len__(self):
        return len(self.labels) * 50

    def __getitem__(self, index):
        label_idx = index // 50
        index = index % 50
        img = self.file["/" + self.labels[label_idx]][index]  # type: ignore

        if self.transform is not None:
            img = self.transform(img)
        return img, wnid_to_class[self.labels[label_idx]]


class IC(
    Dataset,
):
    subsets = {
        "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
        "digital": ["contrast", "elastic_transform", "jpeg_compression", "pixelate"],
        "extra": ["gaussian_blur", "saturate", "spatter", "speckle_noise"],
        "noise": ["gaussian_noise", "impulse_noise", "shot_noise"],
        "weather": ["brightness", "fog", "frost", "snow"],
    }

    def __init__(
        self,
        root=".",
        subset="all",
        subsubset="all",
        noise=list(range(1, 6)),
        transform=None,
    ):
        super().__init__()
        assert subset in IC.subsets or subset == "all"
        assert subsubset in IC.subsets[subset] or subsubset == "all"
        for i in noise:
            assert 0 < i < 6
        if subset == "all":
            assert subsubset == "all"

        self.root = root
        self.transform = transform

        self.subset = subset
        self.subsubset = subsubset
        self.noise = noise

        if subset == "all":
            self.dataloaders = dict()
            for s in IC.subsets:
                self.dataloaders[s] = dict()

                for ss in IC.subsets[s]:
                    self.dataloaders[s][ss] = dict()
                    for n in noise:
                        path = os.path.join(root, s, ss, n)
                        self.dataloaders[s][ss][n] = ImFolder(path, transform)

        else:
            self.dataloaders = dict()
            if subsubset == "all":
                for ss in IC.subsets[subset]:
                    self.dataloaders[ss] = dict()
                    for n in noise:
                        path = os.path.join(root, subset, ss, n)
                        self.dataloaders[ss][n] = ImFolder(path, transform)
            else:
                self.dataloaders[subsubset] = dict()
                for n in noise:
                    path = os.path.join(root, subset, subsubset, n)
                    self.dataloaders[subsubset][n] = ImFolder(path, transform)

    def __len__(self):
        if self._len is None:
            if self.subset == "all":
                self._len = sum(
                    [
                        len(self.dataloaders[s][ss][n])
                        for s in IC.subsets
                        for ss in IC.subsets[s]
                        for n in self.noise
                    ]
                )
            elif self.subsubset == "all":
                self._len = sum(
                    [
                        len(self.dataloaders[ss][n])
                        for ss in IC.subsets[self.subset]
                        for n in self.noise
                    ]
                )
            else:
                self._len = sum(
                    [len(self.dataloaders[self.subsubset][n]) for n in self.noise]
                )
        return self._len

    def __getitem__(self, index):
        cumulative_index = 0
        for s in self.dataloaders:
            for ss in self.dataloaders[s]:
                for n in self.dataloaders[s][ss]:
                    if index < cumulative_index + len(self.dataloaders[s][ss][n]):
                        x, y = self.dataloaders[s][ss][n].__getitem__(
                            index - cumulative_index
                        )
                        return (
                            x,
                            wnid_to_class[self.dataloaders[s][ss][n].idx_to_wnid[y]],
                        )
                    cumulative_index += len(self.dataloaders[s][ss][n])

        raise IndexError("Index out of range")

    @staticmethod
    def eval_model(
        model,
        root=".",
        subset="all",
        subsubset="all",
        noise=list(range(1, 6)),
        device=torch.device("cpu" if not torch.cuda.is_available() else "cuda"),
        transforms=None,
        batch_size=128,
    ):
        pass


# class_name = idx_to_class[label.item()]
