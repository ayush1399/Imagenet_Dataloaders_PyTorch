from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset
from apply_patch import ApplyPatch
from PIL import Image

import pickle
import gzip
import os


class ImagenetPatch(Dataset):
    def __init__(self, root="./", patch="all"):
        self._root = root
        self.__get_image_ids()
        self.__load_patches()
        self.__patch = "all"

        self._preprocess = Compose([Resize(256), CenterCrop(224), ToTensor()])
        self._normalizer = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self._clean_transform = Compose([self._preprocess, self._normalizer])

        self._patch_transforms = []
        for i in range(len(self._patches)):
            self._patch_transforms[i] = Compose(
                [
                    self._preprocess,
                    ApplyPatch(
                        patch=self._patches[i],
                        patch_size=self._info["patch_size"],
                        translation_range=(0.2, 0.2),
                        rotation_range=(-45, 45),
                        scale_range=(0.7, 1),
                    ),
                    self._normalizer,
                ]
            )

    def __get_image_ids(self):
        with open(os.path.join(self._root, "imagenet_test_image_ids.txt"), "r") as f:
            images = f.read().splitlines()
            self._image_files = [i.split("/")[1] for i in images]
            self._image_labels = [i.split("/")[0] for i in images]
            self.__num_patches = len(self._image_files)

    def __load_patches(self):
        with gzip.open(os.path.join(self._root, "imagenet_patch.gz"), "rb") as f:
            imagenet_patch = pickle.load(f)
        patches, targets, info = imagenet_patch
        self._patches = patches
        self._patch_targets = targets
        self._info = info

    def __getitem__(self, idx):
        if self.__patch == "all":
            index = idx % len(self._image_labels)
            image_root = os.path.join("ILSVRC2012", "val", self._image_files[index])
            image_label = self._image_labels[index]
            patch_idx = idx % self.__num_patches

        else:
            image_root = os.path.join("ILSVRC2012", "val", self._image_files[idx])
            image_label = self._image_labels[idx]
            patch_idx = self.__patch

        image = Image.open(os.path.join(image_root))
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_clean = self._clean_transform(image)
        image_patch = self._patch_transforms[patch_idx](image)
        patch_label = self._patch_targets[patch_idx]

        return (image_clean, image_patch), (image_label, patch_label)
