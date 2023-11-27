from .utils import thousandK_to_ImagenetA200 as thousand_k_to_200
from .utils import Accuracy

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import os

num_workers = os.cpu_count()
num_workers = max(1, num_workers) if num_workers is not None else 1

indices_in_1k = [k for k in thousand_k_to_200 if thousand_k_to_200[k] != -1]

# Reverse mapping from 200-class label space to 1000-class label space
mapping_200_to_1000 = {v: k for k, v in enumerate(indices_in_1k)}


class ImagenetA(
    Dataset,
):
    def __init__(self, dataset_dir, transform=None):
        self._dataset_dir = dataset_dir
        self._transform = transform
        self.images = []
        self.labels = []
        self.label_names = {}

        self._load_data()

    def _load_data(self):
        for _, (dirpath, dirnames, filenames) in enumerate(os.walk(self._dataset_dir)):
            # Skip the root directory
            if dirpath == self._dataset_dir:
                continue

            for filename in filenames:
                if filename.lower().endswith(("png", "jpg", "jpeg")):
                    self.images.append(os.path.join(dirpath, filename))
                    label = dirpath.split(os.sep)[-1]
                    if label not in self.label_names:
                        self.label_names[label] = len(self.label_names)
                    self.labels.append(self.label_names[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image, label_200 = Image.open(img_path).convert("RGB"), self.labels[idx]
        label_1000 = mapping_200_to_1000.get(
            label_200, -1
        )  # Default to -1 if not found

        if label_1000 == -1:
            raise ValueError(
                f"Label {label_200} in 200-class space has no corresponding label in 1000-class space"
            )

        if self._transform:
            image = self._transform(image)

        return image, label_1000

    @staticmethod
    def eval_model(
        model,
        root=".",
        device=torch.device("cpu" if not torch.cuda.is_available() else "cuda"),
        transforms=None,
        batch_size=128,
        top5=False,
    ):
        model.eval()
        model.to(device)

        dataset = ImagenetA(root, transform=transforms)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers  # type: ignore
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                if top5:
                    correct_batch, total_batch = Accuracy._top5(outputs, labels)
                else:
                    correct_batch, total_batch = Accuracy._top1(outputs, labels)

                total += total_batch
                correct += correct_batch

        return correct / total
