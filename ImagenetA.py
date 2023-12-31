from .utils import thousandK_wnids
from .utils import Accuracy

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import os

mapping_200_to_1000 = {wnid: i for i, wnid in enumerate(sorted(thousandK_wnids))}


class ImagenetA(
    Dataset,
):
    def __init__(self, dataset_dir, transform=None):
        self._dataset_dir = dataset_dir
        self._transform = transform
        self.images = []
        self.labels = []
        self.label_names = {}
        self.label_map = {}

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

            for k, v in self.label_names.items():
                self.label_map[v] = k

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image, label_200 = (
            Image.open(img_path).convert("RGB"),
            self.label_map[self.labels[idx]],
        )
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
    def pretty_print_acc(acc, args, cfg):
        print("=" + "*=" * 12)
        print(f"MODEL: {args.model: <10} DATASET: {args.dataset}")
        print(f"Config params: {getattr(cfg.data, args.dataset).params}")
        if args.top5:
            print(f"Top-5 Accuracy: {acc*100:.2f}")
        else:
            print(f"Top-1 Accuracy: {acc*100:.2f}")
        print("=" + "*=" * 12)
        print()

    @staticmethod
    def eval_model(
        model,
        root=".",
        device=torch.device("cpu" if not torch.cuda.is_available() else "cuda"),
        transforms=None,
        batch_size=128,
        top5=False,
        num_workers=1,
    ):
        model.eval()
        model = model.to(device)

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
