from torch.utils.data import DataLoader, Dataset
from PIL import Image

from .utils import Accuracy

import torch
import os


class CustomImageFolder(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = []
        self.labels = []

        labels = os.listdir(directory)
        labels = [int(l) for l in labels]

        for label in labels:
            label = str(label)
            for image in os.listdir(os.path.join(directory, label)):
                self.images.append(os.path.join(directory, label, image))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, int(label)


class IV2:
    subdirs = ["all", "matched-frequency", "threshold", "top-images"]

    def __init__(self, root=".", subset="top-images", transform=None):
        assert subset in IV2.subdirs
        self.subset = subset

        if subset == "all":
            for s in IV2.subdirs[1:]:
                setattr(self, s, CustomImageFolder(os.path.join(root, s), transform))
        else:
            setattr(
                self, subset, CustomImageFolder(os.path.join(root, subset), transform)
            )

    @property
    def subsets(self):
        return IV2.subdirs[1:]

    def subdataset(self, subset):
        return getattr(self, subset)

    def __len__(self):
        if self.subset == "all":
            return sum(len(getattr(self, s)) for s in IV2.subdirs[1:])
        else:
            return len(getattr(self, self.subset))

    def __getitem__(self, index):
        if self.subset == "all":
            cumulative_index = 0
            for s in IV2.subdirs[1:]:
                dataset = getattr(self, s)
                if index < cumulative_index + len(dataset):
                    return dataset[index - cumulative_index]
                cumulative_index += len(dataset)
            raise IndexError("Index out of range")
        else:
            return getattr(self, self.subset)[index]

    @staticmethod
    def pretty_print_acc(acc, args, cfg):
        subset = getattr(cfg.data, args.dataset).params[0]
        if subset == "all":
            acc, subset_acc = acc
            print("=" + "*=" * 12)
            print(f"MODEL: {args.model: <10} DATASET: {args.dataset}")
            acc_typ = "Top-5" if args.top5 else "Top-1"
            print(f"ALL: {acc_typ} Acc: {acc * 100:.2f}%")
            for s in IV2.subdirs[1:]:
                print(f"{s.upper()}: {acc_typ} Acc: {subset_acc[s] * 100:.2f}%")
        else:
            print("=" + "*=" * 12)
            print(f"MODEL: {args.model: <10} DATASET: {args.dataset}")
            acc_typ = "Top-5" if args.top5 else "Top-1"
            print(f"{subset.upper()}: {acc_typ} Acc: {acc * 100:.2f}%")

    @staticmethod
    def eval_model(
        model,
        root=".",
        subset="top-images",
        device=torch.device("cpu" if not torch.cuda.is_available() else "cuda"),
        transforms=None,
        batch_size=128,
        top5=False,
        num_workers=1,
    ):
        model.eval()
        model = model.to(device)
        correct = 0
        total = 0

        if subset == "all":
            subset_correct = {s: 0 for s in IV2.subdirs[1:]}
            subset_total = {s: 0 for s in IV2.subdirs[1:]}
            subset_acc = {s: 0.0 for s in IV2.subdirs[1:]}

            for s in IV2.subdirs[1:]:
                dataset = CustomImageFolder(os.path.join(root, s), transforms)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,  # type: ignore
                    pin_memory=True,
                )

                with torch.no_grad():
                    for images, labels in dataloader:
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)

                        if top5:
                            correct_batch, total_batch = Accuracy._top5(outputs, labels)
                        else:
                            correct_batch, total_batch = Accuracy._top1(outputs, labels)

                        subset_correct[s] += correct_batch
                        subset_total[s] += total_batch

                    subset_acc[s] = subset_correct[s] / subset_total[s]

            for key in subset_acc:
                correct += subset_correct[key]
                total += subset_total[key]

            return correct / total, subset_acc

        else:
            dataset = CustomImageFolder(os.path.join(root, subset), transforms)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,  # type: ignore
                pin_memory=True,
            )
            with torch.no_grad():
                for images, labels in dataloader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)

                    if top5:
                        correct_batch, total_batch = Accuracy._top5(outputs, labels)
                    else:
                        correct_batch, total_batch = Accuracy._top1(outputs, labels)

                    correct += correct_batch
                    total += total_batch

                return correct / total
