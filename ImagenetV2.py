from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from os.path import join
from os import cpu_count

from .utils import Accuracy

import torch


class IV2:
    subdirs = ["all", "matched-frequency", "threshold", "top-images"]

    def __init__(self, root=".", subset="top-images", transform=None):
        assert subset in IV2.subdirs
        self.subset = subset

        if subset == "all":
            for s in IV2.subdirs[1:]:
                setattr(self, s, ImageFolder(join(root, s), transform))
        else:
            setattr(self, subset, ImageFolder(join(root, subset), transform))

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
        correct = 0
        total = 0

        if subset == "all":
            subset_correct = {s: 0 for s in IV2.subdirs[1:]}
            subset_total = {s: 0 for s in IV2.subdirs[1:]}
            subset_acc = {s: 0.0 for s in IV2.subdirs[1:]}

            for s in IV2.subdirs[1:]:
                dataset = ImageFolder(join(root, s), transforms)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,  # type: ignore
                    pin_memory=True,
                )

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
            dataset = ImageFolder(join(root, subset), transforms)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,  # type: ignore
                pin_memory=True,
            )
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
