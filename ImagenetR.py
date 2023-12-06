from .utils import imagenetR_wnids as imagenet_r_wnids
from .utils import thousandK_wnids as all_wnids

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from os import cpu_count

import torch

num_workers = cpu_count()
num_workers = max(1, num_workers) if num_workers is not None else 1

wnid_to_index = {wnid: index for index, wnid in enumerate(all_wnids)}
imagenet_r_to_full_map = {
    i: wnid_to_index[wnid] for i, wnid in enumerate(imagenet_r_wnids)
}


class ImagenetR(ImageFolder):
    def __init__(self, root, transform=None):
        super(ImagenetR, self).__init__(root, transform)
        self.class_to_idx = imagenet_r_to_full_map

    def __getitem__(self, index):
        image, label_r = super(ImagenetR, self).__getitem__(index)
        label_full = self.class_to_idx[label_r]

        return image, label_full

    @staticmethod
    def eval_model(
        model,
        root=".",
        device=torch.device("cpu" if not torch.cuda.is_available() else "cuda"),
        transforms=None,
        batch_size=128,
    ):
        model.eval()

        dataset = ImagenetR(root, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,  # type: ignore
            pin_memory=True,
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
