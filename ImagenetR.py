from .utils import imagenetR_wnids as imagenet_r_wnids
from .utils import thousandK_wnids as all_wnids
from .utils import Accuracy

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch

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
    def pretty_print_acc(acc, args, cfg):
        print("=" + "*=" * 12)
        print(f"MODEL: {args.model: <10} DATASET: {args.dataset}")
        if args.top5:
            print(f"Top-5 Acc: {acc * 100:.2f}%")
        else:
            print(f"Top-1 Acc: {acc * 100:.2f}%")
        print("=" + "*=" * 12)

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

                if top5:
                    correct_batch, total_batch = Accuracy._top5(outputs, labels)
                else:
                    correct_batch, total_batch = Accuracy._top1(outputs, labels)

                total += total_batch
                correct += correct_batch

        return correct / total
