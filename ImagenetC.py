from .utils import thousandK_wnids as all_wnids

from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

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
        img = Image.fromarray(img)
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
        if subset != "all":
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
    def pretty_print_acc(accuracies, args, cfg):
        print("=" + "*=" * 12)
        print(f"MODEL: {args.model: <10} DATASET: {args.dataset}")
        print(f"Batch Size: {args.batch_size: <10} Workers: {args.workers}")
        print(f"Top1: {100*accuracies['top1']:.4f} Top5: {100*accuracies['top5']:.4f}")
        for sk in accuracies:
            if sk in ["top1", "top5", "total", "correct_top1", "correct_top5"]:
                continue
            print(
                f"{sk.upper()} Top1 Acc: {100* accuracies[sk]['correct_top1'] / accuracies[sk]['total']:.4f}"
            )
            print(
                f"{sk.upper()} Top5 Acc: {100* accuracies[sk]['correct_top5'] / accuracies[sk]['total']:.4f}"
            )
            for ssk in accuracies[sk]:
                if ssk in ["total", "correct_top1", "correct_top5"]:
                    continue
                print(
                    f"{sk.upper()} {ssk.upper()} Top1 Acc: {100* accuracies[sk][ssk]['correct_top1'] / accuracies[sk][ssk]['total']:.4f}"
                )
                print(
                    f"{sk.upper()} {ssk.upper()} Top5 Acc: {100* accuracies[sk][ssk]['correct_top5'] / accuracies[sk][ssk]['total']:.4f}"
                )

                for n in accuracies[sk][ssk]:
                    if n in ["total", "correct_top1", "correct_top5"]:
                        continue
                    print(
                        f"{sk.upper()} {ssk.upper()} {n} Top1 Acc: {100* accuracies[sk][ssk][n]['correct_top1'] / accuracies[sk][ssk][n]['total']:.4f}"
                    )
                    print(
                        f"{sk.upper()} {ssk.upper()} {n} Top5 Acc: {100* accuracies[sk][ssk][n]['correct_top5'] / accuracies[sk][ssk][n]['total']:.4f}"
                    )

            print()
        print("=" + "*=" * 12)

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
        **kwargs,
    ):
        model.eval()
        model = model.to(device)
        if subset == "all":
            correct_top1, correct_top5, total = 0, 0, 0
            subsets = IC.subsets.keys()
            accuracies = dict()
            for subset in subsets:
                accuracies[subset] = dict()
                accuracies[subset]["total"] = 0
                accuracies[subset]["correct_top1"] = 0
                accuracies[subset]["correct_top5"] = 0
                for subsubset in IC.subsets[subset]:
                    accuracies[subset][subsubset] = dict()
                    accuracies[subset][subsubset]["total"] = 0
                    accuracies[subset][subsubset]["correct_top1"] = 0
                    accuracies[subset][subsubset]["correct_top5"] = 0
                    for n in noise:
                        print(f"Subset: {subset} Subsubset: {subsubset} Noise: {n}")
                        accuracies[subset][subsubset][n] = dict()
                        accuracies[subset][subsubset][n]["total"] = 0
                        accuracies[subset][subsubset][n]["correct_top1"] = 0
                        accuracies[subset][subsubset][n]["correct_top5"] = 0

                        dataset = ImFolder(
                            os.path.join(root, subset, subsubset, str(n)), transforms
                        )
                        dataloader = DataLoader(
                            dataset, batch_size=batch_size, shuffle=False
                        )
                        for x, y in dataloader:
                            x, y = x.to(device), y.to(device)
                            with torch.no_grad():
                                out = model(x)

                                top1_correct = (
                                    torch.eq(torch.topk(out, 1).indices, y.view(-1, 1))
                                    .sum()
                                    .item()
                                )
                                top5_correct = (
                                    torch.eq(torch.topk(out, 5).indices, y.view(-1, 1))
                                    .sum()
                                    .item()
                                )

                                count_items = len(y)

                                accuracies[subset][subsubset][n]["total"] += count_items
                                accuracies[subset][subsubset][n][
                                    "correct_top1"
                                ] += top1_correct
                                accuracies[subset][subsubset][n][
                                    "correct_top5"
                                ] += top5_correct

                                accuracies[subset][subsubset]["total"] += count_items
                                accuracies[subset][subsubset][
                                    "correct_top1"
                                ] += top1_correct
                                accuracies[subset][subsubset][
                                    "correct_top5"
                                ] += top5_correct

                                accuracies[subset]["total"] += count_items
                                accuracies[subset]["correct_top1"] += top1_correct
                                accuracies[subset]["correct_top5"] += top5_correct

                                total += count_items
                                correct_top1 += top1_correct
                                correct_top5 += top5_correct

            accuracies["top1"] = correct_top1 / total
            accuracies["top5"] = correct_top5 / total
            return accuracies


# class_name = idx_to_class[label.item()]
