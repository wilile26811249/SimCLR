from PIL import Image
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms.functional import gaussian_blur


class Cifar10_Pair(datasets.CIFAR10):
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            aug_1 = self.transform(img)
            aug_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return aug_1, aug_2, target


train_transform = T.Compose([
    T.RandomResizedCrop(32),
    T.RandomHorizontalFlip(p = 0.5),
    T.RandomApply([
        T.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p = 0.8
    ),
    T.RandomGrayscale(p = 0.2),
    T.GaussianBlur(3),
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])