import glob

from PIL import Image

from torch.utils.data import Dataset
import torchvision

def Dataset(name, transform=None, mode='train', root='../Datasets/'):
    if name == 'cifar10':
        if mode == 'train':
            return torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        else:
            return torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    elif name == 'cifar100':
        if mode == 'train':
            return torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
        else:
            return torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
    elif name == 'celeba':
        return torchvision.datasets.CelebA(root=root, split=mode, transform=transform, download=True)

    elif name == 'imagenet':
        if mode == 'train':
            return torchvision.datasets.ImageNet(root=root, split='train', transform=transform)
        else:
            return torchvision.datasets.ImageNet(root=root, split='val', transform=transform)
    else:
        raise ValueError('Unsupported datasets: {}'.format(name))

def CIFAR10(transform=None, mode='train', root='../Datasets/'):
    if mode == 'train':
        return torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    else:
        return torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

def CIFAR100(transform=None, mode='train', root='../Datasets/'):
    if mode == 'train':
        return torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
    else:
        return torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)

class NoisedCIFAR10Dataset():
    def __init__(self, path, transform=None):
        if transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        else:
            self.transform = transform

        self.files = sorted(glob.glob(path + '/*.*'))

    def __getitem__(self, idx):
        data = self.transform(Image.open(self.files[idx]))
        target = int(self.files[idx][-5:-4])

        return (data, target)

    def __len__(self):
        return len(self.files)
