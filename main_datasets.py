import os
import torchvision
import torchvision.transforms as transforms
from arguments import getArgs

args = getArgs()
os.makedirs(os.path.join(args.drive_path, 'datasets'), exist_ok=True)

# Define transformations for the training and test sets
transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139969, 0.48215842, 0.44653093), (0.20220212, 0.19931542, 0.20086347))  # Normalize with mean and std of CIFAR-10
])

cifar10_trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.drive_path, 'datasets/cifar10_data'), train=True, transform=transform_cifar10, download=True)
cifar10_testset = torchvision.datasets.CIFAR10(root=os.path.join(args.drive_path, 'datasets/cifar10_data'), train=False, transform=transform_cifar10, download=True)

# SVHN
transform_svhn = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1201, 0.1231, 0.1052)),
])

svhn_trainset = torchvision.datasets.SVHN(root=os.path.join(args.drive_path, 'datasets/svhn_data'), split='train', transform=transform_svhn, download=True)
svhn_testset = torchvision.datasets.SVHN(root=os.path.join(args.drive_path, 'datasets/svhn_data'), split='test', transform=transform_svhn, download=True)

# MNIST
def replicate_channels(x):
    return x.repeat(3, 1, 1)  # Replicate the single channel to 3 channels

transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(replicate_channels)  # Use the defined function instead of a lambda
])

mnist_trainset = torchvision.datasets.MNIST(root=os.path.join(args.drive_path, 'datasets/mnist_data'), train=True, transform=transform_mnist, download=True)
mnist_testset = torchvision.datasets.MNIST(root=os.path.join(args.drive_path, 'datasets/mnist_data'), train=False, transform=transform_mnist, download=True)

# FMNIST
transform_fmnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3205,)),
    transforms.Lambda(replicate_channels)  # Use the defined function instead of a lambda
])

fmnist_trainset = torchvision.datasets.FashionMNIST(root=os.path.join(args.drive_path, 'datasets/fmnist_data'), train=True, transform=transform_fmnist, download=True)
fmnist_testset = torchvision.datasets.FashionMNIST(root=os.path.join(args.drive_path, 'datasets/fmnist_data'), train=False, transform=transform_fmnist, download=True)

def load_cifar10_dataset(train):
    return cifar10_trainset if train else cifar10_testset

def load_mnist_dataset(train):
    return mnist_trainset if train else mnist_testset

def load_svhn_dataset(train):
    return svhn_trainset if train else svhn_testset

def load_fmnist_dataset(train):
    return fmnist_trainset if train else fmnist_testset