import torch
import torchvision
import torchvision.transforms as transforms

def cifar10(batchsize: int = 4, numworkers: int = 2):
   # Define transformations for the training data and testing data
   transform_train = transforms.Compose([
       transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   transform_test = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])

   # Download and load the training data
   trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=numworkers)

   # Download and load the test data
   testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
   testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=numworkers)

   return trainloader, testloader


def cifar100(batchsize: int = 4, numworkers: int = 2):
   # Define a transform to augment the data
   transform = transforms.Compose(
       [transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

   # Download and load the training data
   trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=numworkers)

   # Download and load the test data
   testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
   testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=numworkers)

   return trainloader, testloader
