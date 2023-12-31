import torch
from loguru import logger
import sys
import os

from datasets.cifar import cifar10, cifar100
from models.cnn import *
from models.densenet import *
from models.efficientnet import *
from models.googlenet import *
from models.lenet import *
from models.mobilenet import *
from models.mobilenetv2 import *
from models.regnet import *
from models.resnet import *
from models.shufflenet import *
from models.shufflenetv2 import *
from models.vgg import *

checkpoint_dir = './checkpoints'
input_channels = 3

def load_dataset(dataset: str = '', batch_size: int = 4, num_workers: int = 2):
   """
   Load a specified dataset.

   Parameters:
   dataset (str): The name of the dataset to load. Supported datasets are 'cifar10', 'cifar100', and 'imagenet'.
   batch_size (int, optional): The number of samples per gradient update. Default is 4.
   num_workers (int, optional): The number of subprocesses to use for data loading. Default is 2.

   Returns:
   tuple: A tuple containing the number of input channels, the number of classes, the training loader, and the testing loader.
           If the dataset is not supported, returns None for all values.
   """
   
   logger.info('Loading dataset...')
   
   # Check which dataset to load
   if dataset == 'cifar10':
       num_classes = 10 # Set the number of classes for the CIFAR-10 dataset       
       train_loader, test_loader = cifar10(batchsize=batch_size, numworkers=num_workers)
       logger.info('Dataset Loaded!')
       return input_channels, num_classes, train_loader, test_loader
   elif dataset == 'cifar100':
       num_classes = 100 # Set the number of classes for the CIFAR-100 dataset
       train_loader, test_loader = cifar100(batchsize=batch_size, numworkers=num_workers)
       logger.info('Dataset Loaded!')
       return input_channels, num_classes, train_loader, test_loader
   else:
       train_loader, test_loader = None, None
       logger.info("Dataset is not supported!")
       return None, None, None, None

def progress_bar(current, total, length=40):
   """
   Display a progress bar in the console.

   Parameters:
   current (int): Current progress.
   total (int): Total amount of work to be done.
   length (int, optional): Length of the progress bar. Defaults to 40.

   Returns:
   str: Progress bar string.
   """
   progress = int(length * current / total)
   bar = "[" + "=" * progress + ">" + " " * (length - progress - 1) + "]"
   percent = f"{100 * current / total:.2f}%"
   return f"{bar} {percent}"

def build_model(model_name, input_channels, num_classes, net_size=2):
   """
   Build a neural network model based on the provided parameters.

   Parameters:
   model_name (str): Type of model to be built.
   input_channels (int): Number of input channels for the model.
   num_classes (int): Number of output classes for the model.
   net_size (int, optional): Size of the network. Only used if model_name is 'ShuffleNetV2'. Defaults to 2.

   Returns:
   object: The constructed model object.
   """
   logger.info('Building model...')
   if model_name != 'ShuffleNetV2':
       net = eval(f'{model_name}(input_channels={input_channels}, num_classes={num_classes})')
       logger.info(f'{net}')
       logger.info('Model loaded successfully!')
       return net
   elif model_name == 'ShuffleNetV2':
       net = eval(f'{model_name}(net_size={net_size}, input_channels={input_channels}, num_classes={num_classes})')
       logger.info(f'{net}')
       logger.info('Model loaded successfully!')
       return net
   else:
       logger.info("MODEL IS NOT SUPPORTED!")
       return None

def train(net, device, train_loader, criterion, optimizer, scheduler, model_name, start_epoch=0, total_epochs=2, save_model=False):
   """
   Function to train a model on a given dataset.

   Parameters:
   model (nn.Module): The neural network model.
   device (str): The device where the computations will be performed. It could be either 'cpu' or 'cuda'.
   train_loader (DataLoader): The data loader for the training data.
   criterion (nn.Module): The loss function.
   optimizer (torch.optim): The optimization algorithm.
   scheduler (_LRScheduler): The learning rate scheduler.
   model_name (str): The name of the model.
   start_epoch (int): The starting epoch. Default is 0.
   total_epochs (int): The total number of epochs. Default is 2.
   save_model (bool): Whether to save the model. Default is False.
   """
   # Move the model to the specified device
   model = net.to(device)
    
   # Set the model to training mode
   model.train()

   # Initialize variables to keep track of the total loss and the number of correct predictions
   running_loss = 0.0
   correct = 0
   total = 0
   best_acc = 0.0

   # Loop over the dataset multiple times
   for epoch in range(start_epoch, total_epochs):

       # Loop over the batches in the training data loader
       for batch_idx, (data, target) in enumerate(train_loader):
           
           # Move the data and target to the device where the computations will be performed
           data, target = data.to(device), target.to(device)
           
           # Zero out the gradients from the previous iteration
           optimizer.zero_grad()
           
           # Pass the input data through the model to get the output
           output = model(data)
           
           # Calculate the loss between the model's output and the actual targets
           loss = criterion(output, target)
           
           # Perform backpropagation, calculating the gradients of the loss with respect to the model's parameters
           loss.backward()
           
           # Update the model's parameters based on the calculated gradients
           optimizer.step()
           
           # Add the current loss to the running total of losses
           running_loss += loss.item()
           
           # Get the index of the maximum value in the output tensor along dimension 1, which corresponds to the predicted class
           _, predicted = torch.max(output.data, 1)
           
           # Increment the total count of predictions
           total += target.size(0)
           
           # Increment the count of correct predictions
           correct += (predicted == target).sum().item()
           
           # Calculate the accuracy as the ratio of correct predictions to total predictions
           acc = correct / total

           if batch_idx % 100 == 0 or batch_idx == len(train_loader) - 1:  
              logger.info(f"Training - Epoch {epoch} {progress_bar(batch_idx + 1, len(train_loader))}    Loss: {(running_loss / (batch_idx + 1)):.4f}   Accuracy: {(acc):.4f}", end="\r")
    
       # Save model if it has achieved the best accuracy so far
       if save_model:
           if acc > best_acc:
               state_dict = {
                  'accuracy': acc,
                  'epoch': epoch,
                  'net': model.state_dict(),
               }
               best_acc = acc
               torch.save(state_dict, os.path.join(checkpoint_dir, f'{model_name}_best_model.pth'))

       scheduler.step()

def test(model, device, test_loader, criterion):
   """
   Function to evaluate the performance of a model on a given dataset.

   Parameters:
   model (nn.Module): The neural network model.
   device (str): The device where the computations will be performed. It could be either 'cpu' or 'cuda'.
   test_loader (DataLoader): The data loader for the testing data.
   criterion (nn.Module): The loss function.
   """

   # Set the model to evaluation mode
   model.eval()

   # Initialize variables to keep track of the total loss and the number of correct predictions
   test_loss = 0
   correct = 0
   total = 0

   # Turn off gradient calculation
   with torch.no_grad():
       # Loop over the batches in the test data loader
       for batch_idx, (data, target) in enumerate(test_loader):
           # Move the data and target to the device where the computations will be performed
           data, target = data.to(device), target.to(device)
           
           # Pass the input data through the model to get the output
           output = model(data)
           
           # Calculate the loss between the model's output and the actual targets
           test_loss += criterion(output, target).item()
           
           # Get the index of the maximum value in the output tensor along dimension 1, which corresponds to the predicted class
           pred = output.argmax(dim=1, keepdim=True)

           # Increment the total count of predictions
           total += target.size(0)
           
           # Increment the count of correct predictions
           correct += pred.eq(target.view_as(pred)).sum().item()
           
           logger.info(f"Testing - {progress_bar(batch_idx + 1, len(test_loader))}    Accuracy: {(correct/total):.2f}")

   # Calculate the average loss
   test_loss /= len(test_loader.dataset)

   # Print the average loss and accuracy
   logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))
