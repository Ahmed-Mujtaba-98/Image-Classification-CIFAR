import torch
import torchvision
import argparse
import sys
import os

import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from loguru import logger

from utils import train, test, build_model, load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running parameters")
    parser.add_argument('--dataset', default='cifar10', type=str, required=True, help='Dataset to load',
        choices=[
            'cifar10',
            'cifar100',
        ]                    
    )
    parser.add_argument('--model', default='Net', type=str, required=True, help='Model to load', 
        choices=[
            'Net',
            'DenseNet121', 'DenseNet169', 'DenseNet201', 'DenseNet161',
            'EfficientNetB0',
            'GoogLeNet',
            'LeNet',
            'MobileNet', 'MobileNetV2',
            'RegNetX_200MF', 'RegNetX_400MF', 'RegNetY_400MF',
            'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
            'ShuffleNetG2', 'ShuffleNetG3', 'ShuffleNetV2',
            'VGG11', 'VGG13', 'VGG16', 'VGG19',
        ]
    )
    parser.add_argument('--epochs', default=2, type=int, required=True, help='total number of epochs')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='batch size for dataloader')
    parser.add_argument('--num_workers', default=2, type=int, required=False, help='total number of workers')
    parser.add_argument('--net_size', default=2, type=float, required=False, help='Net size for shufflenetv2')
    parser.add_argument('--save_model', action='store_true', help='Save best model checkpoint')
    parser.add_argument('--resume', action='store_true', help='Resume best model checkpoint training')
    args = parser.parse_args()

    start_epoch = 0

    # Check if the checkpoints directory to save best model exists, if not, create it
    checkpoint_dir = './checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Check for gpu availability
    logger.info(f'Checking for device...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device set to {device}')

    # Load dataset
    input_channels, num_classes, train_loader, test_loader = load_dataset(dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Define model
    net = build_model(model_name=args.model, input_channels=input_channels, num_classes=num_classes, net_size=args.net_size)
    
    # Resume training from saved checkpoints
    if args.resume:
        logger.info(f'Loading {args.model} checkpoint..')
        assert os.path.isdir(checkpoint_dir), 'No checkpoint directory found!'
        ckpt = torch.load(os.path.join(checkpoint_dir, f'{args.model}_best_model.pth'))
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']

    # Define the loss function, optimizer, and learning rate scheduler for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Start training
    logger.info('Training started!')
    train(net=net, device=device, train_loader=train_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
        start_epoch=start_epoch, total_epochs=args.epochs, model_name=args.model, save_model=args.save_model)
    logger.info('Finished Training!')

    # Test model
    logger.info('Testing model!')
    test(model=net, device=device, test_loader=test_loader, criterion=criterion)
    logger.info('All done!')

