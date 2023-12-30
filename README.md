# 🏝️ IMAGE CLASSIFICATION USING PYTORCH 🔥️ - CIFAR DATASET

**About CIFAR10**

The CIFAR-10 dataset is a collection of images commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

**About CIFAR100**

The CIFAR-100 dataset is a collection of 60,000 32x32 color images in 100 different classes. Each class contains 600 images, with 500 images for training and 100 images for testing. The 100 classes are divided into 20 superclasses. For each image, there are two labels: a "fine" label indicating the specific class, and a "coarse" label indicating the superclass. The dataset was introduced by Krizhevsky et al., and is widely used for benchmarking machine learning algorithms.

## ✍️ Prerequisites
You should have anaconda package already installed on your host PC. If not, visit the [Anaconda official site](https://www.anaconda.com/download) to download and install the latest package.

## 👨‍🔧️ Environment Setup
Clone this repository by running the following commands on the terminal,

```
git clone <this_repository>
cd <this_repository>
```
Setup conda environment,

```
conda create -n my_env python=3.10
conda activate my_env
pip install -r requirements.txt
```

## 📌️ List of Supported Models and Datasets
List of models supported for training are,

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

> "Net" is a simple CNN with few convolution and fully-connected layers. You can find all models architecture [here](./models/).

List of datasets supported for training are,

    choices=[
        'cifar10',
        'cifar100',
    ]

## 🌴️ Model Training
To starting training the models, run the below command,

```
python train.py --epochs 10 --dataset cifar10 --model Net
```

To save best model during training, run the below command,

```
python train.py --epochs 10 --dataset cifar10 --model Net --save_model
```

To resume training from saved checkpoint, run the below command,

```
python train.py --epochs 10 --dataset cifar10 --model Net --resume
```

### 📃️ Arguments Description

- **dataset**: Dataset to load, default='cifar10'
- **epochs**: Total number of epochs, default=2s
- **lr**: Learning rate for model trainig, default=0.001
- **batch_size**: Batch size for dataloader, default=4
- **num_workers**: Total number of workers, default=2
- **net_size**: Net size for shufflenetv2, default=2
- **save_model**: Save best model checkpoint
- **resume**: Resume training from checkpoint


## 🌍️ References
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

## 📢️ Other Repositories and Contribute
- Checkout this repository for training image classification on MNIST and Fashion MNIST dataset.
- Feel free to contribute and create a pull request to add additional features. Also, open issues if you face any difficulty, I will happy to assist you in solving the problems.