# A Practical Clean-Label Backdoor Attack with Limited Information in Vertical Federated Learning

This repository contains the implementation code of the paper: "A Practical Clean-Label Backdoor Attack with Limited Information in Vertical Federated Learning".


## Description

(omitted for brevity)

## Setup

1. Clone this repository:


2. Install the required dependencies:

```bash
pip install -r requirements.txt
```


## Training

Taking the attack on the CIFAR10 dataset as an example, other datasets CIFAR100 and CINIC10 are similar

To train a model, first configure the experiment settings in the `/configs` directory. Edit the configuration files according to your needs. Then, run the following command:

```bash
python vfl_cifar10_training.py --config=configs/base/cifar10_bestattack.yml
```

The trained model will be saved in the `/model` directory.

## Testing

Taking the attack on the CIFAR10 dataset as an example, other datasets CIFAR100 and CINIC10 are similar

To test a trained model, run the following command:

```bash
python vfl_cifar10_test.py --config=configs/base/cifar10_bestattack.yml
```

Replace `model/CIFAR10/base/0_saved_models/model_best.pth.tar` with the path to your saved model.

## Pre-trained Models

This repository includes pre-trained models in the `/model` directory. You can find pre-trained models for CIFAR-10, CIFAR-100, and CINIC-10 datasets.



## Dependencies

This project has the following dependencies:

- PyTorch
- torchvision
- NumPy
- scikit-learn
- tqdm

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
