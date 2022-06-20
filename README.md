# SNN-Segmentation

This is the code repository for the paper [Beyond Classification: Directly Training Spiking Neural Networks for Semantic Segmentation](https://arxiv.org/abs/2110.07742).

## Abstract

Spiking Neural Networks (SNNs) have recently emerged as the low-power alternative to Artificial Neural Networks (ANNs) because of their sparse, asynchronous, and binary event-driven processing. Due to their energy efficiency, SNNs have a high possibility of being deployed for real-world, resource-constrained systems such as autonomous vehicles and drones. However, owing to their non-differentiable and complex neuronal dynamics, most previous SNN optimization methods have been limited to image recognition. In this paper, we explore the SNN applications beyond classification and present semantic segmentation networks configured with spiking neurons. Specifically, we first investigate two representative SNN optimization techniques for recognition tasks (i.e., ANN-SNN conversion and surrogate gradient learning) on semantic segmentation datasets. We observe that, when converted from ANNs, SNNs suffer from high latency and low performance due to the spatial variance of features. Therefore, we directly train networks with surrogate gradient learning, resulting in lower latency and higher performance than ANN-SNN conversion. Moreover, we redesign two fundamental ANN segmentation architectures (i.e., Fully Convolutional Networks and DeepLab) for the SNN domain. We conduct experiments on two public semantic segmentation benchmarks including the PASCAL VOC2012 dataset and the DDD17 event-based dataset. In addition to showing the feasibility of SNNs for semantic segmentation, we show that SNNs can be more robust and energy-efficient compared to their ANN counterparts in this domain.

## Repository structure

The repository is compartmentalized as follows:

- `/data`: Directory for datasets (VOC2012 and DDD17) and related utilities
- `/models`: Directory for ANN and SNN models and related utilities
- `/utils`: Directory for other relevant utility functions and constants
- `setup.py`: Function that sets up an environment with relevant objects and config for training or testing for semantic segmentation (used by `train.py` and `test.py`)
- `train.py`: Script to train an ANN or SNN for semantic segmentation
- `test.py`: Script to evaluate a model or convert an ANN to an SNN for semantic segmentation
- `converter.py`: Script to change the format of a pretrained model file's state_dict/thresholds objects
- `environment.yml`: List of package dependencies