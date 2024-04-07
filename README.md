# BF-GCN: An efficient graph learning system for emotion recognition inspired by the cognitive prior graph of EEG brain network

This is a simple demo of BF-GCN based on PyTorch.

# Installation:

- Python 3.11
  
- numpy 1.24.3
  
- pandas 1.5.3
  
- scikit_learn 1.3.0
  
- scipy 1.10.1
  
- torch 2.0.1
  
- NVIDIA CUDA 11.8
  

# Variable Descriptions:

- xdim: the shape of data
  
- kadj: k order of cheybnet
  
- nclass: the number of class
  
- num_out: the hidden dim of node feature
  
- att_hidden: the hidden dim of attention to fuse the three graph branches
  
- att_plv_hidden: the hidden dim of attention to fuse the four bands plv matrix
  
- classifier_hidden: the hidden dim of classifier
  
- avgpool: the size of 2D average-pooling operation
  
- dropout: the rate of dropout
  

# Data:

- Prepare dataset: [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)

we give 100 real samples in demo_data to run the Simple_Demo.py. If you want to use our model to train on SEED, DEAP or your dataset, you can see the usage.

# Usage:

**subject-independent**

In the paper, we utilize the Domain-Adversarial Neural Network (DANN) transfer method for subject-independent experiments. You should use **domain_output** to calculate the adversarial loss.

**subject-dependent**

In the paper, we don't utilize the Domain-Adversarial Neural Network (DANN) transfer method for subject-dependent experiments. So you can disregard domain_output.
