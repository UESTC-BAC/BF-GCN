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

We give 100 real samples in demo_data to run the Simple_Demo.py. If you want to use our model to train on SEED, DEAP or your dataset, you can see the usage.

# Usage:

It is worth pointing out that the proposed BF-GCN graph learning system designs two branches for the emotion recognition procedure, i.e., the fully-connected layer and the domain adversarial layer.

**subject-independent**

For the emotion recognition task with the transfer learning task, i.e., subject-independent emotion recognition, the BFGCN graph learning system adopts the domain adversarial layer to achieve a better transfer learning effect and realize the decoding of cross-individual emotion recognition.

**subject-dependent**

For the emotion recognition task without transfer learning, that is, the subject-dependent emotion decoding experiment, the BF-GCN graph learning system utilizes a simple full connection layer with dropout and a softmax function to decode individual emotional states.

# Citation:
If you find our work helps your research, please kindly consider citing our paper in your publications.
@ARTICLE{10549833, author={Li, Cunbo and Tang, Tian and Pan, Yue and Yang, Lei and Zhang, Shuhan and Chen, Zhaojin and Li, Peiyang and Gao, Dongrui and Chen, Huafu and Li, Fali and Yao, Dezhong and Cao, Zehong and Xu, Peng}, journal={IEEE Transactions on Neural Networks and Learning Systems}, title={An Efficient Graph Learning System for Emotion Recognition Inspired by the Cognitive Prior Graph of EEG Brain Network}, year={2024}, volume={}, number={}, pages={1-15}, doi={10.1109/TNNLS.2024.3405663}}
