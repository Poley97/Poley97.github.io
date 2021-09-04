---
layout: post
title: 'DropBlock: A regularization method for convolutional networks'
date: 2021-08-11
author: Poley
cover: '/assets/img/20210811/DropBlock.png'
tags: 论文阅读
---

本文提出一种对于卷积层的结构化dropout方法，称为DropBlock，体现出比随机dropout更好的性能。


Dropout主要用于卷积神经网络的全连接层中，具有不错的效果。但是卷积结构很少有使用dropout。作者认为dropout的主要缺点在于其随机的丢弃特征，这对于全连接层很有效，但是由于卷积层是空间相关的，只丢掉一个节点的数据，信息仍然可以通过附近的节点继续向下传播。

因此本文提出了DropBlock，借鉴了数据增强方法Cutout，相当于在每个featrue map上都随机做了Cutout。如下图所示

![](/assets/img/20210811/DropBlockF1.png)

在实验中，发现固定比率的DropBlock使得网络并不鲁棒，更好的方法是逐步增加Drop的比率，类似于SceduledDropPath。

算法流程如下所示
![](/assets/img/20210811/DropBlockA1.png)

示意图如下所示
![](/assets/img/20210811/DropBlockF2.png)

一共两个参数，分别是$block\_size$和$\gamma$。

$block\_size$代表对于一个随机采样的点，拓展出多大的dropblock;

$\gamma$是随机选取点的概率，但是这里不是显示给出的，而是通过控制总体点被drop的比例来计算得到，如下

![](/assets/img/20210811/DropBlockE1.png)

实验分析如下，block_size=1相当于随机dropout，可以看到dropblock具有更好的泛化性能。
![](/assets/img/20210811/DropBlockF5.png)

从Class activation mapping 上可以看出，Dropbolck带来了空间上更大的关注区域，使得网络更加鲁棒。
![](/assets/img/20210811/DropBlockF6.png)