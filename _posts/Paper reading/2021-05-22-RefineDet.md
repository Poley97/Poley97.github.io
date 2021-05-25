---
layout: post
title: 'Single-Shot Refinement Neural Network for Object Detection'
date: 2021-05-20
author: Poley
cover: '/assets/img/20210520/RefineDet.png'
tags: 论文阅读
---

> 论文链接 https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.html

# Introduction

单阶段目标速度快，效率高，但是检测精度通常低于两阶段的。一个主要的原因来自于class imbalance problem

两阶段的结构相比单阶段的主要三个优点： 

1.	通过采样降低了class imbalance
2.	使用two-step cascade 回归box ，即两次回归
3.	使用two-stage feature 来描述物体，即两阶段特征。

本文提出两个module
1.	anchor refinement module(ARM):
    +	识别并移除负样本，减小classifier的搜索空间
    +	Coarsely 调整 anchor的loc和sizes，提供更好的初值给subsequent regressor
2. object detection module(ODM)	
    -	使用refined anchor as input 来进一步提升回归质量和预测multi class label

ARM其实就是一个bottom-up主干网络，使用VGG16或者RESNET101，为了丰富特征，在VGG16后面加了两个卷积层conv6_1，conv6_2，或者再RESNET101后面加一个RES模块。参数初始化使用xavier初始化。

ODM是一个top-down网络这个结构基本类似FPN，只是因为当时FPN没有开源，所以作者类比FPN自己写了一个结构。

Transfer Connection Block（TCBS）用于连接两者，就是一个3*3卷积。用于转换特征。使得ARM可以和ODM共享特征。但是值得注意的是，作者只是用TCBS在和anchors有关的feature maps上。即第一次预测anchor的层。之后和top-down的层相加，这和fpn一样。


具体来说，ARM首先按常规划分计算每个cell上n个anchor的objectiveness和xywh参数。经过置信度筛选之后（去除置信度很高的负样本，比如背景置信度>0.99），再经过loss筛选（选择loss大的负样本），使得正负样本1：3，再进一步送入ODM中进行进一步处理。

ODM接受ARM传来的refined anchors作为输入，ODM再输出c+4个output，对应每个class，以及4个相对于refined anchors的偏移，完成预测。

![](/assets/img/20210522/RefineDetArch.png)

![](/assets/img/20210522/RefineDetBlock.png)

总体来说，本文的思想在于分析了为什么two-stage的性能强于one-stage，但速度快于two-stage。通过分析，作者巧妙的将两者结合起来，避免使用RoIpooling来拖慢速度，同时也用到目标框的二次回归以及特征的二次提取，达到的不错的效果。这个分析问题的思路是值得学习的。


