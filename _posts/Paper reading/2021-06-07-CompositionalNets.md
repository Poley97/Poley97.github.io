---
layout: post
title: 'Cascade region proposal and global context for deep object detection'
date: 2021-06-07
author: Poley
cover: '/assets/img/20210607/CRPGC.png'
tags: 论文阅读
---

> 论文链接 ： https://www.sciencedirect.com/science/article/pii/S0925231219309117

# 主要内容

1. Cascade RPN : 注，这和那个Cascade RPN不一样。如下所示

![](/assets/img/20210607/CRPGCF2.png)

这里RPN2对RPN1所有的输出（稠密预测）进行refine。但是实际预测中，只用RPN2的输出实际上对于小目标的性能下降了，但是对于中大目标的性能提升了。因此这里用了一个trick,用了一个面积阈值来判断目标的大小，选择从哪里输出。

2. Augmenting classification with pretrained global context
   考虑到全局信息可能对分类有帮助而对回归没有明显帮助，这里单独提取的全局的ROI，并只用于分类。如下所示。
![](/assets/img/20210607/CRPGCF3.png)

3. 还加了一些tricks，比如预训练、正负样本比例等等……

# Experiments

![](/assets/img/20210607/CRPGCT1.png)

![](/assets/img/20210607/CRPGCT2.png)
