---
layout: post
title: 'Randaugment: Practical automated data augmentation'
date: 2021-11-18
author: Poley
cover: '/assets/img/20211116/Randaugment.png'
tags: 论文阅读
---

作者提出一种相比单独为每一种数据增强搜索一个合适的概率和强度，直接搜索一个single distortion magnitude 并以此来联合控制所有的操作是足够的。进一步提出一种简单搜索空间的自动增强方法，并且允许移除单独的代理任务。

![](/assets/img/20211116/RandaugmentT1.png)


目前已经有很多方法来进行自动的数据增强策略搜索。但是共同的阻碍就是他们需要一个分离并且昂贵的搜索阶段。一种常见的方法来客服这种昂贵的搜索阶段的方法是使用一个代理任务(proxy task)。虽然代理任务可以帮助加速搜索过程，但是同样会带来一些其他的问题。比如在代理任务上搜索到的最优参数可能不一定是真实任务(actual task)的最优参数。

<!--本文中证明了，当增强策略高度依赖于模型和数据集大小时，得到的结果是次优的。并且当可以移除一些代理任务上独立的搜索阶段时，可能可以得到更好的数据增强结果。（没太看懂）-->

本方法的特点： 搜索空间小，负担小，性能强。

这里首先对限制proxy task大小的两个常用维度进行了分析，这两个维度分别是model size 和 dataset size。
![](/assets/img/20211116/RandaugmentF1.png)
可以看出，更小的数据集上，数据增强可以体现出更大的提升能力。这里network使用的是Wide-Resnet，其网络大小可以随着参数*widening*来改变。在不同的网络大小和数据集大小下，得到的最优distortion magnitude如上图所示。可以看到，其最优的distortion magnitude和模型的大小以及训练集的大小都呈正相关。而在AutoAugment方法中，使用了固定的distortion magnitude，显然是次优的额结果。在小数据上，最优的distortion magnitude明显小于大数据集，作者推测这可能是由于小数据上过于激进的增强会导致数据集的低信噪比。

## Automated data augmentation with out a proxy task
![](/assets/img/20211116/RandaugmentF2.png)

一共使用$K=14$种数据增强方法，如下
![](/assets/img/20211116/RandaugmentK.png)

为了进一步降低参数的搜索空间，这里选择每一种数据增强的概率被固定为$\frac{1}{K}$。因此如果选择$N$种数据增强策略，则潜在的搜索选择是$K^N$种。

另一个重要的参数是数据增强的强度。之前的一些文献中表示，多个数据增强，在训练过程中其强度的变化基本符合一个similar schedule。因此本文提出使用一个 single global distortion $M$ ，可能已经足以parameterizing all transformations。本文中使用了四种$M$的schedule，分别是：固定，随机，线性增长以及带有增长上界的随机。

因此，最终的策略搜索简化为只有两个参数，增强变换数$N$和强度$M$。

## Experiments
![](/assets/img/20211116/RandaugmentT2.png)