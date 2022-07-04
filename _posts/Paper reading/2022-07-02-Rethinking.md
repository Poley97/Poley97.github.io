---
layout: post
title: 'Rethinking Pre-training and Self-training'
date: 2022-06-29
author: Poley
cover: '/assets/img/20220627/Rethinking.png'
tags: 论文阅读
---

这篇文章主要是通过实验来探究预训练和自训练（基于伪标签的半监督）方法对于目标检测模型的性能影响。本研究揭示了自训练的泛化性和灵活性，并带来三个新的观点：
1. 更强的数据增强、更多的标注数据进一步磨灭了预训练的价值；
2. 与预训练不同，self-training总是可以从更强的数据增强中收益；

因此，预训练模型在小数据集上效果比较明显，比如PASCAL，而在大数据集上，并没有体现出明显收益，比如COCO。



实验表明，使用ImageNet作为无监督数据辅助COCO进行self-training效果优于使用ImageNet预训练模型作为COCO的初始化参数。在强数据增强的加持下，预训练的效果消失了（甚至会导致性能下降）。强数据增强有益于self-training。另一种常见的预训练方法是使用自监督，产生更加通用的representation。实验表明，这样预训练的效果也是一样的差。当然，预训练也不是完全没用。首先根据预训练的质量，可以大幅提升下游任务的训练速度。其次，当应用的标注数据很难获取时，预训练work well。但是，self-training无需预训练也可以work well。

本文对预训练模型在强数据增强下的角色进行研究，包括不同的与训练方法、以及不同的预训练质量。使用强度不一的4种数据增强方法，分别是
1. FlipCrop;
2. AutoAugment;
3. AutoAugment with higher scale jittering;
4. RandAugment with higher scale jittering，分别称为 Augment-S1, Augment-S2, Augment-S3 and Augment-S4。同时使用三种不同的预训练方案，分别是三种预训练质量：
1. 随机参数
2. 84.5 top1 error
3. 86.9 top1 error

得到的训练结果如下图所示

![](/assets/img/20220627/RethinkingF1.png)

可以看到，实际上预训练模型在强增强(stronger augmentation)以及大量标注数据的情况下对训练起到了负面作用。作者进一步探究了自训练的作用（self-training）。实验表明，自训练在大数据量和强增强下依然对模型性能有正面帮助，然而pre-training则相反，结果如下表所示。

![](/assets/img/20220627/RethinkingT2.png)

![](/assets/img/20220627/RethinkingT3.png)


最后，作者探究了当前流型的自监督方法（self-supervised）预训练方法对于目标检测的影响。自监督方法使用对比学习等方法，目的是获得更加general的representation，在分类等领域具有很好的效果。这里使用在ImageNet上使用自监督训练方法SimCLR方法获得的初始化模型进行测试，结果如下

![](/assets/img/20220627/RethinkingT4.png)


预训练在面对不同任务的时候可能需要进行adaptation，比如ImageNet的特征可能忽略了图像中的位置信息，但是检测是需要用到的。而self-training相当于引入了joint-training，使得网络在学习良好特征表达的同时弥补这种task之间的Gap。进一步验证，作者使用ImageNet的有监督和COCO共同训练。得到下过如图，具有明显的增益。同时，联合训练在提升训练效果的同时，所需要的epoch数远小于常用于微调的预训练模型（19 vs 350）。联合训练的效果如下图所示
![](/assets/img/20220627/RethinkingT7.png)