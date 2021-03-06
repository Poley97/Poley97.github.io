---
layout: post
title: 'End-to-End Semi-Supervised Object Detection with Soft Teacher'
date: 2022-06-27
author: Poley
cover: '/assets/img/20220627/SoftTeacher.png'
tags: 论文阅读
---

这篇文章发表于ICCV2021，本方法同样也是关注与两个方面：1、提高标签质量；2、选择合适的监督（其实也相当于提高标签质量）。

本文的创新点主要在两个方面
+ 首先，使用高阈值过滤目标框，得到高准确率的伪标签。同时考虑到高阈值会错分很多正样本为负，这里引入负样本的权重，以可信度作为负样本的权重来进行学习。即sotfteacher。
+ 分离分类和回归的标签，对于回归使用单独的标签筛选机制，即施加干扰之后的输出结果方差，越大说明可信度越低，进而实现更加准确的监督。

模型的整体结构如下图所示
![](/assets/img/20220627/SoftTeacherF2.png)

## Soft Teacher
传统的伪标签就是使用一个网络对无标签数据进行预测，并使用一个置信度阈值来划分高置信度的输出作为伪标签。这样造成一个问题，就是高阈值导致高精度低召回率，低阈值导致高召回率低精度，一般情况下，为了保证标签的准确性，常常使用前者。

由于伪标签的召回率低，因此按正常的IoU阈值划分前景背景会导致部分前景Box被错误的划分为背景，导致性能的退化。

本文针对这个问题，提出了解决方法：使用背景置信度来软化背景标签（和损失），减少不确定性高的背景包含的潜在错误对训练的影响（注意到，这里只对background标签和损失进行软化，foreground是正常计算的）。但是，如何计算背景的置信度又是一个问题，这里作者提出了BG-T方法，同时总结了常见的若干种其他方法。

几种背景置信度（reliability）的（代理，proxy）指标对比：
1、BG-S直接使用学生对候选框的背景预测得分作为置信度的代理；
2、BG-T（本文方法）将学生候选框送进教师网络的detection head来进行处理，并得到对应的background得分作为置信度的代理。
2、使用student和teacher对背景预测的得分差距的反比（1-diff）作为置信度的proxy；
3、基于IoU的有两种相反的思路，一种认为和gt IoU小于阈值的情况下，越大，约可能是background（可以看做一种hard negative mining）；反之，认为越小越可能是。这里称为IoU和reverse-IoU.

作者在实验中也对这些置信度方法进行了对比，证明BG-T是最优的方法，这里不再赘述。

## Box Jitter

这是作者提出的对Regression监督进行筛选的方法。请看下图

![](/assets/img/20220627/SoftTeacherF3.png)

将分类和回归的伪标签区分开。笔者认为这是一个创新，之前的伪标签方法往往是通过置信度或其他方法同时对两者进行筛选。而这里是分开的，保证了单一标签的更高质量。

从上图中，可以看出，回归质量与IOU的关系更加显著，相比之下，分类的质量和IOU的相关性要弱一些。换而言之，分类得分的高低和回归框的质量没有明显的相关性，因此这里作者提出将两者的质量衡量和筛选机制分开进行。

Box jitter的原理比较简单，即对teacher的candidate box进行扰动，并进行refine，查看其最后输出的variance。对每个维度分别计算方差，并归一化。回归方差小，说明目标框置信度高。否则相反。这里只计算前景得分高于0.5的。计算出置信度并通过筛选，得到用于监督student回归的标签。

