---
layout: post
title: 'BEVFormer v2: Adapting Modern Image Backbones to Bird’s-Eye-View Recognition via Perspective Supervision'
date: 2022-11-29
author: Poley
cover: '/assets/img/20221123/BEVFormerv2.jpg'
tags: 论文阅读
---

作者提出了一个目前基于图像的3D物体检测所存在的一个问题：3D检测无法充分利用图像backbone的最新成果。比如最新的ConvNext-XL在imagenet上Pretrain。尽管参数量比vov99多3.5倍，但是性能仅仅是持平。

作者认为的原因：1、识别和3D场景之间的domain gap；2、3D检测中，backbone后面的Bev encoder和transformer都有很多层，导致backbone的梯度流混乱。

为了解决上述问题，本文提出加入辅助的perspective 3d detection head，直接从backbone的输出上检测3d框并使用对应的辅助Loss,从而强化backbone的学习。进一步的，这可以变为一个two-stage模型，将上述辅助的预测作为一阶段的proposal，并编码到后续的query中。

本文所提出方法的Pipeline如下图所示
![](/assets/img/20221123/BEVFormerv2F1.jpg)

本文的主要创新在于分析并提出了Perspective Supervision。这种监督和之前的3D检测方法对Backbone的监督区别如下图所示
![](/assets/img/20221123/BEVFormerv2F2.jpg)

总体来说，两种监督的区别：dense v.s. sparse and direct v.s. indirect。

一般的方法将所有的img feature都投影到统一的BEV上。之后在根据query,使用deformable自适应的从Bev feature中提取一些local的特征进行聚合。也就是说，Img feature中只有一些部分进入了后续的网络并收到监督。同时，中间还夹杂了很多其他过程，比如view transformation和transformer的decoder和encoder等。即deformable detr head实际上只利用了Img的部分特征，导致对Img的监督并不密集。同时也是间接的。

这个问题其实对现有方法一直存在，但之前不明显是因为之前方法使用的img backbone较小或者具有单目深度的预训练（后者应该是主流，即vov99）。因此，这里提出一个perspective supervision来对Img backbone实现密集的梯度，并且显式的让backbone学习深度信息（通过3D detection的dense prediciton）。

这里加入了Perspective 3d head来直接对img feature 生成dense的3D预测，实现了对Backbone之间而密集的监督，弥补了上述缺点。使得2D Backbone可以快速学习到之前所不具备的深度信息等，进而提高网络的最终性能。

另外的，类似于Trackformer和Transfusion，这里也加入了一个two-stage的结构。注意到，上述的基于图像Backbone的3D物体检测已经可以提供一些相对可靠的结果，因此，参考two-stage的检测流程，这些proposal完全可以作为先验加入到后续的网络中以获得更好的结果。这里的实现方法这部分和dndetr类似，加入一些额外的query，使得query可以更好的索引到目标信息，降低refine的难度，一定程度上也加速了收敛。如下图所示

![](/assets/img/20221123/BEVFormerv2F3.jpg)

这实际上类似于dn-detr，提供的proposal也相当于一定程度上的加噪gt，虽然对应关系没有那么明确，但是相对的也具有更稳定的匈牙利匹配对，一定程度上可以加速收敛。具体方法是，以Proposal的BEV中心位置作为每个图片上的reference points，并添加额外的query，利用这些位置query特征实现对应的预测。(这部分还需要看代码核实一下)

实验结果显示，经过本文的改进，确实可以使得3D检测从更大更好的视觉基础模型中受益，如下所示
![](/assets/img/20221123/BEVFormerv2T1.jpg)

同时，额外监督对于网络Performance的效果是显著的
![](/assets/img/20221123/BEVFormerv2T2.jpg)

并展现出了对不同Backbone一致的性能提升
![](/assets/img/20221123/BEVFormerv2T3.jpg)