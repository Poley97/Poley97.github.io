---
layout: post
title: 'InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions'
date: 2022-11-29
author: Poley
cover: '/assets/img/20221123/InterImage.jpg'
tags: 论文阅读
---


目前的CV领域中，vision foundation model(视觉基础模型，一般为超大模型和数据规模的模型)中，Transformer-based的模型占据了Top performance的主流，而CNN的性能则要略逊一筹。本文提出，通过使用类似的算子level和架构level的设计，cnn-based的模型同样也可以从更大规模的参数和数据中获得更强的性能。就像现有的众多超大规模ViT模型一样。


抛开一些小网络结构，比如LN,FFN,GELU等，不同的网络结构性能的主要区别来自于其core operator，比如卷积、attention等。目前常见的集中core operator的对比如下图所示。

![](/assets/img/20221123/InternImageF1.jpg)

一般来说，global attention的性能和泛化性主要来自于长距离依赖和自适应聚合。但是需要付出很高的成本。相比之下,local attention和large kernel分别只具有其中的一个特性。而本文提出的dynamic sparse kernel，作为一种新的core operator,特点在于：
+ 灵活的动态学习合适的感受野，可以是近距离或者远距离的。
+ offsets和scalar可以自适应的随着输入数据调整
+ 卷积核是常规的3*3，避免优化问题和巨大的计算量

这使得CNN网络在1B参数量和400M训练图片这样的量级上可以具有与Transformer近似的性能。


首先，作者分析了卷积和MHSA相比主要的缺陷在于两个地方：
1、Long-range dependencies。更大的有效感受野更有利于下游任务。通过3*3卷积堆叠的感受野还是相对较小。
2、自适应孔间距聚合。MHSA的聚合权重是根据输入动态调整的，而常规卷积是静态权重，并具有很强的归纳偏置，比如2D局部性、近邻结构和平移不变性等等。这些归纳偏置使得CNN可以收敛的更快，不需要那么多数据，但是也限制了CNN进一步学习更加泛化和鲁棒的特征，从更大scale的数据当中。

目前，CNN中常用的DCNV2是比较接近上述MHSA的优势的。DCNv2具有：1、带offset的邻域采样；2、对每个采样点的scale缩放；3、对采样点加权。可以看出，其中pk是比较灵活的offset，可以建立短距离和长距离依赖的交互。

但是需要注意到，通常DCNV2作为常规卷积的拓展，读取常规卷积的预训练模型并进行fine-tuning。但是这不适合大规模基础模型——其需要trained from scratch。因此这里提出了DCNV3，相比DCNV2，主要有以下几个改进

1、这里参考分离卷积，将卷积中的权重拆分为位置相关权重和共享的位置无关权重。DCNV2中linear projection neurons随着采样点数量变多而增加，从而增加计算负担的问题。这里将其分为depth-wise的权重，对location-aware响应，生成scalr。和一个共享的Point-wise的权重，生成权重W。

2、参考multi-head，对一个卷积层内引入多个offset group，来聚合不同尺度的信息。

3、使用softmax代替sigmoid来获得卷积内采样点归一化的权重。使得网络输出的尺度更加稳定。


通过这些设计，堆叠出了超大模型，如下图所示。并在一些基础的视觉任务上取得了较好的结果

![](/assets/img/20221123/InternImageT1.jpg)

![](/assets/img/20221123/InternImageT3.jpg)