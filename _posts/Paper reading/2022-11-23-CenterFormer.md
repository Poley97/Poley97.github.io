---
layout: post
title: 'CenterFormer: Center-based Transformer for 3D Object Detection'
date: 2022-10-27
author: Poley
cover: '/assets/img/20221123/CenterFormer.jpg'
tags: 论文阅读
---

本文发表于ECCV2022，主要创新是通过时序信息来提升3D检测性能。在检测pipeline上，其使用了detr-based结构。首先通过heatmap选择潜在的center candidate（有效减少query数量），之后通过cross attetnion实现多帧信息的聚合。通过head对query解码得到最终结果。

DETR两个缺点：计算复杂度平方增长，和难以收敛。由于3D检测中，目标在BEV中的比例较小，不能进行非常高倍率的下采样。而DETR的计算复杂度又较高，使得其不好应用于3D检测当中。为了节约计算量，本方法限制注意力的学习区域到一个较小范围内。针对收敛慢的问题，本问题出使用一阶段的预测proposal center来作为query的embedding，使其更容易找到对应的物体和所需要的信息，加快收敛速度。

本文的主要创新：多尺度proposal网络 + proposal初始化的query + deformable cross attention +  multiframe特征聚合。

本文的总体流程如下所示。
![](/assets/img/20221123/CenterFormerF2.jpg)
上图的重点主要还是在cross-attention layer的设计，即如何有效的query多尺度特征。如下图所示


![](/assets/img/20221123/CenterFormerF3.jpg)
为了避免DETR全局attention带来的计算量，这里提出Attending key，即对于一个query的位置，在其在多scale上索引的位置，直接取3*3的邻域来获得多尺度特征。对于多时序的feature map，同样是采用这种投影邻域attending的模式。但是作者提出，对于速度快的物体，其帧间的位置差异较大，因此这样固定的邻域可能不能索引到物体相关的信息（物体已经在其他位置了）。因此，这里提出在上述反复噶的基础上，进一步引入deformable attention,来在不同的scale和timestamp上自适应的选择邻域。

最后，是时序特征的引入，这里比较简单。作者认为，由于我们关心的是当前时间上物体的位置（heatmap），因此应该是根据当前时间的感兴趣位置从过去帧提取特征。这通过一个注意力权重实现，最后，通过一个卷积来对时序信息做fusion，如下图所示。
![](/assets/img/20221123/CenterFormerF4.jpg)


本文的结构比较简单，因此实际上在单帧性能上并不突出。但是transformer query的Pipeling很适合提取timestamp上的信息，这使得本方法在时序上具有较大的增益。在waymo数据集上的训练结果如下
![](/assets/img/20221123/CenterFormerT1.jpg)

![](/assets/img/20221123/CenterFormerT2.jpg)

进一步的消融实验，个人感觉这里作者做的不是很好。因为其引入transformer，尤其是deformable的优势在于对时序信息灵活的融合。但是这个消融是做在single frame上的，这显得deformable就没啥用了。
![](/assets/img/20221123/CenterFormerT4.jpg)