---
layout: post
title: 'Pseudo-labeling for Scalable 3D Object Detection'
date: 2021-10-09
author: Poley
cover: '/assets/img/20211008/PseudoLabeling.png'
tags: 论文阅读
---

> 参考博客: https://zhuanlan.zhihu.com/p/357805444
> 论文链接: https://arxiv.org/abs/2103.02093

# 相关工作
一些 Domain adaptation的工作表明，在自动驾驶数据集上，进行跨数据集的测试时（在一个数据集上训练，并在另一个数据集上测试），模型会出现显著的性能下降。这些性能下降一半被归咎于物体大小尺寸的区别，并且可以通过实现考虑物体尺寸的区别来减小这种下降的幅度。但是最近的另一些研究表明，单个数据集中，跨地理位置的数据同样出现准确性的下降，这就不能用尺寸的差异来解释了。本文中在单一数据集上进行实验，并且可以缓解跨地域检测的性能下降问题。

# 方法

方法比较简单，如下图所示。首先在标注数据上做一个训练，并对未标注数据进行伪标签标注；之后使用所有的伪标签和标注数据一起对学生模型进行训练。

![](/assets/img/20211008/PseudoLabelingF2.png)

注意，这里在划分数据集的时候是按run segment划分的，而不是Frame。比如选择10%的标注数据，指的是选择10%的run segment进行标注，而不是10%的frame。因为同一个run segment的邻近帧之间相似度很高，尤其是在车开的不快的时候。如果选择frame的比例，则会让task变得更简单。

训练中基于PointPillars，但是使用了multi-frame进行训练。**即将前N-1帧的点云数据都变换到最后一帧的坐标系中，并将所有的数据concatenate起来。**这使得点云数据打打丰富，使得网络更容易检测到，以及更高质量的伪标签。

**个人认为通过这种方式来产生高质量的伪标签非常的关键，这可以大大缓解点稀疏和遮挡的问题，降低检测难度，但是这种方法应该只能应用于sequence的数据集中，比如waymo。**

使用置信度阈值来筛选伪标签，对大多数模型，一般使用0.5来划分。对于一些校准度差和系统低置信度的模型，比如多帧Pedestrian models，使用更低的置信度。

这里的teacher和student并没有权重之间的关系，比如EMA，而是只通过标签联系。因此，很直观，**Better tracher lead to better students**。而数据增强可以较好的增加Teacher的训练效果，如下表所示。

![](/assets/img/20211008/PseudoLabelingT1.png)

具体实验结果如下，可以看出，伪标签方法在标注数据占较小比例的时候，本方法有很大的提升。而当标注数据的比例上升时，本方法的效果明显下降。
注：Waymo数据集的训练segments数量为798，Kirkland run segments数量为560。此处分别使用$10\%,20\%,30\%,50\%,100\%$的原始训练数据。$100\%$的原始训练数据在图中对应的比例约为$59\%$。
![](/assets/img/20211008/PseudoLabelingF4.png)
![](/assets/img/20211008/PseudoLabelingF5.png)

可以看到，当Teacher的宽度增加之后，在高训练数据比例下，性能反而出现了下降。

但是通过在无标签数据上创建伪标签进行训练，提高了在无标签上数据上的性能。但是这个性能仍然远低于在waymo验证集上的性能，作者猜测这可能是由于数据分布差异导致的。

![](/assets/img/20211008/PseudoLabelingF6.png)

**自动驾驶领域的重要问题是，在有限的推理时间和固定数量的标注数据下，如何达到更好的性能**。由于实际自动驾驶领域的未标注数据应该远远大于标注数据，因此本文提出的伪标签方法可能是一种比较有效的方式，来将一个复杂的offline模型，通过伪标签，蒸馏到一个高效，实用的模型中。

作者授权获得了更多的waymo未标注数据，展现出在超大未标注数据上，模型性能可以进一步提升。这可能对于实际应用中更有意义。

![](/assets/img/20211008/PseudoLabelingT2.png)

![](/assets/img/20211008/PseudoLabelingT3.png)

# Negative result
作者罗列了一些失败的尝试

+ Soft-lables：前景 $[0,1]$，背景 $1$，性能出现轻微的下降。
+ Multiple iterations： 尝试更多的训练迭代次数，性能略有提升，但是时间消耗太大。
+ Score threshold： 作者尝试了是否要对伪标签的置信度中间设置一个ambiguous range $[lower,upper]$,不分配对应的loss（就像目标检测的正负roi样本一样）。经过一系列尝试，$[0.5,0.5]$取得了最优的效果，证明这个ambiguous range是不需要的。