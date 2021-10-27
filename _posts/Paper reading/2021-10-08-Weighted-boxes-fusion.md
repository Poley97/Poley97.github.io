---
layout: post
title: 'Weighted boxes fusion: Ensembling boxes from different object detection models'
date: 2021-10-08
author: Poley
cover: '/assets/img/20211008/WBF.png'
tags: 论文阅读
---

> 论文链接: https://arxiv.org/pdf/1910.13302.pdf
> 参考博客: https://blog.csdn.net/weixin_42990464/article/details/108204282
> 源码: https://github.com/ZFTurbo/Weighted-Boxes-Fusion

本文提出了一种新的目标检测框后处理方式，称为Weighted boxes fusion，其算法中使用了所有proposed bounding boxes的得分。其在2D测试（COCO）和3D测试（Waymo Open Dataset; Lyft 3D Object Detection for Autonomous Vehicles challenges）取得了相比NMS和soft-NMS等更好的性能。

其作者将源码包装成了pip库，使用起来非常方便。

+ NMS 
![](/assets/img/20211008/WBFF1.png)
NMS选择最高置信度的框，并且使用一个固定的IoU阈值来抑制和其有重叠的框。这可能会影响模型的性能，如上图所示，如果有一系列side by side的目标，那么其中就会有框被消除。这降低了模型的预测精度。

+ Soft-NMS

为了防止上述出现的问题，相比完全移除具有高confidence和IoU的目标框，其根据IoU，成正比的减小其他框的置信度，再通过一个置信度阈值来滤除多余的框。但这样还是会导致一些正确的目标框被消除。及时如此，Soft-NMS依旧展现出很大的提升，相比NMS，在TTA的使用场景下。

**NMS和Soft-NMS的这个缺点已经在对抗攻击中被使用了。**

+ Weighted Boxes Fusion

WBF的步骤如下

1. 将所有预测框（来自不同模型，或同一模型不同增强数据）加入一个列表$\bold{B}$中，按照置信度$\bold{C}$降序排列；
2. 声明空列表$\bold{L},\bold{F}$来用于Boxes clusters 以及 fused boxes。$\bold{L}$中的每个位置都可以包含一个Boxes合集，代表一个聚类；$\bold{F}$只包含一个Box，代表fused box，对应$\bold{L}$中的对应位置。产生fused boxes的方法在之后的步骤中说明；
3. 在$\bold{B}$中循环遍历，寻找和$\bold{F}$中匹配的box。这里的匹配指的是IoU大于某个阈值$\bold{THR}$，文中使用的是$\bold{THR}=0.55$；
4. 如果没有找到匹配，就将这个Box加入$\bold{L},\bold{F}$作为一个新的入口，然后继续处理$\bold{B}$中的下一个box；
5. 如果找到了，就将此bbox加入和$\bold{F}$对应的$\bold{L}$的位置$\bold{pos}$中；
6. 重新计算$\bold{F}[\bold{pos}]$处bbox的坐标，置信度得分，使用$\bold{T}[\bold{pos}]$聚类中所有的box进行融合，公式如下
$$
\begin{equation}
\mathbf{C}=\frac{\sum_{i=1}^{\mathbf{T}} \mathbf{C}_{i}}{T}
\end{equation}
$$
$$
\begin{equation}
\mathrm{X} 1,2=\frac{\sum_{i=1}^{\mathrm{T}} \mathbf{C}_{i} * \mathbf{X} 1,2_{i}}{\sum_{i=1}^{\mathrm{T}} \mathbf{C}_{i}}
\end{equation}
$$
$$
\begin{equation}
\mathbf{Y} 1,2=\frac{\sum_{i=1}^{\mathrm{T}} \mathbf{C}_{i} * \mathbf{Y} 1,2_{i}}{\sum_{i=1}^{\mathbf{T}} \mathbf{C}_{i}}
\end{equation}
$$

即平均置信度，以及置信度加权平均的坐标。这里的加权可以使用非线性权重，比如$C^2,sqrt(C)$等等。这样，拥有大置信度的box可以对fused box 坐标做出更大的贡献。
7. 当$\bold{B}$中所有的boxes都被处理完之后，re-scale $\bold{F}$中的置信度。使用聚类中的Bbox数除以模型数做平均，因为如果一个Box只被少数几个模型检测到，这说明其是不太可靠的。如下。
$$
\begin{equation}
\mathbf{C}=\mathbf{C} * \frac{\min (T, N)}{N}
\end{equation}
$$
或者
$$
\begin{equation}
\mathrm{C}=\mathrm{C} * \frac{T}{N}
\end{equation}
$$

NMS和Soft-NMS都排除了一些Box，而WBF方法使用了所有的boxes，因此，其可以修正多模型都预测不准的情况，如下图所示，WBF可以通过平均来获得更准确的目标框。

![](/assets/img/20211008/WBFF2.png)
