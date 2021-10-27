---
layout: post
title: 'End-to-End Object Detection with Fully Convolutional Network'
date: 2021-10-21
author: Poley
cover: '/assets/img/20211018/EEOD.png'
tags: 论文阅读
---

> 参考博客：https://blog.csdn.net/A_A666/article/details/110821589

本文提出一种去除NMS，实现End-to-End训练的目标检测网络。文中通过实验发现，如何分配一个合适的label是这其中的关键。因此提出了一种**Predection-aware One-To-One (POTO)**费配给制。同时，还提出**3D Max Filtering(3DMF)** 来利用多尺度特征。最终，本文提出的网络结构达到了媲美sota的性能。 

作者认为 One-to-One label assignment对于消除重复框非常重要。但是固定的assignment可能会导致模糊或者非最佳的训练。为了解决这个问题，POTP动态的分配前景样本，同时根据分类和回归的质量。FPN可以使得网络学习到更丰富的特征，之前的工作表明，大量的重复矿都来自于同一尺度上的临近区域。因此坐着提出了3DMF，来提升卷积在局部区域内的判别能力。同时为了学习到更加丰富鲁棒的特征，使用了一对多的assignment作为辅助损失。

![](/assets/img/20211018/EEODF1.png)

对于不同的Assignment的对比如上表所示。可以看到，一般的One-to-One会有更好的无NMS性能，但是有nms的性能不如one-to-many。同样，及时One-to-One的标签分配，其有无NMS的性能差距仍然是不可忽略的。由于One-to-One使得每个实例受到更少的监督，性能也出现了下降。

本文通过混合的标签分配方式弥补了这一点，3DMF+POTO+多标签辅助Loss。
![](/assets/img/20211018/EEODT1.png)
## POTO
目标检测的损失可以泛化的表示为如下形式，分别由前景的分类和回归损失以及背景的分类损失构成。
$$
\begin{equation}
\mathcal{L}=\sum_{i}^{G} \mathcal{L}_{f g}\left(\hat{p}_{\hat{\pi}(i)}, \hat{b}_{\hat{\pi}(i)} \mid c_{i}, b_{i}\right)+\sum_{j \in \Psi \backslash \mathcal{R}(\hat{\pi})} \mathcal{L}_{b g}\left(\hat{p}_{j}\right)
\end{equation}
$$

为了实现End-to-End的匹配，需要建立一个合适的label assignment。可以将其视为一个二分匹配问题，通过匈牙利算法求解，如下
$$
\begin{equation}
\hat{\pi}=\underset{\pi \in \Pi_{G}^{N}}{\arg \min } \sum_{i}^{G} \mathcal{L}_{f g}\left(\hat{p}_{\hat{\pi}(i)}, \hat{b}_{\hat{\pi}(i)} \mid c_{i}, b_{i}\right)
\end{equation}
$$

这样分配的通常需要额外的权重来缓解优化问题。本文提出POTO来根据网络的预测结果动态的调整assignment，如下

$$
\begin{gathered}
\hat{\pi}=\underset{\pi \in \Pi_{G}^{N}}{\arg \max } \sum_{i}^{G} Q_{i, \pi(i)}, \\
\text { where } Q_{i, \pi(i)}=\underbrace{\mathbb{1}\left[\pi(i) \in \Omega_{i}\right]}_{\text {spatial prior }} \cdot \underbrace{\left(\hat{p}_{\pi(i)}\left(c_{i}\right)\right)^{1-\alpha}}_{\text {classification }} . \\
\underbrace{\left(\operatorname{IoU}\left(b_{i}, \hat{b}_{\pi(i)}\right)\right)^{\alpha}}_{\text {regression }} .
\end{gathered}
$$

Here $Q_{i, \pi(i)} \in[0,1]$ represents the proposed matching quality of the $i$-th ground-truth with the $\pi(i)$-th prediction. It considers the spatial prior, the confidence of classification, and the quality of regression simultaneously. $\Omega_{i}$ indicates the set of candidate predictions for $i$-th ground-truth, i.e., spatial prior.

这相当于网络预测置信度和IoU的加权几何平均。上述公式同时考虑了**空间先验，分类置信度以及回归质量**。文中默认设置为$\alpha=0.8$,说明上述几何平均还是以IoU为主，但是依旧加入了网络置信度的成分。**如之前的表格所示，POTO大幅缩小了有无NMS的性能差距，并且提升了one-to-one的性能。**

## 3D Max Filtering

![](/assets/img/20211018/EEODT2.png)

作者进一步探究了重复目标框的来源。作者发现当将NMS变为在单个scale上进行，检测器的性能出现明显的下降。同时，在单个scale上，又设置了不同的抑制范围。通过上述实现，**可以发现大多数重复框都是在最高置信度框的同尺度空间邻域内产生的**。

卷积是对局部的线性加权，因此对于不同位置的模式可能会相同。这是去除重复框的一大阻碍，因为一个实例内的特征都比较近似。 Max filter是一个非线性滤波器，可以用于补偿卷积在Local region判别能力不足的问题。本文将 Max filter拓展到3D上，变为3D Max Filtering。如下所示
$$
\tilde{x}^{s}=\left\{\tilde{x}^{s, k}:=\operatorname{Bilinear}\left(x^{k}\right) \mid \forall k \in\left[s-\frac{\tau}{2}, s+\frac{\tau}{2}\right]\right\}
$$

![](/assets/img/20211018/EEODF3.png)

$$
y_{i}^{s}=\max _{k \in\left[s-\frac{\tau}{2}, s+\frac{\tau}{2}\right]} \max _{j \in \mathcal{N}_{i}^{\phi \times \phi}} \tilde{x}_{j}^{s, k}
$$

分别在邻域范围和scale范围上取max，得到最有判别性的特征。

## Aux loss

使用了上述POTO和3DMF之后，算法性能仍然弱于使用了NMS的FCOS。作者认为这可能是由于one-to-one assignment提供的监督有限。因此，作者使用了ATSS作为辅助损失，在3D MAX之前，用于dense prediction的监督，总体结构如下所示。

![](/assets/img/20211018/EEODF2.png)


## Experiments

![](/assets/img/20211018/EEODF4.png)

可以看到，和FCOS相比，本文提出的方法具有相当小的激活区域，这为去除冗余框做出了关键的贡献。


最终，在COCO数据上，不使用NMS的情况下达到了和FCOS相同的性能水平。