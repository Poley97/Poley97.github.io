---
layout: post
title: 'Range Adaptation for 3D Object Detection in LiDAR'
date: 2021-06-11
author: Poley
cover: '/assets/img/20210611/RA.png'
tags: 论文阅读
---

> 论文链接 : http://openaccess.thecvf.com/content_ICCVW_2019/papers/ADW/Wang_Range_Adaptation_for_3D_Object_Detection_in_LiDAR_ICCVW_2019_paper.pdf


# Introduction 
激光雷达的数据特点决定了其数据点随着距离的增加而明显的degrades，导致在远处物体处检测的性能。本文提出一种cross-range adaptation，来使得远处的物体得到和近处物体类似的性能。

本文实际将其视为一种**cross-domain的迁移学习任务**。将**近距离区域视为sourece domain,远距离区域视为target domain**。

本文的adapation包括local和global两部分，

1. **Global adaptation**： 使用对抗学习来**align the feature in the network feature space**。但是从图像的经验来说，adaptation的结果高度依赖于task有多复杂，比如对于图像（分类？）的结果就很好，但是对图像分割效果就很差。因此，这让我们需要探究对抗学习背后的原理，以及利用起点云的特性。

2. **Local adaptation**: 由于点云目标的尺度并不随物体的距离而发生变化，所以利用这种特性，来挖掘点云空间中，source 和 target ranges中的matched local regions。
   
![](/assets/img/20210611/RAF1.png)

# Related Work

## 3D Object Detection 
略

## Domain Adaptation
主要有两个方向，一个是在一个单独的网络中做domain adaptation，两个域共享所有的参数，同时促使网络去产生domain-invariant features，通过最小化一个额外损失，其用于**鼓励两个域中的近似特征**。

另一个是**对抗学习**，使用一个判别器来判别features的来源，这样domain-invariant features就会被筛选出来。

# Cross-range Adaptation

在**gridded feature space**里，目标是align不同ranges的features，在3D检测网络的某个中间层。即鼓励far-range ovserved objects去产生和near-range similar objects 一致的特征。

![](/assets/img/20210611/RAF2.png)

## Adversarial Global Adaptation

很简单，将区域划分为两部分，分别为near和far，并将其中的特征送入判别器C，来判断features的来源。**这里的特征应该是使用区域内所有的特征，而不仅仅是目标的特征。**

## Fine-grained Local Adaptation

上述的全局对抗自适应提供了global feature alignment，而Fine-grained local adaptation被进一步提出来提升远距离目标的性能。在点云中，物体的尺寸是一致的，而和视角以及位置无关，也就是说,**LiDAR observation pattern 在 near-range areas中similar objects的feature是far-range areas的feature的重复，只是near-range的有更密集的点。**类似的问题在图像中很难解决，因为图像收到视角、光照、尺度等多重影响。

这部分的思想就是让相近的目标具有相近的特征。因此是local的。将远处目标的特征视为近处目标的加权

$$
\begin{equation}
\hat{\mathbf{f}}_{t}=\sum_{i \in \mathcal{N}} w_{i t} \mathbf{f}_{i}
\end{equation}
$$

而这个权重由特征所属于的目标的相似性来决定，

$$
\begin{equation}
w_{i t}=\frac{e^{\left|\mathbf{o}_{i}-\mathbf{o}_{t}\right|}}{\sum_{j \in \mathcal{N}} e^{\left|\mathbf{o}_{j}-\mathbf{o}_{t}\right|}}
\end{equation}
$$

损失如下

$$
\begin{equation}
\mathscr{L}_{l}=\sum_{t \in \mathcal{F}}\left\|\mathbf{f}_{t}-\hat{\mathbf{f}}_{t}\right\|^{2}
\end{equation}
$$

注意，此处切断了$\hat{f_t}$的求导，这是为了防止near-range的特征退化。因为目的是调整远处的特征，而不是近处的。

![](/assets/img/20210611/RAT1.png)