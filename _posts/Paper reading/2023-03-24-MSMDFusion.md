---
layout: post
title: 'MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with
Multi-Depth Seeds for 3D Object Detection'
date: 2023-3-24
author: Poley
cover: '/assets/img/20230311/MSMDFusion.jpg'
tags: 论文阅读  
---


主要创新：一个深度估计方法，用于提高depth的估计质量。一个Gated Modality-Aware卷积模块，用于细粒度上融合体素和像素，然后将其聚合到一个unified space。

作者认为，由于不准确的深度估计，以及模态之前的差异。图像和点云之间是存在inconsistency的。为了使得点云尽可能获得图像的丰富语义信息，在一个可控和细粒度的规则下自适应的从图像virtual points内获得信息是必要的。现有的规则是比较简单的，一般是在BEV空间中相加或者concat。这里对此提出两个创新：1、为了获得可靠的深度，参考邻域内的多个深度被探索。2、这里的卷积实现对virtual point 选择-聚合的过程。实现更好的融合特征提取。同时，本文也在fusion中使用了多尺度信息，并产生了显著的性能增益。

本文整体的结构如下
![](/assets/img/20230311/MSMDFusionF2.jpg)


在具体方法上，首先，作者认为，多模态交互的最终目的是以一种合适的方式将多模态特征融合到一个对场景的，Unified，丰富的表征。为了有效的融合，需要图像具有更好的深度估计，形成virtual points在3D空间中。其次，在3D空间中需要有效的融合virtual points和lidar points，在一个可控的，细粒度的条件下。

在深度估计上，本文的方法是，将3D点投影到2D，并且只保留在图像前景中。图像的seed，也同样在图像前景中随机取。并且使用距离其最近的真实深度作为估计深度。
但是这样忽略了图像之间的空间相似度。因此这里从周围的空间取多个参考深度，作为这个点的预测深度。如下图所示

![](/assets/img/20230311/MSMDFusionF3.jpg)

和以往方法直接使用2D语义作为virtual point的类别不同。这里还加入了virtual point与depth的交互，来产生depth-aware 的语义特征，因为这实际上在3D空间中对于语义信息有显著的作用。这里的方法是，显式的用深度信息为depth-aware特征预测一个非归一化的权重，相当于一个attention。如下所示
$$
\begin{equation}
\begin{aligned}
s_i^k & =\operatorname{Sigmoid}\left(\operatorname{Linear}\left(\left[C^d\left[u_i^s, v_i^s\right] ; d_i^{s, k}\right]\right)\right) \\
c_i^k & =C^d\left[u_i^s, v_i^s\right] * s_i^k
\end{aligned}
\end{equation}
$$

之后是对virtual的融合，这里提出的方法称为 Gated Modality-Aware Convolution。总体示意如下
![](/assets/img/20230311/MSMDFusionF4.jpg)

图像经过上述深度估计和投影后，变成相同分辨率的voxel。之后进行细粒度的特征融合。此时有三种特征，v,c和vc。由于Lidar在3D检测中的主导地位，这里使用Lidar作为guding modality来提取其他模态的信息。如下所示

$$
\begin{equation}
\tilde{f_i^C}=\operatorname{ReLU}\left(\operatorname{Linear}\left(f_j^L\right)\right) * f_i^C
\end{equation}
$$
由于Camera和Lidar的Voxel并非一一对应。这里是将最近的Lidar voxel作为cam voxel的ref。因此这里需要进行voxel的pair,不做优化的话，成本较高。


这里优化的方法是，选取一部分cam voxel作为区域的代表。利用这些cam voxel做穷举匹配。之后将其Indices分配给预定义半径内的其他cam voxel。这可以使得穷举的成本线性减少。如下图所示

![](/assets/img/20230311/MSMDFusionF5.jpg)

最后，在多尺度上，即每个尺度的特征分别融合，然后降采样相加即可，如下

$$
\begin{equation}
\hat{F}_{i+1}^M=F_{i+1}^M+\text { DownSample }\left(\hat{F}_i^M\right),
\end{equation}
$$
实验结果上，最后是获得了比较不错的性能。
![](/assets/img/20230311/MSMDFusionT1.jpg)

消融实验也说明了所提出模块的有效性。
![](/assets/img/20230311/MSMDFusionT4.jpg)

![](/assets/img/20230311/MSMDFusionT5.jpg)