---
layout: post
title: '3D Dual-Fusion: Dual-Domain Dual-Query Camera-LiDAR Fusion for 3D Object Detection'
date: 2022-12-05
author: Poley
cover: '/assets/img/20221205/3DDualFusion.jpg'
tags: 论文阅读
---


主要创新：1、双query deformable attention来有交互的融合两个domain的信息 2、3D局部自注意力来将Voxel的先验引入dual-query decoding

但是注意到，本文相当于对Voxel Backbone的一个信息增强，在其中融入了img的信息。但是整体架构依旧以Voxel为主，并沿用了Voxel RCNN的pipeline。

其总体结构示意图如下图所示

![](/assets/img/20221205/3DDualFusionF1.jpg)

这里的dual-query实际上是一种跨模态更新feature map的方式。后续可以跟任何的head。本方法也利用LiDAR的稀疏性来降低模型的复杂度和内存消耗。

总的来说，本方法是直接更新3D Sparse voxel backbone中的sparse feature。然后再进行后续的处理。

![](/assets/img/20221205/3DDualFusionF2.jpg)

整体流程相对简单，又三个部分，分别是Adaptive Gated Fusion Network, 3D Local self-attention和dual-query Cross Attention。

Adaptive Gated Fusion实际上是参考之前3D CVF等工作，通过一个gate加权来实现特征的融合。这里作者将其用在了对img feature map的更新上。即将lidar投影到camera并加权融合。

$$
\begin{equation}
\begin{gathered}
A=T\left(F_v\right) \times \sigma\left(\operatorname{Conv}_v\left(T\left(F_v\right)+F_c\right)\right) \\
B=F_c \times \sigma\left(\operatorname{Conv}_c\left(T\left(F_v\right)+F_c\right)\right) \\
F_c^{\prime}=\operatorname{Conv}_r(A \oplus B),
\end{gathered}
\end{equation}
$$

本文的标题是Dual-query，即双query，在Lidar和camera模态各一个。但是从实际处理来看，其img的query更像是一种基于特征的位置编码。


Query分为两个部分，voxel-q和camera-q。v是稀疏的，因此指定query和v的稀疏特征数量相同（一一对应）。c-q和v-q的数量相同，每个代表非空体素在图像上的投影，并用于优化Image-domain的特征。

v-query通过local 的Self attention进行更新。通过ball query或者其他的近邻搜索方式寻找自己的Local group并进行self attention。

Dual-Query Cross Attention则稍微复杂，c-query的初始化使用稀疏voxel特征和对应投影位置的camera特征。并使用DETR中相同的位置编码。使用Deformable-attention来实现对camera-query的更新。但是，其中的deformable的mask是又c-q和v-q共同决定的，即v-q指导c-q query的信息。如下所示
$$
\begin{equation}
\Delta p_{m q k}=F F N\left(q_{c, q}\right), A_{m q k}=F F N\left(q_{c, q}+q_{v, q}^{\prime}\right)
\end{equation}
$$

之后，dual-query同样通过gated fusion进行信息的交互，
$$
\begin{equation}
\begin{gathered}
q_c^{\prime \prime}=q_c^{\prime}+q_v^{\prime} \times \sigma\left(\operatorname{Conv}_1\left(q_c^{\prime}+q_v^{\prime}\right)\right) \\
q_v^{\prime \prime}=q_v^{\prime}+q_c^{\prime} \times \sigma\left(\operatorname{Conv}_2\left(q_c^{\prime}+q_v^{\prime}\right)\right),
\end{gathered}
\end{equation}
$$

如下图所示
![](/assets/img/20221205/3DDualFusionF3.jpg)
最后，c-q以embedding的形式加到v-q上，并从图像特征FC中提取特征。以丰富体素自身的特征。

在性能上，本方法没有体现出非常高的优势，但也是不错的工作。

![](/assets/img/20221205/3DDualFusionT2.jpg)