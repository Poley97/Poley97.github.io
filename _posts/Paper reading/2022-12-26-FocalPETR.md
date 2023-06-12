---
layout: post
title: 'Focal-PETR: Embracing Foreground for Efficient Multi-Camera 3D Object Detection'
date: 2022-12-26
author: Poley
cover: '/assets/img/20221226/FocalPETR.jpg'
tags: 论文阅读  
---

现有方法中，主要分为显式几何的模型（比如LSS）和隐式几何模型（比如PETR）。前者通过模型参数显式的基于图像特征构建3D特征，而后者通过位置编码的相关性隐式的引入图像的3D几何信息。LSS-Based的方法比较具有较好的性能优势，但LSS这套对高并行的GPU不是太友好，因为其包含了很多复杂的indexing operation。PETR这样的并行能力很强，但是训练时间长，显存需求大。作者这里提出的方法主要是针对PETR的改进，在继承其优秀并行性的同时。
这里认为这是由于训练目标函数和query聚合的信息存在不一致导致的。显式距离估计中通过额外的监督来对齐特征，而PETR里是没有的。

作者认为，PETR两个问题：语义模糊，前景的query没有和前景建立很强的关联性。并且初始化位置距离前景都很远。
空间misalignment。几何线索只用来做位置编码而没有座位object query所收集的信息。具体如下图所示

![](/assets/img/20221226/FocalPETRF2.jpg)

对此，作者提出主要改动：
1. 添加三个辅助任务，实现对token的instance-guieded。
2. 使用前景token代替全部token，实现速度的提升。
3. 将空间信息embed到Image token里。

本文提出的网络结构整体如下

![](/assets/img/20221226/FocalPETRF3.jpg)

首先回顾一下PETR中query特征的方式，其key，value的设置如下

$$
\begin{equation}
\begin{gathered}
{[\boldsymbol{k}, \boldsymbol{v}]=\left[\boldsymbol{F}_j^{u, v}+\boldsymbol{E}_j^{u, v}, \boldsymbol{F}_j^{u, v}\right]} \\
\boldsymbol{q}_L=\operatorname{Softmax}\left(\boldsymbol{q}_{L-1} \cdot \boldsymbol{k}^T\right) \cdot \boldsymbol{v}
\end{gathered}
\end{equation}
$$

这里作者认为，只使用几何信息做位置编码，相当于aggregated的信息是geometry-ignored的。

之后，针对前文提到的问题，作者提出了相应的改进，首先是针对query位置和前景信息相关性差的问题。

如上图所示，这里主要改动是加入了三个辅助任务，并使用其中的cls和iou来进行Img feature前景点的采样。具体实现细节上，这里分为三部分：
+ Class-aware Sampling：使用FCOS的head，进行cls和reg任务。其中，reg只作为辅助loss，而cls还负责对图像的前景进行采样。这里使用匈牙利匹配进行正样本匹配，使得此head可以产生高召回率又无重叠的目标框。
+ IoU-aware Sampling: 加入一个GIoU的预测，并对Class Branch做指数加权
+ Centroid-aware Sampling: PETR相当于通过position embedding隐式的学习相机和3D点之间的投影关系，因此这里加入2.5D的辅助loss，认为可以对上述隐式的学习过程起到帮助。其Gaussian Heatmap如下（这是否就是Centerness的heatmap？）
$$
\begin{equation}
\mathcal{H}=\exp \left(-\frac{\left(x-c_x\right)^2+\left(y-c_y\right)^2}{2 \delta_{\min \left(l^*, t^*, r^*, b^*\right)}}\right)
\end{equation}
$$

其次，针对query几何信息缺失的问题，这里提出Spatial Alignment Module。这里似乎只是简单的以相机内参（intrisic）和图片中对应位置的射线（tracing ray）来作为输入，为原有特征回归出一个权重和偏置，如下所示
$$
\begin{equation}
\boldsymbol{F}_j^{* u, v}=\mathcal{T}_w\left(\boldsymbol{r}_j^{u, v}, \boldsymbol{f}_j\right) \cdot \boldsymbol{F}_j^{u, v}+\mathcal{T}_b\left(\boldsymbol{r}_j^{u, v}, \boldsymbol{f}_j\right)
\end{equation}
$$
注意到，根据消融实验，这里的ray有两种选择：1、只包括ray的方向；2、包括ray的方向，以及其view的所有volume。但是性能上没啥区别。
![](/assets/img/20221226/FocalPETRT5.jpg)

在保留原有的position embedding的同时，上述加权特征将被直接作用于K,V特征，如下图所示
![](/assets/img/20221226/FocalPETRF4.jpg)
由于上述对img前景点进行了采样，因此很自然的，网络具有了更快的性能。同时，也使得query将注意力更多的集中在前景点上，加快了网络的收敛。即，实现了性能和训练速度的同时提高，如下图所示
![](/assets/img/20221226/FocalPETRF5.jpg)

具体的性能对比量化如下图所示

![](/assets/img/20221226/FocalPETRT1.jpg)

在采样率上，通过引入辅助分支对img进行前景采样，可以在很小的性能损失下，极大的提升模型的效率

![](/assets/img/20221226/FocalPETRT4.jpg)
