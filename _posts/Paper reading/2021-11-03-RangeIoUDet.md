---
layout: post
title: 'RangeIoUDet: Range Image based Real-Time 3D Object Detector Optimized by Intersection over Union'
date: 2021-11-03
author: Poley
cover: '/assets/img/20211101/RangeIoUDet.png'
tags: 论文阅读
---

本文从实时性的角度出发，提出一种高效的单阶段3D目标检测框架，称为RangeIoUDet，使用range image最优输入。其从2D Range Image中提取点特征，送给RPN并得到3D Proposal。通过point-based IoU 和 box-based IoU 监督来优化点特征和3D目标框。并提出 3D Hybrid GIoU来产生高质量的框，同时提供准确的质量评估。

这篇文章主要关注与可以应用于实际应用的算法。目前流行的3D目标检测算法基本基于3D卷积和体素，或者点特征等，复杂度高，推理慢，不适合用于实际应用中。可以实用的算法那应该具有以下特点：
+ 结构简单，易于部署
+ 基于2D卷积
+ 推理速度快

Range Image这种数据表达形式符合上述的特点，由于其实一种compact的数据形式，因此可以直接使用2D卷积来处理，在速度和模型结构上具有优势。但是，之前提出的一些方法由于使用了两阶段方法，因此在速度上优势反而不明显。**这就需要提出一种高性能快速的单阶段3D检测器，基于Range Image。**

Range Image虽然处理素速度快，但是之前提出的framework效果都不是太好。因为Range Image存在一些缺点。最主要的是3D local relationship的缺乏。由于Range Image是投影得到的，因此和图像一样，很远的点和很近的点在投影上可能是相邻的。**由于使用的是2D卷积，因此这些投影面上相邻，实际上相隔很远的点会具有类似的特征，导致不准确的点特征。**

为了解决这个问题，就需要使用额外的loss来监督。本文提出使用**point-based IoU**。

除了点特征，为了提高单阶段目标检测器的性能，设计强力的损失函数来监督是必要的。这里提出了 **3D Hybrid GIoU loss**，通过结合使用 Smooth L1, 3D GIoU 等来时间更好的回归效果。

![](/assets/img/20211101/RangeIoUDetF2.png)

### 关于Range Image

Range Image是点云进行球形投影得到的图像，其分辨率主要取决于激光雷达的水平角分辨率和垂直激光线束。以Velodyne 64E为例，其每条激光上有2048个点，一共64条激光，因此是$64\times 2048$的分辨率。个人认为这也是最符合激光雷达成像原理的数据表达方式（激光雷达可以是做2.5D，因为其和相机一样受到视野遮挡而不能获得完整的3D信息）。每个点包含5维的信息，分别是$(x,y,z)$，range $r$和intensity $e$。特别的，如果只关注相机视野内的场景（正前方90度），并且去除无用的激光线之后，上述尺寸可以缩小成$48 \times 512$，大大提高算法的处理效率。


## Model

如上图所示，本文先在Range Image上用encoder-decoder模型来得到同分分辨率的一副range image，由于其和点云基本是一一对应的而关系，这里每个像素就可以视为通过range image和2D 卷积得到的点特征。之后投影到3D空间，进行点特征层面的额外监督，以及投影到BEV视图中，进行目标检测任务。

![](/assets/img/20211101/RangeIoUDetF3.png)

上述使用2D encoder-decoder得到的点特征做分割来实现对点特征的额外监督。这里发现(a)的效果并不好，因为没有聚合局部特征，在剩下三种中，平行聚合的(d)具有更好的效果，而更深网络的(c)没有，因为(c)导致更更好的分割性能，但是对检测的帮助更小。笔者认为这可能是辅助网咯的一个普遍现象，可以在以后的实践中多加注意。

作者发现smooth L1损失在作为目标框回归损失的时候，具有一定的局限性，如下图所示。在xy位置不准确的情况下，LW的loss降低，反而导致了目标框IOU的下降，即框质量的下降。这是因为smooth L1忽略了目标框回归7个参数之间的相对关系。因此，引入IoU是必要的。
![](/assets/img/20211101/RangeIoUDetF4.png)
首先，由于3D目标框带旋转，因此其IOU的计算比较复杂，作者提出了如下的计算方式：
+ enclose box 使用 pred 和 gt的朝向平均来得到
+ 3D IoU可以通过计算2D带旋转的IoU 之后再乘以高度获得
+ 2D带旋转IoU的交可以通过分割成若干三角形来计算，如下图所示

![](/assets/img/20211101/RangeIoUDetF5.png)

这样可以高效并行的计算IoU，之后使用和GIoU一样的方式计算GIoU3D,如下
$$
\begin{equation}
\mathcal{L}_{G I o U_{3 D}}=1-\frac{A^{i}}{U}+\frac{A^{c}-U}{A^{c}}
\end{equation}
$$

同样，作者发现Smooth L1 loss之所以效果不是最佳，在于其忽略了目标框参数之间的相关性。但是其本身在数值回归上的性能还是很优越。因此这里作者选择结合smooth l1 loss和IOU loss的优点，使用smooth l1 loss单独回归目标框，剩下的维度使用IoU loss来作为监督，如下。
$$
\begin{equation}
\mathcal{L}_{H y G I o U_{r e g}}=\sum_{b \in x, y, z} \mathcal{L}_{\text {smoothL1 }(\Delta b)}+\mathcal{L}_{G I o U_{3 D}}+\alpha \mathcal{L}_{\text {dir }}
\end{equation}
$$

同样，类似于2D检测，这里加入了一个质量评估回归（IoUy预测）。使用上述的3DIoU得到。

![](/assets/img/20211101/RangeIoUDetT1.png)