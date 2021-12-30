---
layout: post
title: 'Geometry-based Distance Decomposition for Monocular 3D Object Detection'
date: 2021-11-18
author: Poley
cover: '/assets/img/20211116/M3OD.png'
tags: 论文阅读
---


本文提出一种基于几何的距离分解方法。通过分解可以鲁邦的距离预测以及可以跟踪不同场景下距离预测不确定度的成因，并且使得预测更准确，鲁棒，可解释性更强。


## Related works
待整理

## MonoRCNN
结构如下所示，其中包括两个3D检测相关的head，分别是3D Attribute Head和3D Distance Head，分别用于回归3D目标的属性、距离和不确定度。
![](/assets/img/20211116/M3ODF2.png)

### Basic Framework

一般单目3D目标检测有两个假设：（1）只考虑3Dbox的yaw角；（2）每个图像对应的相机内参是已知的，不管是训练还是推理的时候。

和基于3D点云的目标检测不同，基于单目相机的目标检测输出的结果较多，分别有
+ 类别标签*cls*和置信度*score*；
+ 2D bounding box，通过左上右下两点表示，$\mathcal{b}=(x_1,y_1,x_2,y_2)$;
+ 3D bounding box的 2D 投影中心，$\mathcal{p}=(p_1,p_2)$;
+ 3D bounding box的物理尺寸，$\mathcal{m}=(W,H,L)$;
+ 3D bounding box的yaw角，$\mathcal{a}=(\sin(\theta),\cos(\theta))$;
+ 3D bounding box的中心在图像坐标系内的位置$Z$。


### 3D Distance Head

本文中的一个创新是基于几何分解了距离估计。如下所示
$$
\begin{equation}
Z=\frac{f H}{h}=f H h_{r e c}
\end{equation}
$$

这将距离估计分解成了两个可解释的变量。$H$代表物体本身的尺寸，这是物体的内在属性。而$h$可以解释称场景中的位置信息，可以通过2D图像回归得到。

同样，回归物体在画面中的高度相比回归物体8个角点在画面中的投影要稳定的多，不容易受到其他因素的影响，如下图所示
![](/assets/img/20211116/M3ODF3.png)

并且物体的高度也是物体的若干尺寸中，最稳定最简单的变量，属于很强的先验知识。对于KITTI val split中的目标尺寸估计误差如下所示。同时目标的长度和宽度会收到yaw角的影响，只有高度永远不受影响。

![](/assets/img/20211116/M3ODT1.png)

这种分解方式得到两个具有强关联的变量，网络在训练过程中可以学习到两者之间的相关性，从而在一个预测不准确的时候，倾向于对应的改变另一个值的预测，最终得到相对准确的距离估计。分析如下，将物体的高度$H$视为一个常量，物体和相机坐标系原点的距离$Z$视为一个随机变量，则对应的$h_{rec}$也是一个随机变量。假设距离$Z$服从一个分部$D$,则可以得到
$$
\begin{equation}
Z=f H h_{r e c} \sim \mathcal{D}
\end{equation}
$$

两边求期望，得到
$$
\begin{equation}
H \mathbb{E}\left[h_{r e c}\right]=\frac{\mathbb{E}[Z]}{f}
\end{equation}
$$
这说明$H$和其对应的$h_{rec}$的期望的乘积是一个常数，注意到，这个关系实际上是类别无关的。因此物体的距离$Z$和物体的类别无关，因此上述关系对所有类别的物体都适用。

两者的相关关系可视化如下所示，通过计算相关系数也可以得到两者很强的负相关性，说明网络已经学习到了这种特性。

![](/assets/img/20211116/M3ODF4.png)

在不确定度的估计上，直接基于uncertainty-aware regression loss来回归上述两个高度，如下
$$
\begin{equation}
\begin{gathered}
L_{H}=\frac{L_{1}(\hat{H}, H)}{\sigma_{H}}+\lambda_{H} \log \left(\sigma_{H}\right) \\
L_{h_{r e c}}=\frac{L_{1}\left(\hat{h}_{r e c}, h_{r e c}\right)}{\sigma_{h_{r e c}}}+\lambda_{h_{r e c}} \log \left(\sigma_{h_{r e c}}\right)
\end{gathered}
\end{equation}
$$

![](/assets/img/20211116/M3ODF5.png)

可以看到，当物体距离非常近或者很远时，其物理高度的不确定度会增加。这也符合我们日常生活中的认知。在近距离时，不确定度主要来自于物体被截断的视角。而在远处时，不确定度主要来自于图像中过少的像素。

### Experiments
![](/assets/img/20211116/M3ODT3.png)
![](/assets/img/20211116/M3ODF7.png)