---
layout: post
title: 'Cascade R-CNN: Delving into High Quality Object Detection'
date: 2021-05-20
author: Poley
cover: '/assets/img/20210522/CascadeRCNN.png'
tags: 论文阅读
---
> 论文链接：https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html
# Introduction 

目标检测是一个复杂的问题，主要可以分为两部分。首先是识别问题（recognition）,然后是定位(localization)。两个问题都有一定的难度，因为detecor需要面对很多近似的false positives(其实就是IoU相对低，但是训练中被认为是正样本的样本)，也就是"close but not correct" bounding boxes。所以detector就需要找到ture positive，从这些close false positives中。

目前很多目标检测器都基于two-stage R-CNN框架，这个框架中，classification 和 regression 被认为是两个task，然后做一个multi-task learning。但是和目标识别的不同，IoU $u$是判断一个样本正负的基准，典型的值是$u=0.5$。但是图1所示，大部分人类会吧$IoU>0.5$的当作close false positive。

![图1](/assets/img/20210522/CascadeRCNNF1.png)

由于在$u=0.5$下采样得到的样本非常的多而且多样化，所以detecotes也很难有效的去拒绝close false positives。

这里将hypothsis(也就是输入patch)的quality定义为和真值的IoU，detector的quility定义为其训练使用的Iou。其得到的曲线如图一所示，其基本的思想就是一个单独的detecotr只会对单独的一个level是optimal的。可以看到，每个不同quality的detector都对和其自身quility相近的输入具有最大的提升（也就是最大的IoU提升）。也就是说，heigher quality detection 需要一个近似quality 的hypotheses来匹配它，才能达到最好的效果。

但是，从上图也可以看出，对于$u=0.7$的检测器，实际上性能是不如剩下两个的。这是由于hypothese的分布实际上式非常不平衡的，偏向于低质量。所以，强行提升IoU阈值会导致正训练样本数量的指数下降，这样使得"high $u$" training 很容易过拟合，另一个问题就是 训练和预测时候的hypothese的不一致，综上所述，high quality detecors are only necessarily optimal for high quality hypothese。

本文提出Cascade R-CNN,来解决上述的问题，分为多个stage ，使用上一个Stage的输出来训练下一个Stage。这也是受到了上述实验的启发，也就是回归器的输出普遍具有比输入更高的IoU，见图一。这就说明通过一个certain IoU训练的detector的输出是对于下一个阶段更高IoU训练的detector的一个好的分布。

# Related Work

略

# Object Detection

## Bounding Box Regression

一般的回归的coding方法如下

$$
\begin{equation}
\begin{array}{l}
\text { vector } \Delta=\left(\delta_{x}, \delta_{y}, \delta_{w}, \delta_{h}\right) \text { defined by }\\
\begin{aligned}
\delta_{x}=\left(g_{x}-b_{x}\right) / b_{w}, & \delta_{y}=\left(g_{y}-b_{y}\right) / b_{h} \\
\delta_{w}=\log \left(g_{w} / b_{w}\right), & \delta_{h}=\log \left(g_{h} / b_{h}\right)
\end{aligned}
\end{array}
\end{equation}
$$

但是回归经常用于进行minor adjustments，所以这个数值会变得很小，导致回归loss一般远远小于classification loss。所以使用一个先验的$\Delta$来对上述做一个放大（标准化）。

$$\begin{equation}
\delta_{x}^{\prime}=\left(\delta_{x}-\mu_{x}\right) / \sigma_{x}
\end{equation}$$

有一些方法认为一次回归对于检测器不够，于是多了很多次，成为$iterative bounding box regression$，如下图2所示。

![图2](/assets/img/20210522/CascadeRCNNF3.png)

但是就像图1所示的，这些回归器都是针对一个低的IoU训练的，比如$u=0.5$，所以实际上它对于高的Iou输入时次优的，比如图1中$u=0.5$的曲线，甚至在IoU>0.85的时候性能会发生退化。所以，Usually there is no benefit beyond applying f twice。

# Detection Quality

一般proposal的真值是通过IoU阈值给定的

$$
\begin{equation}
y=\left\{\begin{array}{cl}
g_{y}, & I o U(x, g) \geq u \\
0, & \text { otherwise }
\end{array}\right.
\end{equation}
$$

目标检测的矛盾就在于，无论用什么IoU阈值，太高了就会使得正样本很少。反之，则会让检测器没有办法收到足够的刺激来抑制close false positives。

一般常用的proposal network，比如RPN通常具有Low quality，所以detector必须可以对于low quality hypothese具有更高的判别性。所以，这导致detector会产生很多人认为是close false positves的检测结果。


一个简单的想法是通过多个判别器来依次使用不同的IoU来训练判别样本。如下
$$
\begin{equation}
L_{c l s}(h(x), y)=\sum_{u \in U} L_{c l s}\left(h_{u}(x), y_{u}\right)
\end{equation}
$$

但是这种方法没有考虑到不同判别器接收到的正样本数量差异。

# Cascade R-CNN

和上述不同的，Cascade R-CNN使用多个不同的回归器来对结果进行多阶段的分类和回归，如图2所示。即

$$
\begin{equation}
f(x, \mathbf{b})=f_{T} \circ f_{T-1} \circ \cdots \circ f_{1}(x, \mathbf{b})
\end{equation}
$$

每个回归其都是对于其Stage resampled 分布的最优回归器。考虑到不同层预测的偏移量不同，越高的层实际预测出来的偏移量越小，所以可以对每层设置不同的方差$\Delta$，之前所说的，强化回归的Loss。

![图3](/assets/img/20210522/CascadeRCNNF4.png)


由于RPN自身是low quality的，所以会不可避免使得高quality的无效学习。Cascade通过resampling mechanisem来解决这个问题。这是受到图1的启发。每Stage都给出一组bbox$(x_i,\bold{b}_i)$,然后下一个Stage再从中进行采样，这样就可以保持positive examples的数量。让每一个Stage的正样本都很丰富，提高了训练质量。每层都Loss都单独计算，如下

$$
\begin{equation}
L\left(x^{t}, g\right)=L_{c l s}\left(h_{t}\left(x^{t}\right), y^{t}\right)+\lambda\left[y^{t} \geq 1\right] L_{l o c}\left(f_{t}\left(x^{t}, \mathbf{b}^{t}\right), \mathbf{g}\right)
\end{equation}
$$

