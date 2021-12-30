---
layout: post
title: 'CenterNet：Keypoint Triplets for Object Detection'
date: 2021-12-22
author: Poley
cover: '/assets/img/20211222/CenterNet.png'
tags: 论文阅读
---

本方法是在CornerNet上的改进，主要解决CornerNet存在的两个问题：
+ CornerNet会产生大量的错误目标框，不好进行筛选；
+ CornerNet对于边缘信息很敏感，而不能收集到足够的关于目标框内特征的信息；

为此，作者提出了两个解决办法
+ Center Perdiction和Center Pooling： 通过引入Center Keypoint来使用三个点判断一个目标框而不是两个。为了收集到足以确定CenterPoint的信息，作者引入Center pooling来解决。
+ 针对Corner点对于边缘敏感的问题，作者将Center和Corner pooling级联起来，使得Corner也收集到目标框内的信息。

作者分析发现，CornerNet存在错误目标框过多的问题，因为其更关注边缘，而关注不到目标框内部的信息。因此作者认为通过引入位于目标框中心区域的关键点可以有效的协助Corner判别目标框正确性。如下图所示

![](/assets/img/20211222/CenterNetF1.png)

故引入Centerpoint的预测以及其相应的机制。网络的整体架构如下

![](/assets/img/20211222/CenterNetF2.png)

其逻辑很简单，选择Corner点之后可以确定其中心区域，如果中心与区中存在Center keypoint并且和Corner point的类别一致，则说明是正确的目标框。中心区域的划分如下所示

![](/assets/img/20211222/CenterNetF3.png)

但是这个区域的比例会影响检测的结果。更大比例的中心区域导致大目标更低的目标框精度，而更小比例的中心区域导致小目标更低的召回率。
中心坐标的确定方法如下所示
$$
\begin{equation}
\left\{\begin{aligned}
\operatorname{ctl}_{\mathrm{x}} &=\frac{(n+1) \mathrm{tl}_{\mathrm{x}}+(n-1) \mathrm{br}_{\mathrm{x}}}{2 n} \\
\mathrm{ctl}_{\mathrm{y}} &=\frac{(n+1) \mathrm{tl}_{\mathrm{y}}+(n-1) \mathrm{br}_{\mathrm{y}}}{2 n} \\
\mathrm{cbr}_{\mathrm{x}} &=\frac{(n-1) \mathrm{t} l_{\mathrm{x}}+(n+1) \mathrm{br}_{\mathrm{x}}}{2 n} \\
\mathrm{cbr}_{\mathrm{y}} &=\frac{(n-1) \mathrm{tl}_{\mathrm{y}}+(n+1) \mathrm{br}_{\mathrm{y}}}{2 n}
\end{aligned}\right.
\end{equation}
$$

其中n代表划分网格的数量。这里对于小目标使用小的$n=3$，而大目标使用大的$n=5$。

Certer Pooling和Corner Pooling基本一致，但是由于Center的左右上下都有目标信息，因此这里直接在垂直和水平上取MAX即可。Cascade即把Corner pooling接在Center pooling之后，这样Corner pooling可以间接的获取到目标中心的信息，如下图所示
![](/assets/img/20211222/CenterNetF4.png)
![](/assets/img/20211222/CenterNetF5.png)