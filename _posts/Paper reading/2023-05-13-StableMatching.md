---
layout: post
title: 'Detection Transformer with Stable Matching'
date: 2023-5-13
author: Poley
cover: '/assets/img/20230513/StableMatching.jpg'
tags: 论文阅读  
---
这里主要提出一个路径不不统一的问题。即当一个IOU高CLS低和一个反之的预测框出现时。由于两者都计算COST，因此GT可能匹配到两者任何一个，因此导致模型在不断的分别优化这两个框（相互冲突）。本文则通过引入position-supervies loss来缓解这个问题，实现训练的stabilizing。如下图所示
![](/assets/img/20230513/StableMatchingF2.jpg)

前文提到的两种优化路径即1、优化最好的预测框的分数 2、优化得分最高的框到GT。传统检测方法的assgin都是通过positional accuracy only的，因此属于第一种方法，而DETR通过cost计算，因此两者兼有。到这里可能会产生一个疑问，就是既然cls而cls是one-to-one assign的关键，否则GT附近的pred可能会被轮流当做Positive从而使得Matching更不稳定。

为了维持优化路径的稳定，这里提出使用第一种优化路径。为什么不用第二种呢？作者提出，另一个优化路径是有损性能的。因为存在多个目标时，靠近任何一个目标的框都可能又高cls，因此这样在gt匹配的时候还是会有不稳定的情况。

因此，为了使得DETR的target assign遵守第一种路径，作者提出了如下改进（改进主要针对cls损失和cost）

一般DETR-based方法中常用的都是focal-loss如下所示
$$
\begin{equation}
\mathcal{L}_{c l s}=\sum_{i=1}^{N_{p o s}}\left|1-p_i\right|^\gamma \mathrm{BCE}\left(p_i, 1\right)+\sum_{i=1}^{N_{n e g}} p_i^\gamma \mathrm{BCE}\left(p_i, 0\right),
\end{equation}
$$

而对应的cls cost为
$$
\begin{equation}
\mathcal{C}_{c l s}(i, j)=\left|1-p_i\right|^\gamma \operatorname{BCE}\left(p_i, 1\right)-p_i^\gamma \operatorname{BCE}\left(1-p_i, 1\right)
\end{equation}
$$

这里优化的方法主要是将positional metric like IOU 指标引入cls，以区分上述两种优化路径，当assign到不期望的优化路径时（即将gt分配给cls分数高但是iou分数低的pred），可以产生较小的损失，如下所示

$$
\begin{equation}
\begin{aligned}
\mathcal{L}_{c l s}^{(\text {new })}= & \sum_{i=1}^{N_{\text {pos }}}\left(\left|f_1\left(s_i\right)-p_i\right|^\gamma \operatorname{BCE}\left(p_i, f_1\left(s_i\right)\right)\right. \\
& +\sum_{i=1}^{N_{\text {neg }}} p_i^\gamma \operatorname{BCE}\left(p_i, 0\right),
\end{aligned}
\end{equation}
$$

简单地说，使用一些软化的指标来代替原有cls分类中的1-0的gt，比如iou。即用iou做目标的分类得分。这里发现使用Iou的平方加上一个缩放是最好的，可以避免一些退化。其中缩放有两种方法，1、确保最高的si2等于所有可能的pairs里iou最高的值，2、确保其为1.0。作者对这两种方法都做了消融，其分辨对大query数和小query数具有更好的效果。

之后，则是针对cls cost的优化。目的：鼓励产生低分类得分，高iou的预测。和一般的cls loss 与 cost共享相同的函数不同，这里之所以不直接使用分类损失，是因为上述分类损失对于高iou高cls和低iou低cls都有很低的cost，而这不是匹配中希望见到的。这里匹配中实际上使用cls*iou作为得分来进行cost的计算。因此其cls可以计算如下
$$
\begin{equation}
\begin{aligned}
\mathcal{C}_{c l s}^{\text {(new) }}(i, j)= & \left|1-p_i f_2\left(s_i^{\prime}\right)\right|^\gamma \mathrm{BCE}\left(p_i f_2\left(s_i^{\prime}\right), 1\right) \\
& -\left(p_i f_2\left(s_i^{\prime}\right)\right)^\gamma \mathrm{BCE}\left(1-p_i f_2\left(s_i^{\prime}\right), 1\right)
\end{aligned}
\end{equation}
$$

综上，这里实际上是确定了唯一的优化路径，即使高IOU的预测框具有高score，而取缔了反之的优化路径。从对模型的训练统计可以看出，引入这种方法后，模型各层之间的target assign一致性更高。

![](/assets/img/20230513/StableMatchingF3.jpg)

本文由于优化的是Loss和cost，因此可以灵活嵌入现有的DETR方法中，实现性能的提升而不带来额外的损耗。个人认为具有一定的实用性。
![](/assets/img/20230513/StableMatchingT2.jpg)