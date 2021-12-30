---
layout: post
title: 'End-to-End Semi-Supervised Object Detection with Soft Teacher'
date: 2021-11-25
author: Poley
cover: '/assets/img/20211122/SoftTeacher.png'
tags: 论文阅读
---

本文提出一种端到端的半监督目标检测方法，通过逐步的提高伪标签的质量来使目标检测器的训练从中受益。

![](/assets/img/20211122/SoftTeacherF1.png)

无监督方法上，目前伪标签是一个主流方法。目前大多数方法都通过一个两阶段的方法来实现，即现在有标签数据上训练，再在无标签数据上生成伪标签，再重新训练。但是这种方法的性能受限于网络产生伪标签的质量，因为一开始在小规模有标注数据上的网络性能并不一定很好。

本文提出的方法可以进行端到端的训练，同时进行伪标签标注和训练，并且使用伪标签和少数标注数据进行训练。两个模型同时被应用在图片上，一个用于训练，一个用于标注。两个模型通过EMA来传递参数。

一个比较创新的地方在于本方法不是使用一个模型来产生Hard label 的box，而是由teacher直接access到student产生的每一个box candidates，并直接为其做assessment。对于每个cadidates，使用一个很高的阈值来区分其是前景还是背景，来得到尽可能准确的目标框，但是这样会使得一些positive box被遗漏。本文提出使用一个reliability measure来加权每个被判断为背景的Box candidate的loss。这相比施加硬性伪标签的效果更好，这里起名为"soft teacher"。在分类的训练上，teacher的目标检测得分就可以作为不错的reliability measure，而在回归上，本文通过一种box jittering approach，对于一个pseudo-foreground box candidate，将其jitter（抖动）若干次，并且用teacher进行回归，计算回归目标框的方差来作为reliability measure。

## Methodology
![](/assets/img/20211122/SoftTeacherF2.png)

### End-to-End Pseudo-Labeling Framework

在训练中，有标签数据和无标签数据一起训练，产生的loss按一定权重叠加，如下
$$
\begin{equation}
\mathcal{L}=\mathcal{L}_{s}+\alpha \mathcal{L}_{u}
\end{equation}
$$

其中
$$
\begin{equation}
\mathcal{L}_{s}=\frac{1}{N_{l}} \sum_{i=1}^{N_{l}}\left(\mathcal{L}_{\mathrm{cls}}\left(I_{l}^{i}\right)+\mathcal{L}_{\mathrm{reg}}\left(I_{l}^{i}\right)\right)
\end{equation}
$$

$$
\begin{equation}
\mathcal{L}_{u}=\frac{1}{N_{u}} \sum_{i=1}^{N_{u}}\left(\mathcal{L}_{\mathrm{cls}}\left(I_{u}^{i}\right)+\mathcal{L}_{\mathrm{reg}}\left(I_{u}^{i}\right)\right)
\end{equation}
$$

在训练的开始阶段，teacher和student都使用随机初始化。

### Soft Teacher
检测器的性能极大的依赖于伪标签的质量。作者发现使用很高的前景阈值来过滤掉大部分student-generated box candidates可以比阈值更好的效果，但是这样会产生很低的recall。在这种情况下，此时如果我们直接使用candidates和伪标签的IOU来判断，即需要很高的IOU阈值来为candidates分配标签，则会使得很多前景candidate被错误的assigned as negatives。

为了缓解这个问题，通过计算reliability来判断每个box candidate是一个真背景的可能性，分类损失如下
$$
\begin{equation}
\mathcal{L}_{u}^{\mathrm{cls}}=\frac{1}{N_{b}^{\mathrm{fg}}} \sum_{i=1}^{N_{b}^{\mathrm{fg}}} l_{\mathrm{cls}}\left(b_{i}^{\mathrm{fg}}, \mathcal{G}_{\mathrm{cls}}\right)+\sum_{j=1}^{N_{b}^{\mathrm{bg}}} w_{j} l_{\mathrm{cls}}\left(b_{j}^{\mathrm{bg}}, \mathcal{G}_{\mathrm{cls}}\right)
\end{equation}
$$
其中权重由reliability决定
$$
\begin{equation}
w_{j}=\frac{r_{j}}{\sum_{k=1}^{N_{b}^{\mathrm{bg}}} r_{k}}
\end{equation}
$$
这里的reliability通过teacher的bg分类得分得到，即直接将student box candidate输入teacher的head中，得到一个分类得分作为置信度。

本文还尝试了其他的分类reliability方案，比如student的bg得分，teacher和student的Prediction difference，或者IoU等等。

### Box Jittering
![](/assets/img/20211122/SoftTeacherF3.png)
如上图所示，可以看出目标框的回归质量和目标框的前景得分并没有非常明显的正相关关系。这意味着一个前景得分很高的目标框并不一定可以提供准确的定位信息。这意味着并不能像上述那样选择分类目标一样选择回归目标。

这里使用jitter的方法来估计伪标签的质量。即对于teacher产生的每个伪标签$b_i$,对其进行jittering，在其周围产生一个jittered box，并经过refine网络得到refine box $\hat{b}_i$，收集$N_{jitter}$个并计算他们的方差作为localization reliability
$$
\begin{equation}
\bar{\sigma}_{i}=\frac{1}{4} \sum_{k=1}^{4} \hat{\sigma}_{k}
\end{equation}
$$
$$
\begin{equation}
\hat{\sigma_{k}}=\frac{\sigma_{k}}{0.5\left(h\left(b_{i}\right)+w\left(b_{i}\right)\right)}
\end{equation}
$$
更小的回归方差意味着更高的可信度。通过这个指标，训练中可以筛选回归方差低于某个阈值的box candidate来训练student的box regression branch，loss如下
$$
\begin{equation}
\mathcal{L}_{u}^{\mathrm{reg}}=\frac{1}{N_{b}^{\mathrm{fg}}} \sum_{i=1}^{N_{b}^{\mathrm{fg}}} l_{\mathrm{reg}}\left(b_{i}^{\mathrm{fg}}, \mathcal{G}_{\mathrm{reg}}\right)
\end{equation}
$$

## Experiments
![](/assets/img/20211122/SoftTeacherT5.png)