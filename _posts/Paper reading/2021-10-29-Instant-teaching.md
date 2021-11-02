---
layout: post
title: 'Instant-Teaching: An End-to-End Semi-Supervised Object Detection Framework'
date: 2021-10-29
author: Poley
cover: '/assets/img/20211027/InstantTeacher.png'
tags: 论文阅读
---

>参考博客：https://zhuanlan.zhihu.com/p/359154264

提出一种 Instant Teaching 的半监督物体检测框架，利用 pseudo labeling和拓展的waek-strong data augmentations 来在训练过程中进行teaching。为了缓解 Confirmation Bias（确认偏误）和伪标签的质量问题，进一步提出 co-rectify scheme。


目前SSOD领域最好的方法是STAC。STAC方法首先需要训练一个teacher model，然后使用期去产生无标签数据的伪标签。之后在模型训练的过程中，伪标签不会再更新。这带来两个问题，在Instant-Teaching中都被解决。首先是伪标签更新问题，在每个训练迭代内，都会先对经过weak data augmentation 的未标注数据进行伪标签的生成。其次，对于同一张图，会在经过strong data augmentation之后用于student的训练。这样保证了伪标签的质量会随着模型质量的提升而提升。另一个问题是SSL中常见的confirmation bias问题，也就是确认偏置。确认偏置这个概念来自于心理学，指人当认定了正确的事情的时候，就会倾向于寻找可以支持自己观点的证据而非可以反驳自己观点的证据，也就是自己发现和纠正自己的错误。模型的确认偏执同理，因此本文提出一个co-rectify scheme（共同矫正）来使student和teacher相互矫正自己的错误预测。

## Method 

### Problem definition

In semi-supervised object detection (SSOD), we are given a set of labeled data $\mathcal{D}_{l}=\left\{\left(\mathbf{x}_{i}^{l}, y_{i}^{l}\right)\right\}_{i=1}^{n_{l}}$ and a set of unlabeled data $\mathcal{D}_{u}=\left\{\mathbf{x}_{j}^{u}\right\}_{j=1}^{n_{u}}$, where $\mathrm{x}$ and $y$ denote image and ground-truth annotations (class labels and bounding box coordinates) respectively. The goal of $S S O D$ is to train object detectors on both labeled and unlabeled data.

整体结构如下所示
![](/assets/img/20211027/InstantTeacherF1.png)

### Instant pseudo labeling
和一般的伪标签方法一样，先产生伪标签，再做监督。这里的弱增强只使用random flip。损失由监督损失和无监督损失组成，如下

$$
\begin{equation}
\ell=\ell_{s}+\lambda_{u} \ell_{u}
\end{equation}
$$
$$
\begin{equation}
\begin{aligned}
\ell_{s}=& \sum_{l}\left[\frac{1}{N_{c l s}} \sum_{i} L_{c l s}\left(p\left(c_{i} \mid \alpha\left(\mathbf{x}_{l}\right)\right), c_{i}^{*}\right)\right.\\
&\left.+\frac{\lambda}{N_{r e g}} \sum_{i} c_{i}^{*} L_{r e g}\left(p\left(\mathbf{t}_{i} \mid \alpha\left(\mathbf{x}_{l}\right)\right), \mathbf{t}_{i}^{*}\right)\right]
\end{aligned}
\end{equation}
$$

$$
\begin{equation}
\begin{aligned}
\ell_{u}=& \sum_{u}\left[\frac{1}{N_{c l s}} \sum_{i} L_{c l s}\left(p\left(c_{i} \mid A\left(\mathbf{x}_{u}\right)\right), \hat{c}_{i}^{u}\right)\right.\\
&\left.+\frac{\lambda}{N_{r e g}} \sum_{i}\left(\max \left(c_{i}^{u}\right) \geq \tau\right) L_{r e g}\left(p\left(\mathbf{t}_{i} \mid A\left(\mathbf{x}_{u}\right)\right), \mathbf{t}_{i}^{u}\right)\right]
\end{aligned}
\end{equation}
$$

### Weak-strong data augmentations

Weak-strong data augmentations 是一个Promising practice，已经被证明在半监督任务中的作用。这样的数据增强强制模型保持两者之间的一致性，鼓励模型从伪标签中学习到有用的信息。

而本方法的关键就在于weak和strong之间的区别。这里将STAC的strong进一步增强，加入了Mixup和Mosaic。

这里使用一个有标签图像以及对应的标签和一个无标签数据以及对应的伪标签进行增强。以一定比例加权两者的图像信息，对两者的标签置信度进行比例调整。比例从Beta分布中随机产生，如下所示。

$$
\begin{equation}
\left\{\begin{aligned}
\lambda_{m} & \sim \operatorname{Beta}\left(\alpha_{m}, \alpha_{m}\right) \\
\mathbf{x}_{u} &=\lambda_{m} \mathbf{x}_{u}+\left(1-\lambda_{m}\right) \mathbf{x}_{l} \\
c_{u} &=\lambda_{m} c_{u} \cup\left(1-\lambda_{m}\right) c_{l} \\
b_{u} &=b_{u} \cup b_{l}
\end{aligned}\right.
\end{equation}
$$

![](/assets/img/20211027/InstantTeacherF2.png)

### Co-rectify

这是半监督学习中的常见问题。当模型产生一个高置信度的错误预测时，所产生的伪标签会进一步强化模型对于这种错误的认知。Co-rectify的可以成功的原因是因为两个模型不会收敛成同一个模型，这就需要保证两个模型各自独立的收敛。作者使用了两种方法来保证这一点：(1)使用不同的初始参数；(2) 使用不同的数据增强和伪标签。

其方法如图1所示，首先通过一个模型产生伪标签预测结果，并以这个结果当作proposal送到另一个模型的head中进行refine，得到新的伪标签预测结果。之后平均两者的分类置信度，并根据置信度对两者的预测矿进行加权（个人感觉类似于wbf的加权方式），如下式所示

$$
\begin{equation}
\left\{\begin{aligned}
\left(c_{i}, \mathbf{t}_{i}\right) &=f_{a}\left(\mathbf{x}_{u}\right) \\
\left(c_{i}^{r}, \mathbf{t}_{i}^{r}\right) &=f_{b}\left(\mathbf{x}_{u} ; \mathbf{t}_{i}\right) \\
c_{i}^{*} &=\frac{1}{2}\left(c_{i}+c_{i}^{r}\right) \\
\mathbf{t}_{i}^{*} &=\frac{1}{c_{i}+c_{i}^{r}}\left(\mathbf{t}_{i} c_{i}+\mathbf{t}_{i}^{r} c_{i}^{r}\right)
\end{aligned}\right.
\end{equation}
$$

## Experiments

![](/assets/img/20211027/InstantTeacherT1.png)