---
layout: post
title: 'GROUP DETR: FAST DETR TRAINING WITH GROUPWISE ONE-TO-MANY ASSIGNMENT'
date: 2022-11-29
author: Poley
cover: '/assets/img/20221123/GroupDETR.jpg'
tags: 论文阅读
---

目前对检测器的训练主要分为两种target assign方式，分别是传统方法常用的One-to-many方法和DETR-based方法常用的One-to-One方法。其中，One-to-One的assign方法是DETR-based方法可以实现端到端物体检测的关键所在。但是，其也存在缺点，即收敛性非常差。DETR在COCO上需要500Epoch收敛，及时现在通过若干方法改善了其收敛性，其一般也需要50Epoch才能收敛。相比之下，一般检测器典型的收敛时间是12epoch（即1x schedule）。而DETR-based在1x的训练达不到可以接受的结果。

作者通过实验对目前几种DETR-based方法的收敛性进行了探究，其中影响收敛性的因素有：query数量、target assign方法和训练时长，结果如下图所示。

![](/assets/img/20221123/GroupDETRF1.jpg)

从上图可以看到，总query数量和匹配到正样本的query数量越多，网络收敛越快。One-to-Many可以使得更多的query匹配到gt样本，但是这样也破坏了网络的端到端特性，尽管在NMS的加持下，这样的网络收敛的更快。

因此，作者提出group-wise的One-to-Many分配策略，即获得One-to-many的收敛性，同时又保持One-to-One的端到端特性。结构如下图所示

![](/assets/img/20221123/GroupDETRF2.jpg)


简单来说，这里使用K组query，他们共享网络参数但又互不可见。在训练时，所有的gt对每个group内进行one-to-one的assgin，因此，在总体上，其是一个one-to-K的assign。这样有效增加了网络训练时候的正样本query数量。在infer时，只保留一组query，从而保留end-to-end特性。注意到，这里不同group之间在self-attention中是互不可见的，这可以通过mask实现，类似于DN-DETR。

作者进一步对这个进行了分析，如下图所示

![](/assets/img/20221123/GroupDETRF3F4.jpg)

这里定义了两个度量，分别是perturbation distance PD

$$
\begin{equation}
\mathbf{P D}=\frac{1}{2 N} \sum_{i=1}^N\left(\| \mathcal{P}\left(\mathbf{q}_i^1\right), \mathcal{P}\left(\mathcal{N}_{2 \rightarrow 1}\left(\mathbf{q}_i^1\right)\|+\| \mathcal{P}\left(\mathbf{q}_i^2\right), \mathcal{P}\left(\mathcal{N}_{1 \rightarrow 2}\left(\mathbf{q}_i^2\right) \|\right)\right.\right.
\end{equation}
$$

和 matching distance MD

$$
\begin{equation}
\mathbf{M D}=\frac{1}{M} \sum_{i=1}^M \|\left(\mathcal{P}\left(\mathbf{q}_{\sigma_1(i)}^1\right), \mathcal{P}\left(\mathbf{q}_{\sigma_2(i)}^2\right) \|\right.
\end{equation}
$$

MD表示两个group之间，匹配到相同gt的query的平均距离。可以看MD是很大的，这说明本方法有效为一个gt分配到了多个不太相同的query，从而加快了训练速度。

从实验性能上来看，在50e的训练上，本方法具有显著优势
![](/assets/img/20221123/GroupDETRT1.jpg)

并且这个训练策略适用于多种DETR-Based模型，并且在12e的收敛性上都具有显著的提升。
![](/assets/img/20221123/GroupDETRT2.jpg)