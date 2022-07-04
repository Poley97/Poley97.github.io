---
layout: post
title: 'Data-Uncertainty Guided Multi-Phase Learning for
Semi-Supervised Object Detection'
date: 2022-06-29
author: Poley
cover: '/assets/img/20220627/DataUncertaintyGuided.png'
tags: 论文阅读
---
这篇论文发表于CVPR2021。论文的整体思路很清晰，总而言之，是为了提高伪标签质量以及降低错误标签对训练的影响。半监督的目的在于突破全监督训练的上限，但是很容易受到噪声的影响。这里综合考虑无标注数据的divergent types，根据其难度；最终使用不确定度实现数据选择和RoI的reweight，利于网络学习知识：
1.针对噪声标签导致的模型对easy性能下降，采用分段训练的方法，先学easy，再学hard，在保证模型对easy拟合的情况下逐渐增加样本难度，类似于adaboost，最后综合多模型的输出。
2.针对漏检带来的噪声，对RoI重加权，降低漏检带来的负面影响（模型认为是正样本的位置标签是负样本，带来很大的损失，对模型的梯度产生显著影响。）

论文方法的整体结构如下
![](/assets/img/20220627/DataUncertaintyGuidedF1.png)

本文首先对与半监督训练中伪标签质量对于训练的影响进行了分析。这里将最简单的伪标签方法（使用预训练模型）称为one-phase方法。作者提出，伪标签中带来的噪声也会被拟合，并且拟合能力更强，超过正确样本，称为label noise overfitting。

半监督学习中，随着模型泛化性的增强，其在无标签数据上的绝对性能实际上是在下降的。在多phase训练中，由于phases 1时模型在无标注数据上的性能已经下降，因此影响到phase 2时伪标签质量，导致模型的最终训练效果也出现了下降。如下图所示，模型使用VOC07作为有监督数据，VOC12作为无监督数据，进行训练的结果如下。
![](/assets/img/20220627/DataUncertaintyGuidedF2.png)

可以看到，经过Phases0的训练，模型的泛化性得到了提升，但是在无监督训练集上的性能却下降了（这就是产生伪标签的性能），今儿，在phase 2上，测试集性能也出现了下降。作者认为这是由于phase 1中模型产生伪标签质量的下降引起的。其原理即，噪声会向模型提供不稳定的监督（噪声伪标签），会格外吸引模型优化和拟合的注意力，称为 label noise overfitting。从上图（b）中也可以看出，实际上伪标签增强了强化了模型对于hard样本的能力，而使得对easy的性能发生了退化，这是因为模型在训练过程中会更加关注hard样本（其贡献更大的Loss）。引入伪标签之后，easy的性能下降，hard的提升，说明模型对于hard sample过度关注了（其中也包括噪声）（对准确率），同时，伪标签的错分（主要是漏检）导致网络被错误的监督，产生不确定的输出，并贡献很大的loss，因此对训练起到dominant（对应查全率）。

## Multi-Phase Learning 
作者提出的解决方法是 Multi Phase Learning，因为单phase的SSOD解决不了上述问题，因为总会存在伪标签噪声，其一定会获得模型更多的关注。因此，使用多个模型将easy和difficult分开训练。笔者认为，这很类似于Adaboost，对于多模型输出这里最后使用WBF进行融合。

### Uncertainty Guided Training
上述思路得以实现的关键是如何区分样本的难度，这里作者使用图片里的正样本平均置信度来决定当前样本的难度（这实际上是从准确率的角度考虑，因为其并不将漏检考虑在内）。平均置信度越高，说明伪标签的准确性越高，样本越简单，反之亦然。
$$
\begin{equation}
\bar{s}_{m}=\sum_{m=1}^{M} s_{m n} / M
\end{equation}
$$

### Region Uncertainty Guided RoI Re-weighting
另一个需要考虑的问题即查全率，漏检会导致模型将潜在的正样本划分为负样本，产生很大的loss，并且不利于网络的性能。但是完全的不漏检又会大大降低伪标签的质量，也是不可取的。因此这里提出了re-weighting方法，尽量避免对伪标签中可能漏检的目标产生损失，进而忽略他们对于训练的影响。

这里的基本思路是，一般来说，没有与任何positive有交集的RoI更有可能是miss-annotated。故根据一个目标框与其他positive 目标框的最大iou来给定其权重如下
$$
\begin{equation}
w_{i}=a+(1-a) e^{-b e^{-c \cdot \mathrm{IoU}_{\mathrm{i}}}}
\end{equation}
$$

同时，作者还引入了特征相似度来判断一个roi与positive roi的相关性，如果两者特征相关性高，那么其更有可能是一个漏检标签。相关性计算使用余弦相关度，如下所示

$$
\begin{equation}
d_{i j}=\frac{\left|f_{i}^{T} f_{j}\right|}{\left\|f_{i}\right\|\left\|f_{j}\right\|}
\end{equation}
$$

2D中若一个框大目标框中的情况，这种情况下，目标框的特征会和大目标相似，但是其并不是一个postive，因此，这里作者还引入了Intersection over foreground (IoF)对这种情况进行限制。结合两者，得到
$$
\begin{equation}
D_{i}=1-\max _{j} d_{i j}\left(1-\mathrm{IoF}_{\mathrm{ij}}\right)
\end{equation}
$$

与上述iou-based的权重结合，得到
$$
\begin{equation}
w_{i}=\left(a+(1-a) e^{-b e^{-c_{1} \cdot I o U_{i}}}\right) e^{-b e^{-c_{2} D_{i}}}
\end{equation}
$$

这里使用的是一个Gomper函数，其特点是两边平缓，中间非常陡峭。当IoU和D接近0时，权重接近于0。即尽量不让miss annotated对网络的训练产生影响，而是将其的梯度忽略。最后由于特征本身也是基于含噪声学习得到的，因此其距离判断可能并不可靠。这里根据RoI的位置关心引入额外的全连接成用于对特征的相似度进行监督。
$$
\begin{equation}
y_{i j}=I\left(\mathrm{IoF}_{\mathrm{ij}}>t\right) \text { or } I\left(\mathrm{IoF}_{\mathrm{ji}}>t\right)
\end{equation}
$$
$$
\begin{equation}
L_{s i m}=y_{i j}\left(1-d_{i j}\right)^{2}+\left(1-y_{i j}\right) d_{i j}^{2}
\end{equation}
$$

整体的结构如下所示

![](/assets/img/20220627/DataUncertaintyGuidedF4.png)

而这一步中的RoI re-weighting也确实体现出了不错的效果，可以有效的应对Miss annotated情况，如下

![](/assets/img/20220627/DataUncertaintyGuidedF5.png)
