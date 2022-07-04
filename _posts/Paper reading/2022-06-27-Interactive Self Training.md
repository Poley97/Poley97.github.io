---
layout: post
title: 'Interactive Self-Training with Mean Teachers for Semi-supervised Object Detection'
date: 2022-06-27
author: Poley
cover: '/assets/img/20220627/InteractiveSelfTraining.png'
tags: 论文阅读
---

本文发表在CVPR2021上，其中intractive体现来两个地方：
+ 多iter结果融合，促进结果的稳定性
+ 多head预测，为彼此提供互补信息(需要避免多head collapsing)。

本方法同样基于伪标签方法，其最终目的就是提高伪标签的质量。作者提出，在半监督训练中，网络输出的结果（伪标签）并不稳定，这使得训练不容易收敛，导致性能的下降，如下图所示。

![](/assets/img/20220627/InteractiveSelfTrainingF1.png)

作者提出的解决方法是通过emsemble来提升伪标签的质量。使用了memory，通过更新memory bank来更新无标注数据的伪标签。如下图所示

![](/assets/img/20220627/InteractiveSelfTrainingF2.png)

在分类问题中，memory一般是去单独追踪每一个图像的embedding feature，但是对于图像检测来说，目标数量和顺序的不确定给追踪记录feature带来了很大的困难，取而代之的是记录检测结果。因此，对应的融合方法也从EMA变成了NMS。

同时，本文也提出使用两个RoI Head来提供互补的信息（但是没说为什么这么做，以及互补的信息有什么用？但是性能确实有所提升）。为了保证互补性，需要避免两个RoI head在训练中发生collapes进而拥有相同的权重。这里引入了两个方法来保证这一点：
1、使用不同的参数进行初始化
2、使用Mean-teacher + DropBlock. DropBlock对featuremap中的区域进行随机遮挡，使得两个student RoI head关注画面的不同部分，进而不会趋向于相同的参数。Teacher使用对应Student的参数进行EMA更新，也不会趋同。

这里的双RoI head只是为了产生伪标签，并不是最终训练的模型。其最终目的是使用互补信息产生更高质量的伪标签，对最终的任意检测模型进行基于伪标签的半监督训练。