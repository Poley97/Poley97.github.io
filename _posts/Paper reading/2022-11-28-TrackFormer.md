---
layout: post
title: 'TrackFormer: Multi-Object Tracking with Transformers'
date: 2022-10-27
author: Poley
cover: '/assets/img/20221123/Trackformer.jpg'
tags: 论文阅读
---

本文发表于CVPR2022，提出了一种全新的图像追踪范式，即tracking-by-attention。
![](/assets/img/20221123/TrackformerF1.jpg)
目前tracking主要分为三种类型。
+ tracking-by-detection，即每一帧进行单独检测，之后再对不同帧之间的检测框进行单独的关联操作。tracking by detection主要需要解决的是data关联的问题。又可以细分为三种方法：通过图解决，一般是寻找图的最小cost优化，基于距离/学习。缺点，不好优化，速度慢；Appearance-based则通过物体之间的相似度来进行匹配。容易受到密集场景物体间遮挡的问题；Motion-based进行轨迹预测来进行追中。但是3中的非线性运动在2D中的轨迹仍然具有挑战性。
+ tracking-by-regression，取代了数据关联操作，在每个轨迹上使用连续的回归来追踪物体改变的位置。可以达到top性能，但是需要额外的图优化或运动行为预测。通过连续回归进行追中，缺乏物体identity的信息，需要借助额外的re-id和运动预测模型。
+ 作者提出，track-by-attention。无需使用数据关联，只需要联合进行tracking和detection。实际上这种方法通过atention(本文中通过query）隐式的完成了data association,使得整个网络更加简洁并且可以使用end-to-end的形式进行训练。

本文的整体结构如下图所示
![](/assets/img/20221123/TrackformerF2.jpg)

主要分为两个部分，一个是static query，即detr head自带的那部分query。另一部分是track queries，即从上一帧继承过来的有对应目标的query，这样直接解决了目标的id关联的问题。每个物体被显式的表达为一个decoder queries。这里通过query实现frame之间的关联。前一帧的output embedding被用于为一下帧的检测query做初始化。query之间的Self-attention也很自然的防止新的query检测到已有query的对应的物体，非常巧妙。

对于前一帧的检测结果，根据query预测的置信度筛选valid object(query)并传递给下一帧；而被识别为背景的query则被抛弃。考虑到检测中还可能出现短暂的遮挡导致目标丢失的情况，因此一般情况下re-id在tracking中也是必须的。为了应对这种情况，这里提出类似于Memory，对于移除的queries，保留一定的时间（frame）。注意到，这里仍然需要使用NMS。用于处理一些Self-attention不能解决的重复框。（个人认为可能是由于不统一的初始化造成的。）memory住的query，当其的预测结果大于阈值是，就会触发re-id，继续这个物体的轨迹。由于query中包含的位置信息，长时间远距离的移动，query无法索引到。但是短时间的recovery却是query非常擅长的。因此，这样非常适合常见的short-term occlusions。

在训练上，这里的训练方法是，一次输入两张图像，Object检测loss和track loss放在一起训练。对于tracking的identities和groundtruth的匹配。这里分别对匹配、遮挡和新物体（未被遮挡）设计了三种不同的匹配规则。如下，

$K_t \cap K_{t-1}$ : Match by track identity $k$.
$K_{t-1} \backslash K_t$ : Match with background class.
$K_t \backslash K_{t-1}$ : Match by minimum cost mapping.
即针对这三种情况，分别：继承target，分配background以及通过匈牙利匹配重新分配target。

比较关键的，作者针对这套pipeline专门设计了三种数据增广方法，来对物体的位置、运动、missing和遮挡进行干扰。
+ 随机使用前某一帧（一段时间Range内）和当前帧匹配。以模拟大距离的一定和低帧数数据。
+ 在传递的时候，以一定比例传递FN（漏检）到下一帧中。保持足够的FN（漏检）比例是联合训练有效的关键。
+ 同理，在传递的过程中，加入FP，即一些背景query，这些在下一帧的训练中应当被识别为背景，对应为被遮挡的情况。相当于也保证了FP的数量。

其余是一些常见的，比如对目标框空间扰动，等等。由于query对信息的编码是隐式的，因此不能对query扰动。

在性能上，是达到了当时SOTA水平。
![](/assets/img/20221123/TrackformerT1.jpg)

从消融实验来看，其所提出的增广方法对于训练具有非常显著的效果。
![](/assets/img/20221123/TrackformerT4.jpg)

