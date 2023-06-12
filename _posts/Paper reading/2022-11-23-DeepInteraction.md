---
layout: post
title: 'DeepInteraction: 3D Object Detection via Modality Interaction'
date: 2022-10-27
author: Poley
cover: '/assets/img/20221123/DeepInteraction.jpg'
tags: 论文阅读
---

本文发表在NeurIPS2022上。本文的设计思想在于，充分保留单模态信息，并学习利用他们独特的特性来进行目标检测。本文中使用模态交互取代了模态融合来实现多模态感知。即学习和保留两个modality-specific representations并使用跨模态的交互来实现模态间的信息交互，同时保留各模态的长处。如下图所示

![](/assets/img/20221123/DeepInteractionF1.jpg)

作者认为统一的BEV表征虽然很适合融合多模态相机特征，但是不利于表征点云中的空间信息（由于视角转换），从而潜在的降低了camera BEV表征的质量。因此，本文中并不适用统一的表征方式。
这里提出两种结构，分别是特征性交互的encoder和预测性交互的decoder。两个主干分别提取特征并mapping到BEV上。之后是encoder-decoder结构。encoder通过双向交互来逐步交换两个模态的信息。decoder则实现多模态预测，使用一个多级结构。


首先是representational interaction 的 encoder结构。
![](/assets/img/20221123/DeepInteractionF2.jpg)

总体思路比较简单，两个模态中，分别以一个模态为主（作为query），分别去索引其他模态的对应位置的邻域信息(作为key 和 value)。

Encoder是一个MIMO结构，多模态输入、多模态输出。主要包括，多模态表征交互，跨模态表征学习，表征融合。进一步的，其分为两个部分，分别是
+ Multi-modal representational interaction (MMRI)：如上图所示，即一个模态从另一个模态中，query相关的信息，来丰富本模态的特征。这里的query并不是全局的。

对于img2lidar的特征而是通过坐标投影，找到其在另一个模态中的邻域特征。这需要显式的深度补全，通过深度补全建立从img到bev的mapping。因此图像点在bev上的邻域特征索引也通过其图像邻域的索引映射得到。

对于pillar2img的特征，pillar2img的映射（包含邻域），通过pillar内的点全部投影到Img上获得。（整个投影过程给人的感觉是简单粗暴）

通过上述的双向映射，将跨模态的邻域设置为k,v用于自适应的聚合，丰富本模态的特征。

同样使用intra-modal的local attention来丰富本模态内的信息。即在本模态中，使用和上述相同的邻域去聚合特征。

最后，聚合跨模态和模态内学习到的特征。作为模态的输出。如下所示

$$
\begin{equation}
\begin{aligned}
\boldsymbol{h}_p^{\prime} &=\operatorname{FFN}\left(\text { Concat }\left(\operatorname{FFN}\left(\text { Concat }\left(\boldsymbol{h}_p^{p \rightarrow p}, \boldsymbol{h}_p^{c \rightarrow p}\right)\right), \boldsymbol{h}_p\right)\right) \\
\boldsymbol{h}_c^{\prime} &=\operatorname{FFN}\left(\text { Concat }\left(\operatorname{FFN}\left(\text { Concat }\left(\boldsymbol{h}_c^{c \rightarrow c}, \boldsymbol{h}_c^{p \rightarrow c}\right)\right), \boldsymbol{h}_c\right)\right)
\end{aligned}
\end{equation}
$$
即完成了encoder的工作。

![](/assets/img/20221123/DeepInteractionF3.jpg)

decoder上还是self-attention+cross-attention的形式。

只不过这里提取特征的方式还是基于proposal的roi feature。roi也是逐级更新的，相当于一个cascade结构。通过将3D框投影到2D获得最小的axis-aligned的目标框，并在对应模态上提取roi特征，注意到这里的query和roi feature特征的交互方式比较特别，这里将query映射为一组1*1卷积的参数并将其应用到对应的roi特征上。

最后，在实验结果上，其达到了Nuscenes上的SOTA结果。注意到，这里的query只需要200，这是因为这里初始化的boxes应该是从dense feature map预测得到。因此相当于已经对前景进行了筛选，不需要很多的query，和Transfusion相同。

![](/assets/img/20221123/DeepInteractionT1.jpg)


其融合效果如图所示，可以看到，通过融合visual clues（即图像信息），点云的Bev Map上对一些困难物体（远距离、小、相连物体）明显有了更强的响应。
![](/assets/img/20221123/DeepInteractionF5.jpg)