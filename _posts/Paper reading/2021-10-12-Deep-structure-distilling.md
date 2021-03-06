---
layout: post
title: 'Deep Structured Instance Graph for Distilling Object Detectors'
date: 2021-10-12
author: Poley
cover: '/assets/img/20211012/DSIG.png'
tags: 论文阅读
---

设计了一个图，其中的节点对应物体实例的instance proposal-level features ，边对应节点之间的关系。通过定义一个自适应的背景损失权重来减小节点噪声，以及剪枝。之后将整个图作为encoded knowledge representation ， 从teacher向 student传递。

知识蒸馏中一直存在的问题是特征不平衡，如果只用结果做监督，则只有softmax层的结果会被学生考虑。另一种方法是全面转移完整的global feature map，但是这样会带来太多不必要的需要匹配的像素。还有方法使用手工的mask来提取靠近实例的特征，但是这样就只会有非常少的背景特征被蒸馏，同样造成有用信息的损失。一个关键问题就出现了：**如何利用背景特征并且达到一个平衡？**

作者同样认为，基于像素匹配的蒸馏学习忽略了实例层面的关系，这使得对于同一个test image，teacher和student各自实例见的分布发生变化，如下图所示。因此，另一个问题出现了：**如何利用深度神经网络之间的隐关系？**

![](/assets/img/20211012/DSIGF1.png)

综上，本文通过建立实例之间的图来完成知识传递和蒸馏。

![](/assets/img/20211012/DSIGF2.png)

# 建图过程

+ 为了保证两者特征的对应关系，很明显两者应该使用一样的proposal，也就是共享一个rpn。
+ 节点：每个Proposal通过IoU来判断其实是正样本还是负样本。但是这里并不直接抛弃负样本，而是使用一个自适应背景损失。
+ 边：两个节点的边通过一个相似度函数来确定
$$
\begin{equation}
e_{p q}:=\operatorname{sim_{f}function}\left(v_{p}, v_{q}\right)
\end{equation}
$$
$$
\begin{equation}
s\left(v_{p}, v_{q}\right)=\frac{v_{p} \cdot v_{q}}{\left\|v_{p}\right\| \cdot\left\|v_{q}\right\|}
\end{equation}
$$
+ 背景样本挖掘：大量的背景节点会带来很高的损失，在背景相关的边上。因此进行剪枝，只选择一部分高质量的背景样本（带来的损失高于阈值，而非高置信度背景）。如果有正样本$n$个，对应的邻接矩阵为$n\times n$，此方法将样本数扩充为$\hat{n} \times \hat{n}$。其中的$\hat{n}-n$个样本就是额外扩充的背景样本。

+ 图蒸馏损失,其中$\lambda_2$用于缓解不平衡问题。
$$
\begin{equation}
\begin{aligned}
L_{\mathcal{G}}=& \lambda_{1} \cdot L_{\mathcal{V}}^{f g}+\lambda_{2} \cdot L_{\mathcal{V}}^{b g}+\lambda_{3} \cdot L_{\mathcal{E}} \\
=& \frac{\lambda_{1}}{N_{f g}} \sum_{i=1}^{N_{f g}}\left\|v_{i}^{t, f g}-v_{i}^{s, f g}\right\|^{2}+\frac{\lambda_{2}}{N_{b g}} \sum_{i=1}^{N_{b g}}\left\|v_{i}^{t, b g}-v_{i}^{s, b g}\right\|^{2} \\
&+\frac{\lambda_{3}}{N^{2}} \sum_{i=1}^{N} \sum_{j=1}^{N}\left\|e_{i j}^{t}-e_{i j}^{s}\right\|^{2}
\end{aligned}
\end{equation}
$$
$$
\begin{equation}
\lambda_{2}=\alpha \cdot \frac{N_{f g}}{N_{b g}}
\end{equation}
$$

+ 总损失： 图像分类中的知识迁移常用KL散度损失，这里的目标框分类和回归也是类似的问题，因此也使用了KL散度损失，总损失如下。

$$
\begin{equation}
\begin{aligned}
L=& L_{\text {Det }}+L_{\mathcal{G}}+L_{\text {Logits }} \\
=& L_{R P N}+L_{\text {RoIcls }}+L_{\text {RoIreg }} \\
&+L_{\mathcal{G}}+L_{\text {Logits }}
\end{aligned}
\end{equation}
$$
其中的$L_{logits}$代表最终分类和回归的KL散度损失。

# Experiment

![](/assets/img/20211012/DSIGF3.png)

![](/assets/img/20211012/DSIGT1.png)