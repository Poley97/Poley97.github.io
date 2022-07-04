---
layout: post
title: 'CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point
Cloud Understanding'
date: 2022-06-24
author: Poley
cover: '/assets/img/20220622/CrossPoint.png'
tags: 论文阅读
---

这篇文章发表于CVPR2022，主要内容是通过引入cross modal的 self-supervised来学习更好的点云编码。其整体结构如下图所示，输入3D点云，使用渲染模型渲染出对应的2D图像（任意视角），之后和点云进行对比学习。
![](/assets/img/20220622/CrossPointF1.png)

本文的方法具有三个特点：
+ 关联同时出现在点云和图像模态中的模式（patterns），比如细粒度的part-level的属性；
+ 通过引入对增广的invariance，从空间和语义属性上获取知识；
+ 对2D图像特征进行编码，作为点云特征增广的中心，从而增强3D-2D对应关系的不变性。

本方法不需要memory bank。因为认为增强和2D-3D对应已经提供了足够的特征信息。

本文的出发点在于受到人识别物体的启发，人可以很好的将2D-3D物体关联起来，得到鲁棒的编码结果。因此，本文尝试引入2D的特征编码，来和点云的transformation invariant进行关联（嵌入）。实验表明，简单的3D-2D对应即可以有益于3D点云理解。同时，孤立2D图像特征被嵌入在对应的3D点云特征附近，避免点云编码网络了对特定增广方法的Bias。最终，2D，3D的backbone都会因为联合学习而受益。
![](/assets/img/20220622/CrossPointF2.png)

在Loss设计上，比较简单。输入为一个点云与其从任意视角得到的渲染图像。对点云采用两个随机参数的增广（扰动），通过相同的编码器和映射，得到编码特征，应用对比损失，如下

$$
\begin{equation}
l\left(i, t_{1}, t_{2}\right)=-\log \frac{\exp \left(s\left(\mathbf{z}_{i}^{t_{1}}, \mathbf{z}_{i}^{t_{2}}\right) / \tau\right)}{\sum_{\substack{k=1 \\ k \neq i}}^{N} \exp \left(s\left(\mathbf{z}_{i}^{t_{1}}, \mathbf{z}_{k}^{t_{1}}\right) / \tau\right)+\sum_{k=1}^{N} \exp \left(s\left(\mathbf{z}_{i}^{t_{1}}, \mathbf{z}_{k}^{t_{2}}\right) / \tau\right)}
\end{equation}
$$

注意这里没有使用memory bank，使用的负样本来自于minibatch。在图像中这样会影响自监督学习的性能，但作者认为在点云中这样就够了。

两个对称的对比损失构成点云整体的对比损失。

$$
\begin{equation}
\mathcal{L}_{\text {imid }}=\frac{1}{2 N} \sum_{i=1}^{N}\left[l\left(i, t_{1}, t_{2}\right)+l\left(i, t_{2}, t_{1}\right)\right]
\end{equation}
$$

图像中的操作和点云上类似，但是不是在两个模态上分别使用IMID，而是cross-modal的使用IMID，使得两者在被映射到同一特征空间后具有相近的距离。目的是让图像特征尽可能靠近点云特征向量的中心。同时，图像特征和点云特征也可以起到相互稳定的作用，防止针对特定增强的bias的出现。
$$
\begin{equation}
\mathbf{z}_{i}=\frac{1}{2}\left(\mathbf{z}_{i}^{t_{1}}+\mathbf{z}_{i}^{t_{2}}\right)
\end{equation}
$$
$$
\begin{equation}
c(i, \mathbf{z}, \mathbf{h})=-\log \frac{\exp \left(s\left(\mathbf{z}_{i}, \mathbf{h}_{i}\right) / \tau\right)}{\sum_{\substack{k=1 \\ k \neq i}}^{N} \exp \left(s\left(\mathbf{z}_{i}, \mathbf{z}_{k}\right) / \tau\right)+\sum_{k=1}^{N} \exp \left(s\left(\mathbf{z}_{i}, \mathbf{h}_{k}\right) / \tau\right)}
\end{equation}
$$
$$
\begin{equation}
\mathcal{L}_{c m i d}=\frac{1}{2 N} \sum_{i=1}^{N}[c(i, \mathbf{z}, \mathbf{h})+c(i, \mathbf{h}, \mathbf{z})]
\end{equation}
$$

