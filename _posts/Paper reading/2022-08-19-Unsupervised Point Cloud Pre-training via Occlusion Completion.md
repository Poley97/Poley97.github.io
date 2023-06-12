---
layout: post
title: 'Unsupervised Point Cloud Pre-training via Occlusion Completion'
date: 2022-08-19
author: Poley
cover: '/assets/img/20220818/oCoC.jpg'
tags: 论文阅读
---

参考博客：https://zhuanlan.zhihu.com/p/395652964

近年来3d计算机视觉相关的研究越来越多，但和研究需求相比，已标注的3d数据集仍然较少。为了更好的利用这些数据进行3d视觉的研究，模型的预训练十分重要。因此，这篇文章提出了OcCo: 一种简单有效的，无监督的方法，得到的预训练模型可以广泛运用在3d分类，分割等任务上。

# 基本思路
当从不同视角观察一个物体（由点云组成）时，不同部分的点会被遮挡住。此文设计了一个基于编码器-解码器的模型，输入遮挡的点云，然后输出补全的点云。在进行这个‘补全任务‘时，模型可以学到点位置和语义的信息，从而为其他任务（比如分类）提供有用的预训练权重。

![](/assets/img/20220818/oCoCF2.jpg)

## Step 1: 模拟遮挡
为了模拟遮挡，需要先把点云投射到一个相机坐标系中，确定会被遮挡的点，然后再投射回去。

+ 将点云投射到相机坐标系（2d）。
通过定义相机的矩阵/参数，可以确定不同的相机角度，并将点云投射到相机的坐标系中。
+ 找到被遮挡的点。此时在相机平面上有相同坐标的2d点就是可能被遮挡的点。为了确定这些点的前后关系，此文用Delaunay triangulation重建了mesh，然后将隐藏面里面的点标为被遮挡的点。
+ 投射回世界坐标系。将被遮挡的点删除后，继续使用之前定义的相机参数，将剩余的点投射回去。

笔者认为这相当于一定程度上模拟了
激光雷达对点云的2.5D扫描（即只能扫描可见位置）这里作者使用这种方法来创建了补全任务的输入（以完整点云作为监督）

## Step 2 : 完成补全任务
+ 通过一个编码器-解码器的网络结构，输入第一步得到的点云，进行编码（可使用PointNet, DGCNN等），生成一个1024-dimension的矢量，然后解码（可使用PCN），得到补全的点云。解码器会生成两种输出：一个粗糙的点云（1024个点），一个更细致的点云（16384个点）。
+ 损失函数由两个部分组成：输出的粗糙点云$P_{coarse}$和真实值之间的Chamfer Distance (CD) ，和输出的细致点云和真实值$P_{fine}$之间的Chamfer Distance (CD)。这部分基本上来自于PCN，没有什么创新。

注： Chamfer Distance是一种点云配准和分析中常用的损失，可以收敛到局部最优。

$$
\begin{equation}
\begin{aligned}
&\mathrm{CD}(\hat{\mathcal{P}}, \mathcal{P})= \\
&\quad \frac{1}{|\hat{\mathcal{P}}|} \sum_{\hat{x} \in \hat{\mathcal{P}}} \min _{x \in \mathcal{P}}\|\hat{x}-x\|_{2}+\frac{1}{|\mathcal{P}|} \sum_{x \in \mathcal{P}} \min _{\hat{x} \in \mathcal{\mathcal { P }}}\|x-\hat{x}\|_{2} .
\end{aligned}
\end{equation}
$$
$$
\begin{equation}
\ell:=\operatorname{CD}\left(\hat{\mathcal{P}}_{\text {coarse }}, \mathcal{P}_{\text {coarse }}\right)+\alpha \mathrm{CD}\left(\hat{\mathcal{P}}_{\text {fine }}, \mathcal{P}_{\text {fine }}\right)
\end{equation}
$$


此文选择了ModelNet40作为数据集，对每个点云，随机生成了10个不同相机角度，然后采用了不同编码器 (PointNet, PCN, DGCNN) 进行试验。为了测试预训练模型的效果，作者将预训练模型放到不同任务中（Few-shot learning，分类，分割等）进行测试，并和其他方法（Jigsaw, random等）进行了比较。在这些任务中，此文提出的预训练方法，都能取得更好的结果（比如下图所示的分类结果）。

![](/assets/img/20220818/oCoCT3.jpg)

分类任务的结果（来自原文）
之后此文还对预训练模型得到的特征进行了各种可视化，作进一步分析。从中可以看出，在进行补全任务时，模型已经很好的学习到了简单的几何信息。如下图所示，后续的特征维度中，一些特征已经展现出了和某些几何形状的强关联性，从中已经可以看出一些几何图形，比如圆锥，圆柱体等。这些分析很好的证实了此文方法的原理。

![](/assets/img/20220818/oCoCF3.jpg)