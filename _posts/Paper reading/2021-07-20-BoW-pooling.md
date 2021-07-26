---
layout: post
title: 'BoW Pooling: A Plug-and-Play Unit for Feature Aggregation of Point Clouds'
date: 2021-07-20
author: Poley
cover: '/assets/img/20210720/BoW.png'
tags: 论文阅读
---

> 论文链接：https://www.aaai.org/AAAI21Papers/AAAI-1163.ZhangX.pdf

点云的特征提取往往使用max pooling。但是坐着认为这会不可避免的丢失一些discrimination information。于是本文提出了BoW pooling，分析局部特征的统计特性，并且创立一个直方图来反应各个局部是如何组成一个全局的几何形状的。

# Introduction
如今点方法普遍使用average pooling 或者 max pooling，在实践中，max pooling的效果略好一些。但是其有两个主要缺点。

+ 对所有的特征使用 max pooling来得到global representation可能导致很多点相同的贡献。虽然global representaion是通过在每一个维度上判断特征的最大值来得到的。 那么每个点的贡献就通过其有多少值出现在global representation中来衡量，如下图所示，大多数点都做出了相同的贡献。**而实际上不同位置的点对于几何感知的重要性是不同的。**

![](/assets/img/20210720/BoWF1.png)

+ 使用max pooling从原理上的可解释性不强。

Bag of words是一种非常经典的方法。参考它的思路，本文作者提出一个3D模型也可以被视为primitives的组合，只是type,number,arrangement有所不同。因此本文创建一个dictionary来保存这些primitives的features，然后将global representation of an object表示为一个直方图，来表征primitivs如何构建一个3D物体。同时，引入了 truncated linear unit(TLU)来避免前面说的到平均贡献。它通过限制局部特征的表达能力来实现这一点，对比如上图所示。

# Method

![](/assets/img/20210720/BoWF2.png)
## BoW pooling: A Generalized BoW Layer
对于已有的K个聚类中心，通过距离阈值来进行二值判断，并最终组成一个K维global representation

$$
\begin{equation}
V_{j}=\sum_{i=1}^{n} I\left(\operatorname{dist}\left(x_{i}, c_{j}\right)<\operatorname{dist}\left(x_{i}, c_{k \neq j}\right)\right)
\end{equation}
$$

距离度量使用的是余弦距离，这样相似度矩阵可以简单的通过矩阵乘法得到。

## Soft point assignment to the points

### Softmax-like function
上述的分配方式是不可导的，因此可以用另一种方式，即softmax-like function。(其实也就是constractive learning里那个判别函数,个人认为)

$$
\begin{equation}
V_{i j}=\frac{e^{\alpha x_{i}^{T} c_{j}}}{\sum_{k=1}^{K} e^{\alpha x_{i}^{T} c_{k}}}
\end{equation}
$$

### Truncated linear unit

设计初衷是一个点对于远距离的center的贡献一般可以忽略，所以根据相似度做一个判别，将点归属给相似度最大的L个中心。

$$
\begin{equation}
V_{i j}= \begin{cases}x_{i}^{T} c_{j}, & \text { if } j \in I_{i} \\ 0, & \text { otherwise }\end{cases}
\end{equation}
$$

则点特征可以表示为如下形式

$$
\begin{equation}
\mathcal{T}\left(q_{i}\right)=\left(q_{i}^{1} \cdot \beta^{1}, \ldots, q_{i}^{\mathcal{D}} \cdot \beta^{\mathcal{D}}\right)
\end{equation}
$$

其中$/beta$是上述的指示符。最后通过一个aggregation操作将各点的截断特征融合

$$
\begin{equation}
f_{\text {global }}=\mathcal{A}\left(\mathcal{T}\left(q_{i}\right), \ldots, \mathcal{T}\left(q_{n}\right)\right)
\end{equation}
$$

TLU的目的在于增加外巨额局部特征的难度，并且拓展了不同区域之间的discrimination capacities gap。因此会有更小的一组局部特征提供对于Global representation的主要贡献，可以帮助网络更好的容忍同意类别内的shape variations。尤其是非刚性形状。

## Mixing Dictionary Update Strategy
可以使用梯度来直接更新字典，但是这样字典会不能总结出呗多个类别所共享的primitives（全局性不够）。

更新策略使用EM/Kmeans来进行。但是对于每个batch都做EM算法，会让dictionary迭代调整，很难平衡时间消耗和收敛效率。因此本文用一种启发式的方法，每T个epoch更新一次字典。（这跟memory也很像，或者说这就是memory）。并且初始值$c_i$直接通过BP学习得到(这里没说怎么得到)。

# Experiments
![](/assets/img/20210720/BoWT2.png)

![](/assets/img/20210720/BoWF4.png)
