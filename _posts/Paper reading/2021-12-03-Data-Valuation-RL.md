---
layout: post
title: 'Data Valuation using Reinforcement Learning'
date: 2021-12-03
author: Poley
cover: '/assets/img/20211202/DVRL.png'
tags: 论文阅读
---

> 参考博客：https://blog.csdn.net/jILRvRTrc/article/details/111569960

本文章收录于ICML2020，使用基于元学习的全新方法解决了量化训练数据价值的问题。我们的方法将数据估值整合至预测器模型的训练过程中，该模型学习识别对于给定任务具有更高价值的样本，从而改善预测器和数据估值的性能。本文使用强化学习来进行数据价值评估（DataValuation using Reinforcement Learning, DVRL）。

毫无疑问，更大规模和更高质量的数据集可以使得网络得到更强的性能。但是构建这样的数据集并不容易。

常见的数据问题有以下几种：
1. 低质量数据
2. 错误标注数据
3. 训练集和测试集的不匹配。在这些情况下，可以通过谨慎的从训练集中选择和测试集相关的场景来达到更高的性能。

在深度学习中，评估单个样本的价值是很难的，因为对于一个large-sacle的数据集，你不可能遍历它的所有子集用来训练，**这会带来不可接受的计算成本**。

本文将数据价值评估集成到模型的训练过程，使得模型可以在训练过程中受到额外的监督，同时使得模型和价值预测的性能。

本文使用一个data value estimator(DVE)来评估数据的价值，由于其在原理上就是不可导的，这里使用了强化学习来利用DVE给出的reward。

![](/assets/img/20211202/DVRLF1.png)

整体的算法流程如上所示。对于输入的样本，首先经过Data Value Estimator，其输出是一个选择概率向量，这个概率向量对应一个多项式分布（Multinomial distribution）。之后从这个多项式分布和对应的预测概率中采样，得到本次训练使用的样本，并进行正常的网络训练和反向梯度传播。之后在一个小的验证集上衡量模型的Loss，并和之前的滑动平均值进行对比，作为DVE的Reward，反过来监督DVE。

基本问题定义如下：
Framework: We denote the training dataset as $\mathcal{D}=$ $\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{N} \sim \mathcal{P}$ where $\mathbf{x}_{i} \in \mathcal{X}$ is a $d$-dimensional feature vector, and $y_{i} \in \mathcal{Y}$ is a corresponding label. We consider a disjoint testing dataset $\mathcal{D}^{t}=\left\{\left(\mathbf{x}_{j}^{t}, y_{j}^{t}\right)\right\}_{j=1}^{M} \sim \mathcal{P}^{t}$ where the target distribution $\mathcal{P}^{t}$ does not need to be the same with the training distribution $\mathcal{P}$. We assume an availability of a (small) validation dataset $\mathcal{D}^{v}=\left\{\left(\mathbf{x}_{k}^{v}, y_{k}^{v}\right)\right\}_{k=1}^{L} \sim \mathcal{P}^{t}$ that comes from the target distribution $\mathcal{P}^{t}$.

预测网络的目标函数可以表示为如下，即通过DVE加权的损失最小化
$$
\begin{equation}
f_{\theta}=\arg \min _{\hat{f} \in \mathcal{F}} \frac{1}{N} \sum_{i=1}^{N} h_{\phi}\left(\mathbf{x}_{i}, y_{i}\right) \cdot \mathcal{L}_{f}\left(\hat{f}\left(\mathbf{x}_{i}\right), y_{i}\right) .
\end{equation}
$$

而DVE的预测可以表示为如下形式
$$
\begin{equation}
\begin{array}{rc}
\min _{h_{\phi}} & \mathbb{E} \\
\text { s.t. } f_{\theta}= & \arg \min _{\hat{f} \in \mathcal{F}} \mathbb{E}_{(\mathbf{x}, y) \sim P}
\end{array}\left[\begin{array}{c}
\left.\mathcal{L}_{h}\left(f_{\theta}\left(\mathbf{x}^{v}\right), y^{v}\right)\right] \\
\left.h_{\phi}(\mathbf{x}, y) \mathcal{L}_{f}(\hat{f}(\mathbf{x}), y)\right]
\end{array}\right.
\end{equation}
$$

To encourage exploration based on the uncertainty in the exponentially-large selection space, we model training sample selection in DVE stochastically. Let $w=h_{\phi}(\mathbf{x}, y)$ denote the probability that $(\mathbf{x}, y)$ is used to train the predictor model $f_{\theta} ; h_{\phi}(\mathcal{D})=\left\{h_{\phi}\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{N}$ is the probability distribution for inclusion of each training sample; $s \in\{0,1\}^{N}$ is a binary vector that represents the selected samples. If $s_{i}=1 / 0,\left(\mathbf{x}_{i}, y_{i}\right)$ is selected/not selected for training the predictor model. $\pi_{\phi}(\mathcal{D}, \mathbf{s})=\prod_{i=1}^{N}\left[h_{\phi}\left(\mathbf{x}_{i}, y_{i}\right)^{s_{i}} \cdot(1-\right.$ $\left.\left.h_{\phi}\left(\mathbf{x}_{i}, y_{i}\right)\right)^{1-s_{i}}\right]$ is the probability that the selection vector $\mathbf{s}$ is selected based on $h_{\phi}(\mathcal{D})$.

由于采样过程是不可导的，因此DVE不能使用梯度下降来进行优化。处理不可导优化瓶颈的方法有很多，比如Gumbel-softmax或者随机反向传播。本文中使用强化学习来作为优化的手段。在该RL设置中，代理的操作（DVE）是其数据选择，并且包含预测模型训练和评估的环境根据当前批次数据的状态相应地为每个操作提供奖励。我们采用强化算法使用策略梯度进行优化，并从接近目标任务性能的小验证集获得奖励。

故根据single step reward based on the action，得到
$$
\begin{equation}
\begin{aligned}
\hat{l}(\phi)=& \underset{\left(\mathbf{x}^{v}, y^{v}\right) \sim P^{t},}{\mathbb{E}^{t}}\left[\mathcal{L}_{h}\left(f_{\theta}\left(\mathbf{x}^{v}\right), y^{v}\right)\right] \\
& \mathbf{s} \sim \pi_{\phi}(\mathcal{D}, \cdot) \\
&=\int P^{t}\left(\mathbf{x}^{v}\right) \sum_{\mathbf{s} \in[0,1]^{N}} \pi_{\phi}(\mathcal{D}, \mathbf{s}) \cdot\left[\mathcal{L}_{h}\left(f_{\theta}\left(\mathbf{x}^{v}\right), y^{v}\right)\right] d \mathbf{x}^{v},
\end{aligned}
\end{equation}
$$
$$
\begin{equation}
\begin{aligned}
&\text { which has the gradient: }\\
&\nabla_{\phi} \hat{l}(\phi)=\int P^{t}\left(\mathbf{x}^{v}\right) \sum_{\mathbf{s} \in[0,1]^{N}} \nabla_{\phi} \pi_{\phi}(\mathcal{D}, \mathbf{s})\\
&\cdot\left[\mathcal{L}_{h}\left(f_{\theta}\left(\mathbf{x}^{v}\right), y^{v}\right)\right] d \mathbf{x}^{v}\\
&=\int P^{t}\left(\mathbf{x}^{v}\right)\left[\sum_{\mathbf{s} \in[0,1]^{N}} \frac{\nabla_{\phi} \pi_{\phi}(\mathcal{D}, \mathbf{s})}{\pi_{\phi}(\mathcal{D}, \mathbf{s})} \pi_{\phi}(\mathcal{D}, \mathbf{s})\right.\\
&\left.\cdot\left[\mathcal{L}_{h}\left(f_{\theta}\left(\mathbf{x}^{v}\right), y^{v}\right)\right]\right] d \mathbf{x}^{v}\\
&=\int P^{t}\left(\mathbf{x}^{v}\right)\left[\sum_{\mathbf{s} \in[0,1]^{N}} \nabla_{\phi} \log \left(\pi_{\phi}(\mathcal{D}, \mathbf{s})\right) \cdot \pi_{\phi}(\mathcal{D}, \mathbf{s})\right.\\
&\left.\cdot\left[\mathcal{L}_{h}\left(f_{\theta}\left(\mathbf{x}^{v}\right), y^{v}\right)\right]\right] d \mathbf{x}^{v}\\
&=\underset{\left(\mathbf{x}^{v}, y^{v}\right) \sim P^{t}, \atop \mathbf{s} \sim \pi_{\phi}(\mathcal{D}, \cdot)}{\mathbb{E}}\left[\mathcal{L}_{h}\left(f_{\theta}\left(\mathbf{x}^{v}\right), y^{v}\right)\right] \nabla_{\phi} \log \left(\pi_{\phi}(\mathcal{D}, \mathbf{s})\right) .\\
&\text { where } \nabla_{\phi} \log \left(\pi_{\phi}(\mathcal{D}, \mathbf{s})\right)\\
&=\nabla_{\phi} \sum_{i=1}^{N} \log \left[h_{\phi}\left(\mathbf{x}_{i}, y_{i}\right)^{s_{i}} \cdot\left(1-h_{\phi}\left(\mathbf{x}_{i}, y_{i}\right)\right)^{1-s_{i}}\right]
\end{aligned}
\end{equation}
$$

这里还额外加入了一个边缘信息（margin information）来为DVE提供更多的信息，即在原训练数据上训练的模型和标签之间的区别。

$$
\begin{equation}
m(\mathbf{x}, y)=\left|y-f_{v}(\mathbf{x})\right|
\end{equation}
$$

其算法伪代码如下所示
![](/assets/img/20211202/DVRLA1.png)

## Experiments
![](/assets/img/20211202/DVRLF2.png)
