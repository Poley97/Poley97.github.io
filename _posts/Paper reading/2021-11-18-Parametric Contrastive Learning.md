---
layout: post
title: 'Parametric Contrastive Learning'
date: 2021-11-18
author: Poley
cover: '/assets/img/20211116/PaCo.png'
tags: 论文阅读
---

提出 Parametric Contrastive Learning(PaCo) 方法，自适应的增强将同类样本聚集在一起的力量，进而加强对于Hard example的学习。

尽管监督对比学习(supervised contrastive learning)在平衡数据集上效果很好，但是其高频率类别相比低频率类别具有更好的损失下限（lower bound of loss），导致其具有远高于低频类别的贡献。

如下图所示，训练中各个类别的平均梯度随着类别出现的频率而显著下降。本文的目的在于解决上述监督对比学习的不平衡问题，通过一组参数化的class-wise learnable centers。
![](/assets/img/20211116/PaCoF2.png)
## Non-parametric Contrastive Loss
典型的对比学习损失是InfoNCE,其是一个无参数的损失函数，如下所示
$$
\begin{equation}
\mathcal{L}_{q, k^{+},\left\{k^{-}\right\}}=-\log \frac{\exp \left(q \cdot k^{+} / \tau\right)}{\exp \left(q \cdot k^{+} / \tau\right)+\sum_{k^{-}} \exp \left(q \cdot k^{-} / \tau\right)}
\end{equation}
$$

因为传统的交叉熵（配全连接层），可以表示如下，是参数化的
$$
\begin{equation}
\mathcal{L}_{\text {cross-entropy }}=-\log \frac{\exp \left(q \cdot w_{y}\right)}{\sum_{i=1}^{n} \exp \left(q \cdot w_{i}\right)} .
\end{equation}
$$

## Parametric Contrastive Learning 
### Supervised Contrastive Learning

MoCo中的监督对比学习如下
$$
\begin{equation}
\mathcal{L}_{i}=-\sum_{z_{+} \in P(i)} \log \frac{\exp \left(z_{+} \cdot T\left(x_{i}\right)\right)}{\sum_{z_{k} \in A(i)} \exp \left(z_{k} \cdot T\left(x_{i}\right)\right)} .
\end{equation}
$$

其中
$$
\begin{equation}
\begin{aligned}
&A(i)=\left\{z_{k} \in \text { queue } \cup Z_{v 1} \cup Z_{v 2}\right\} \backslash\left\{z_{k} \in Z_{v 1}: k=i\right\} \\
&P(i)=\left\{z_{k} \in A(i): y_{k}=y_{i}\right\}
\end{aligned}
\end{equation}
$$

这里把同类的样本都算作正样本，而不是来自同一张图片的样本，即加入了监督信息。

### Theoretical Motivation
#### Analisis of Supervised Contrastive Learning
![](/assets/img/20211116/PaCoT1.png)

如上表所示，直接将监督对比学习用于长尾分布反而出现了性能的下降。因为其更多的关注高频类别，这对不平衡数据是不利的。

从上述监督对比学习的公式可以看出，当两个samples是true positive pair时，最优的相似概率是$1/K_y$，其中$K_y$是batch中包含的此类别样本数。如果$q(y)$是本类别在数据集中出现的频率，那么可以得到$K_y \approx length(queue)\cdot q(y)$。

因此，对于高频的类别，其损失的下界要明显高于低频类别（$-\log(\frac{1}{K_y})$）。这意味着训练会更多的倾向于高频类别。

### Rebalance in Contrastive Learning
为了解决对比学习中的上述问题，本文引入一组 parametric class-wise learnable centers $\mathbf{C}=\{c_1,c_2,...,c_n\}$本文对对比学习进行了加权，如下所示。

$$
\mathcal{L}_{i}=\sum_{z_{+} \in P(i) \cup\left\{c_{y}\right\}}-w\left(z_{+}\right) \log \frac{\exp \left(z_{+} \cdot T\left(x_{i}\right)\right)}{\sum_{z_{k} \in A(i) \cup \mathbf{C}} \exp \left(z_{k} \cdot T\left(x_{i}\right)\right)}
$$
where
$$
w\left(z_{+}\right)=\left\{\begin{array}{l}
\alpha, \quad z_{+} \in P(i) \\
1.0, \quad z_{+} \in\left\{c_{y}\right\}
\end{array}\right.
$$
and
$$
z \cdot T\left(x_{i}\right)= \begin{cases}z \cdot \mathcal{G}\left(x_{i}\right), & z \in A(i) \\ z \cdot \mathcal{F}\left(x_{i}\right), & z \in \mathbf{C}\end{cases}
$$

个人理解是为每个类别增加了一个类别中心特征，并作为一个额外的正样本。同时，对原有监督对比损失进行加权。如此，两个true positive样本之间的最优相似度就变为$\frac{\alpha}{1+\alpha\cdot K_y}$，而和其对应中心的最优相似度为$\frac{1}{1+\alpha\cdot K_y}$，同样，$K_y \approx length(queue)\cdot q(y)$。则对于两个出现频率不同的类别$y_h,y_t$，其各自的true positive pair的相似度就由$\frac{1}{K_{y_{h}}} \longrightarrow \frac{1}{K_{y_{t}}}$变为$\frac{1}{\frac{1}{\alpha}+K_{y_{h}}} \longrightarrow \frac{1}{\frac{1}{\alpha}+K_{y_{t}}}$。$\alpha$越小i，两者的损失下界越接近，这有利于低频类别的学习。

另一方面，当$\alpha$越小，样本之间的对比强度就越来越低，样本和中心的对比就越来越强烈，**这使得损失逐步趋近于有监督的交叉熵**。为了兼顾平衡与对比学习，作者认为$\alpha=0.05$是一个较好的选择。

![](/assets/img/20211116/PaCoF3.png)
### PaCo under Balanced Setting

在平衡状态下，$q^{*}=q\left(y_{i}\right)=q\left(y_{j}\right)$ and $K^{*}=K_{y_{i}}=K_{y_{j}}$

此时，loss可以写成如下形式

$\begin{aligned} \mathcal{L}_{i} &=\sum_{z_{+} \in P(i) \cup\left\{c_{y}\right\}}-w\left(z_{+}\right) \log \frac{\exp \left(z_{+} \cdot T\left(x_{i}\right)\right)}{\sum_{z_{k} \in A(i) \cup \mathbf{C}} \exp \left(z_{k} \cdot T\left(x_{i}\right)\right)} \\ &=-\log \frac{\exp \left(c_{y} \cdot \mathcal{F}\left(x_{i}\right)\right)}{\operatorname{ExpSum}}-\alpha \sum_{z_{+} \in P(i)} \log \frac{\exp \left(z_{+} \cdot \mathcal{G}\left(x_{i}\right)\right)}{\operatorname{ExpSum}} \\ &=\mathcal{L}_{\text {sup }}+\alpha \mathcal{L}_{\text {supcon }}-\left(\log P_{\text {sup }}+\alpha K^{*} \log P_{\text {supcon }}\right) \\ &=\mathcal{L}_{\text {sup }}+\alpha \mathcal{L}_{\text {supcon }}-\left(\log P_{\text {sup }}+\alpha K^{*} \log \left(1-P_{\text {sup }}\right)\right), \end{aligned}$
where $\left\{\begin{aligned} P_{s u p} &=\frac{\sum_{c_{k} \in \mathbf{C}} \exp \left(c_{k} \cdot \mathcal{F}\left(x_{i}\right)\right)}{\operatorname{ExpSum}} \\ P_{\text {supcon }} &=\frac{\sum_{z_{k} \in A(i)} \exp \left(z_{k} \cdot \mathcal{G}\left(x_{i}\right)\right)}{\operatorname{ExpSum}} . \end{aligned}\right.$

此时损失变成了一个监督交叉熵损失+一个固定权重的监督对比损失。当两个损失冲突时，可能会导致收敛速度变慢或者次优。PaCo则通过动态的调整两者的强度来自适应的避免冲突。

PaCo中除了交叉熵和对比损失外，后面对了一项额外损失
$$\mathcal{L}_{\text {extra }}=-\log \left(P_{\text {sup }}\right)-\alpha K^{*} \log \left(1-P_{\text {sup }}\right)
$$

在$q^*=0.001,\text{length}(queu)=8192,\alpha=0.05, \alpha K^*=0.41$的情况下，其函数曲线如下，其中横轴是$P_{sup}$
![](/assets/img/20211116/PaCoF4.png)

可以看到当$P_{sup}=0.71$时，$L_extra$达到最小。这个结论在非平衡状态下也成立。

综上可以得到两个true positive之间的最优相似度是0.035。

设$P_{sup}=p$，$L_{supcon}$随着$p$的增大而减小，如下所示

$\begin{aligned} \mathcal{L}_{\text {supcon }} &=-\sum_{z_{+} \in P(i)} \log \frac{\exp \left(z_{+} \cdot \mathcal{G}\left(x_{i}\right)\right)}{\sum_{z_{k} \in A(i)} \exp \left(z_{k} \cdot \mathcal{G}\left(x_{i}\right)\right)} \\ &=-\sum_{z_{+} \in P(i)} \log \frac{\frac{\exp \left(z_{+} \cdot \mathcal{G}\left(x_{i}\right)\right)}{\operatorname{Exp} S u m}}{\frac{\sum_{z_{k} \in A(i)} \exp \left(z_{k} \cdot \mathcal{G}\left(x_{i}\right)\right)}{\operatorname{ExpSum}}} \\ &=-K^{*} \log \frac{0.035}{1-p} \end{aligned}$

这意味着随着各类的特征愈发的向类别中心集中，$L_{supcon}$需要显著减小来获得最优解，即增大类间距离。这意味着一个逐步增强的对比损失，最终使得目标更加更加集中于类别中心。

### Center Learning Rebalance
上述工作只进行了对比损失的平衡，但是同样，对于center的学习也需要平衡。这里作者直接引入了另一个工作 Balance Softmax，只针对类别中心的损失做一个平衡，如下，得到最终的损失

$$
\mathcal{L}_{i}=\sum_{z_{+} \in P(i) \cup\left\{c_{y}\right\}}-w\left(z_{+}\right) \log \frac{\psi\left(z_{+}, T\left(x_{i}\right)\right)}{\sum_{z_{k} \in A(i) \cup \mathbf{C}} \psi\left(z_{k}, T\left(x_{i}\right)\right)}
$$
where
$$
\psi\left(z_{k}, T\left(x_{i}\right)\right)= \begin{cases}\exp \left(z_{k} \cdot \mathcal{G}\left(x_{i}\right)\right), & z_{k} \in A(i) \\ \exp \left(z_{k} \cdot \mathcal{F}\left(x_{i}\right)\right) \cdot q\left(y_{k}\right), & z_{k} \in \mathbf{C}\end{cases}
$$

## Experiments 
![](/assets/img/20211116/PaCoF6.png)