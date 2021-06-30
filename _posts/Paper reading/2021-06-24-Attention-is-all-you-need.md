---
layout: post
title: 'Attention is all you need'
date: 2021-06-24
author: Poley
cover: '/assets/img/20210624/ATT.png'
tags: 论文阅读
---

> 论文链接：https://arxiv.org/pdf/1706.03762v5.pdf
> 参考博客：https://zhuanlan.zhihu.com/p/46990010
**Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.**

# Model Archtecture

![](/assets/img/20210624/ATTF1.png)

## Attention
Attention的定义如下：

**An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors**

**注意力函数可以描述为将一个query和一组key-value对做映射并输出，其中它们都是向量。**

本文使用的attention称为**Scaled Dot-Product Attention**，非常经典，公式如下，其中$d_k$是Dimension。
$$
\begin{equation}
\operatorname{Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
\end{equation}
$$

常见的attention function主要有两种，**additive attention and dot-product**。这两个理论复杂度差不多，但是dot在实践中更快，空间复杂度更低，因为他可以使用高度优化的矩阵乘法代码。

对于小的$d_k$来说，两者的性能差不多，但是对于大$d_k$来说，additive attention更好，这可能是由于点积的结果随着维度的增长而增长，因此这里加入了一个归一化因子$\frac{1}{\sqrt{d_k}}$。

## Multi-Head Attention
相比使用一个单独维度的attention function，坐着发现使用多个不同维度的attention function重复多次效果更好。再输出时将不同维度的输出接到一起即可。如下图所示。

![](/assets/img/20210624/ATTF2.png)

可以表示为如下形式

$$
\begin{equation}
\begin{aligned}
\operatorname{MultiHead}(Q, K, V) &=\text { Concat }\left(\text { head }_{1}, \ldots, \text { head }_{\mathrm{h}}\right) W^{O} \\
\text { where head }_{\mathrm{i}} &=\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right)
\end{aligned}
\end{equation}
$$

## Point-wise Fee-Forward Networks
即再每个注意力函数之后加一个变换，对于每个位置的信息单独处理，如下
$$
\begin{equation}
\operatorname{FFN}(x)=\max \left(0, x W_{1}+b_{1}\right) W_{2}+b_{2}
\end{equation}
$$

## Positional Encoding
因为上述的注意力模块是没有捕捉到时序信息的，因此需要人为加入时序信息，即**Positional Encoding**。

$$
\begin{equation}
\begin{aligned}
P E_{(p o s, 2 i)} &=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(\text {pos }, 2 i+1)} &=\cos \left(\text { pos } / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
\end{equation}
$$

上述提供的是绝对位置信息，而相对位置信息也很重要。这里选用sin,cos函数是因为其满足性质
$$
\begin{equation}
\begin{aligned}
&\sin (\alpha+\beta)=\sin \alpha \cos \beta+\cos \alpha \sin \beta \\
&\cos (\alpha+\beta)=\cos \alpha \cos \beta-\sin \alpha \sin \beta
\end{aligned}
\end{equation}
$$

这表明p+k位置可以向量可以表示为p位置的线性组合，这为捕捉相对位置信息提供了可能。

在其他NLP论文中，大家也都看过position embedding，通常是一个训练的向量，但是position embedding只是extra features，有该信息会更好，但是没有性能也不会产生极大下降，因为RNN、CNN本身就能够捕捉到位置信息，但是在Transformer模型中，Position Embedding是位置信息的唯一来源，因此是该模型的核心成分，并非是辅助性质的特征。

也可以采用训练的position embedding，但是试验结果表明相差不大，因此论文选择了sin position embedding，因为

这样可以直接计算embedding而不需要训练，减少了训练参数
这样允许模型将position embedding扩展到超过了training set中最长position的position，例如测试集中出现了更大的position，sin position embedding依然可以给出结果，但不存在训练到的embedding。

## Why Self Attention

这里将Self-Attention layers和recurrent/convolutional layers来进行比较，来说明Self-Attention的好处。假设将一个输入序列分别用

+ Self-Attention Layer
+ Recurrent Layer
+ Convolutional Layer
来映射到一个相同长度的序列

我们分析下面三个指标：

+ 每一层的计算复杂度
+ 能够被并行的计算，用需要的最少的顺序操作的数量来衡量
+ 网络中long-range dependencies的**path length**，在处理序列信息的任务中很重要的在于学习long-range dependencies。影响学习长距离依赖的关键点在于前向/后向信息需要传播的步长，输入和输出序列中路径越短，那么就越容易学习long-range dependencies。因此我们比较三种网络中任何输入和输出之间的最长path length
结果如下所示

![](/assets/img/20210624/ATTT1.png)

### 并行计算
Self-Attention layer用一个常量级别的顺序操作，将所有的positions连接起来

Recurrent Layer需要$O(b)$个顺序操作

### 计算复杂度分析
如果序列长度$n <$表示维度$d$ ，Self-Attention Layer比recurrent layers快，这对绝大部分现有模型和任务都是成立的。

为了提高在序列长度很长的任务上的性能，我们对Self-Attention进行限制，只考虑输入序列中窗口为$r$ 的位置上的信息，这称为Self-Attention(restricted), 这回增加maximum path length到$O(n/r)$ .

### length path
如果卷积层kernel width$k<n$ ，并不会将所有位置的输入和输出都连接起来。这样需要$O(k/n)$ 个卷积层或者 $O(log_k(n))$ 个dilated convolution，增加了输入输出之间的最大path length。

卷积层比循环层计算复杂度更高，是k倍。但是Separable Convolutions将见效复杂度。

同时self-attention的模型可解释性更好(interpretable).