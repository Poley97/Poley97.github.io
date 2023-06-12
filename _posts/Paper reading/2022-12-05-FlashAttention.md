---
layout: post
title: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness'
date: 2022-12-05
author: Poley
cover: '/assets/img/20221205/FlashAttention.jpg'
tags: 论文阅读
---
## Introduction
Transformer由于其O(N^2)的时间复杂度限制了其序列长度。目前的一些优化方法主要通过在Model quality和compute complexity之间进行trade off，但这不一定使其实现wall-clock speed up。

作者提出现有方法一定程度上被GPU的IO所限制，因此作者提出IO-Aware的方法Flash Attention，很大程度上消除了模型中由于GPUIO造成的性能瓶颈，在运行速度上和显存占用上均有很大的改进。

现有的方法主要从sparse-approximation或者low-rand approximation的角度来优化，即减少FLOP。但是这不意味着运行时间的直接减少。作者提出现有很多Transformer的操作都被memory access所瓶颈，但是pytorch和tensorflow这样的并不支持对于memory access的细粒度控制。

为了实现IO-Aware的方法，应该尽量避免从HBM中读取矩阵，而应该多从片上的SRAM中读取。这对应两个需求：
1. softmax不需要access 整个输入 
2. 不需要存储大型的中间态的注意力矩阵用于反传。

具体方法：
1. 将注意力计算分分块，递增的进行softmax（平铺）
2. 保存softmax 归一化factor，在前向网络中，来在反传中on-chip计算attention。这比从HBM中读取矩阵更快。
3. 通过CUDA实现对内存的精细控制。即使Flops更多了，但是速度也更快，使用的显存也更少。

进一步的，针对经常使用的mask,作者提出block-sparse Flash Attention。即FA的稀疏版本，进一步加速。其IO和FA相差一个关于稀疏度（ratio）的比例因子。

## Background
现有方法，计算速度和显存速度息息相关。这说明操作逐步被内存访问所瓶颈。不同位置的读写速度如下图所示
![](/assets/img/20221205/FlashAttentionF1.jpg)

GPU的执行方式是，执行方式，将kernal从HBM中读取数据到寄存器和SRAM中，再输出回HBM。算法是compute-bound还是Memory-bound。衡量方法，arithmetic intensity，算数强度，即每个比特运算的内存访问量。
前者典型的是大channel卷积和大中间维度的矩阵乘法，后者典型是elementwise和reduction。

缓解上述问题的一种有效的方法是kernel fusion。即读取一组数据后连续进行多个操作。但是训练中中间变量也要存到HBM中，减弱了kernel fusion的效率。

标准的Transformer操作如下

$$
\begin{equation}
\mathrm{S}=\mathrm{QK}^{\top} \in \mathbb{R}^{N \times N}, \quad \mathrm{P}=\operatorname{softmax}(\mathrm{S}) \in \mathbb{R}^{N \times N}, \quad \mathrm{O}=\mathrm{PV} \in \mathbb{R}^{N \times d},
\end{equation}
$$

其中将中间矩阵存在显存中，非常浪费。而且其中softmax,masking和dropout都是很费时间的elementwise方法。尤其是softmax，其在column上是耦合的，因此softmax的归一化因子需要将attention矩阵的整行读入，内存访问量很大。其标准的流程如下

![](/assets/img/20221205/FlashAttentionA0.jpg)

其中，根据矩阵乘法的算法，矩阵乘法是按Block读的，elementwise是无所谓的。



## FlashAttention

### Tiling and Recomputation
总体来说，将QKV分解成Block。在叠加每个block的输出之前，通过正确的归一化因子缩放，就可以得到正确结果。

对于一个$x \in \mathbb{R}^B$,softmax的计算可以写成如下形式（为了防止溢出，乘了一个负指数）
$$
\begin{equation}
m(x):=\max _i \quad x_i, \quad f(x):=\left[\begin{array}{lll}
e^{x_1-m(x)} & \ldots & e^{x_B-m(x)}
\end{array}\right], \quad \ell(x):=\sum_i f(x)_i, \quad \operatorname{softmax}(x):=\frac{f(x)}{\ell(x)}
\end{equation}
$$

将整个vector一起计算softmax。并且拓展到多个vector串联的情况上。这一步主要是为了分解过大的N（矩阵列数）和softmax之间的耦合关系。将很大的列数分解成若干小Block进行计算，使得一个Block的计算可以在a time下完成（而不需要反复的内存读取）。
For vectors $x^{(1)}, x^{(2)} \in \mathbb{R}^B$, we can decompose the softmax of the concatenated $x=\left[x^{(1)} x^{(2)}\right] \in \mathbb{R}^{2 B}$ as:
$$
\begin{aligned}
& m(x)=m\left(\left[x^{(1)} x^{(2)}\right]\right)=\max \left(m\left(x^{(1)}\right), m\left(x^{(2)}\right)\right), \quad f(x)=\left[e^{m\left(x^{(1)}\right)-m(x)} f\left(x^{(1)}\right) \quad e^{m\left(x^{(2)}\right)-m(x)} f\left(x^{(2)}\right)\right] \\
& \ell(x)=\ell\left(\left[x^{(1)} x^{(2)}\right]\right)=e^{m\left(x^{(1)}\right)-m(x)} \ell\left(x^{(1)}\right)+e^{m\left(x^{(2)}\right)-m(x)} \ell\left(x^{(2)}\right), \quad \text { softmax }(x)=\frac{f(x)}{\ell(x)}
\end{aligned}
$$
这里相比上述，只是在combine的时候需要对不同的Block乘以对应的指数系数。因此可以分块计算，再combine。

在Recomputation上，这里不保存中间变量，在反传的过程中使用同样的方法计算注意力矩阵。**但是，这里保存softmax的全局归一化因子**。以避免重复无用的内存访问（这部分所占据的存储空间很像，相比于内存访问代价），相当于是选择性的checkinngpoint。综上，其算法如下
![](/assets/img/20221205/FlashAttentionA1.jpg)

Q一次至少要读一整个d维数据。KV随意。一次读取M/4大小的矩阵，原因是， Q K V O各占M的1/4。主要还是解决全局的elementwise的问题，主要是softmax。这里使用叠加递增的方式解决了这个问题。

另外的，对于又mask的情况，这里类似于上面，只是在block计算的时候直接跳过zero-block。根据mask的稀疏率，计算速度可以进一步提升。

在实验上

![](/assets/img/20221205/FlashAttentionF2.jpg)
![](/assets/img/20221205/FlashAttentionF3.jpg)
其相比标准的transformer，在显存占用和运算时间上都有了极大的提升，尤其是对Pytorch的实现版本，实现了20倍的显存优化。但是注意到，这里获得的收益是逐渐边缘化的，当on-chip-cache足够大时，runtime的性能瓶颈会逐渐从memory access转换到数学运算上，这部分就不是本方法优化的内容了。CMT的sequence实际上比他长，这里是1024平方，cmt可能是900*20000。因此应该也会获得很大的收益。

以下是其部分实验结果，总体来说，其在运算上和原有transformer是等价的，但是训练时间和占用显存大幅减小，使得在相同的时间/内存代价下，可以训练更大的模型，变相提高了模型的精度。
![](/assets/img/20221205/FlashAttentionT1.jpg)
![](/assets/img/20221205/FlashAttentionT2.jpg)
![](/assets/img/20221205/FlashAttentionT3.jpg)
![](/assets/img/20221205/FlashAttentionT4.jpg)

