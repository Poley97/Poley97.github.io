---
layout: post
title: 'Momentum2 Teacher: Momentum Teacher with Momentum Statistics for Self-Supervised Learning'
date: 2022-06-29
author: Poley
cover: '/assets/img/20220627/Momentum2Teacher.png'
tags: 论文阅读
---

这是旷视团队在2021年发表在arXiv上的一篇论文，内容比较简单，即提出了使用Momentum来在自监督中对teacher的BN层进行更新，从而获得更加稳定的统计特性。

自监督方法中，获得稳定的统计特性至关重要。因此，像BYOL这样的SOTA方法一般都使用4096这样的大batchsize来获得更稳定的数据统计特性，当使用小batch时（比如128），性能便出现显著下降。

这里提出Momentum Teacher，相比传统方法。传统Teacher使用动量更新参数，based on student参数。BN参数这来自于Teacher的输入参数自身，而这里对BN的针对参数自身historical的，而不是student的。通过引入historical，相当于隐形的增大了batchsize，从而引入更加稳定的统计特性。

![](/assets/img/20220627/Momentum2TeacherF1.png)

作者首先对Synced BN和BN在自监督模型中的性能和成本进行了评估，如下，
![](/assets/img/20220627/Momentum2TeacherT1.png)

基于以上，可以得出4个结论，笔者认为这些结论也许也适用于其他模型和任务
1. Synced BN对于训练的帮助非常明显；
2. Synced BN降低了训练速度；
3. 不必同时在t s上都使用Synced BN；
4. 一个稳定的Teache

综上，本文的目的是获得stable statistics。常规方法，通过大batch，低效；本文方法，通过momentum。

![](/assets/img/20220627/Momentum2TeacherF2.png)

更新方法也很简单，如下

$$
\begin{equation}
\begin{aligned}
&\mu_{t}=(1-\alpha) \mu_{t}+\alpha \mu_{t-1} \\
&\sigma_{t}=(1-\alpha) \sigma_{t}+\alpha \sigma_{t-1}
\end{aligned}
\end{equation}
$$

同时，面对自监督中需要student teacher交换输入样本的情况，为了避免两者之间的信息泄漏，这里作者使用了lazy update方法。

即对两组loss
$$
\begin{equation}
\begin{aligned}
\mathcal{L}_{1} &=\| \operatorname{student}(v), \text { teacher }\left(v^{\prime}\right) \| \\
\mathcal{L}_{2} &=\| \operatorname{student}\left(v^{\prime}\right), \text { teacher }(v) \|
\end{aligned}
\end{equation}
$$

分别计算其更新的参数

$$
\begin{equation}
\begin{aligned}
&\left\{\mu_{t}^{1}, \sigma_{t}^{1}\right\}=(1-\alpha)\left\{\mu_{t}^{v^{\prime}}, \sigma_{t}^{v^{\prime}}\right\}+\alpha\left\{\mu_{t-1}, \sigma_{t-1}\right\} \\
&\left\{\mu_{t}^{2}, \sigma_{t}^{2}\right\}=(1-\alpha)\left\{\mu_{t}^{v}, \sigma_{t}^{v}\right\}+\alpha\left\{\mu_{t-1}, \sigma_{t-1}\right\}
\end{aligned}
\end{equation}
$$

在进行对应的统计变换即可
$$
\begin{equation}
\left\{\mu_{t}, \sigma_{t}\right\}=(1-\alpha)\left\{\frac{\mu_{t}^{v^{\prime}}+\mu_{t}^{v}}{2}, \frac{\sigma_{t}^{v^{\prime}}, \sigma_{t}^{v}}{2}\right\}+\alpha\left\{\mu_{t-1}, \sigma_{t-1}\right\}
\end{equation}
$$