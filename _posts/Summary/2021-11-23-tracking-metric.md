---
layout: post
title: '目标跟踪的性能评估标准'
date: 2021-11-23
author: Poley
cover: '/assets/img/mycat3.jpg'
tags: 总结
---

# 2D 目标跟踪

## 单目标跟踪
传统的评估tracker的方式是：在测试序列上运行一遍该跟踪算法（其中第一帧以ground truth作初始化），然后计算average precision或sucess rate。我们把这种只在测试序列上运行一遍的评估方法叫做one-pass evaluation (OPE)。然而tracker对初始化可能比较敏感，不同的起始帧可能对performance有很大的影响。因此，还需要对算法的鲁棒性进行评估，包括时间鲁棒性评估（TRE）和空间鲁棒性评估（SRE）。接下来详细介绍OPE和Robustness的评估指标

### ALE(Average Location Error)

+ 定义：平均定位误差，即预测框与真实框中心位置的欧式距离取帧平均
+ 用途：用来判断两个框的靠近程度

### AOR(Average Overlap Rate)

#### Overlap Rate threshold
+ 定义：平均重叠率，即预测的b-box与ground truth的交并比取帧平均
+ 范围：0~100%
+ 用途：用来判断两个框的重叠程度

#### Location Error threshold
+ 定义：需要人为设定的定位误差的阈值，Location Error低于该阈值的框被认为是命中目标，反之则被认为未命中
+ 用途：作为区分框是否命中目标的指标
+ 定义：需要人为设定的重叠率的阈值，重叠率高于该阈值的框被认为是命中目标，反之则被认为未命中
+ 范围：0～100%
+ 用途：作为区分框是否命中目标的指标

![](/assets/img/20211122/pp.png)
![](/assets/img/20211122/SP.png)

#### TRE（Temporal Robustness Evaluation）
定义：时间鲁棒性评估。从整个序列中截取若干段（可以重复），每段的初始帧利用ground truth进行初始化，在每一段上分别运行跟踪算法，对每一段分别进行评估，最后对总体信息进行统计。
#### SRE（Spatial Robustness Evaluation）
定义：空间鲁棒性评估。对起始帧的ground truth进行shift或scale操作形成若干段测试序列，在每一段上分别运行跟踪算法，对每一段分别进行评估，最后对总体信息进行统计

## 多目标跟踪
CLEAR MOT Metrics认为一个好的多目标跟踪器应该有如下三点特性：
1.所有出现的目标都要能够及时找到（检测的性能）
2.找到目标位置要尽可能可真实目标位置一致（检测的性能）
3.保持追踪一致性，避免跟踪目标的跳变 （匹配的性能）
所以可以看出，多目标跟踪和目标检测是密不可分的，检测的性能不可避免的会对跟踪的性能造成影响。

MOTChallenge的评价指标一共有十一个，分别是

Measure|Better|Perfect|Description
--|--|--|--|
MOTA|higher|100%|跟踪的准确度，和出现FN，FP，IDs的数量负相关，可能出现负值。
MOTP|higher|100%|跟踪的精度，GT和检测的bbox的匹配交叠
IDF1|higher|100%|引入track ID的F1
FAF|lower|0|每帧的平均误报警数
MT|higher|100%|命中的轨迹占总轨迹的占比，定义命中的轨迹为长度小于ground truth 80%的轨迹
ML|lower|0|丢失的轨迹占总轨迹的占比，定义丢失轨迹为长度小于ground truth 20%的轨迹
FP|lower|0|FP的总数量，false positives也就是误检
FN|lower|0|FN的总数量，false negatives也就是漏检
IDs|lower|0|ID改变的总数量
Frag|lower|0|轨迹被打断的总数量
Hz|higher|Inf|处理速度，不包括检测器的耗时，而且这个指标由作者提供，MOTChallenge是计算不出来的，因为递交的是offline文件。

### MOTA
$$
\mathrm{MOTA}=1-\frac{\sum_{t}\left(\mathrm{FN}_{t}+\mathrm{FP}_{t}+\mathrm{IDSW}_{t}\right)}{\sum_{t} \mathrm{GT}_{t}}
$$


其中，FN为False Negative，FP为False Positive，IDSW为ID Switch代表ID的切换。，GT为Ground Truth 物体的数量。MOTA考虑了tracking中所有帧中对象匹配错误，主要是FN，FP，ID Switch。MOTA给出了一个非常直观的衡量跟踪器在检测物体和保持轨迹时的性能，与物体位置的估计精度无关。MOTA取值应小于100，当跟踪器产生的错误超过了场景中的物体，MOTA会为负数。需要注意的是，此处的MOTA以及MOTP是计算所有帧的相关指标再进行平均（既加权平均值），而不是计算每帧的rate然后进行rate的平均。
注意MOTA中的FN，FP是检测的结果，而不是跟踪的结果，也就是说MOTA中只有IDs是和跟踪有关系的，剩下的都是检测。MOTA相比于IDF1要更偏向与检测。
### MOTP
$$
\mathrm{MOTP}=\frac{\sum_{t, i} d_{t, i}}{\sum_{t} c_{t}}
$$
其中，d为检测目标i和给它分配的ground truth之间在所有帧中的平均度量距离，在这里是使用bounding box的overlap rate来进行度量（在这里MOTP是越大越好，但对于使用欧氏距离进行度量的就是MOTP越小越好，这主要取决于度量距离d的定义方式）；而c为在当前帧匹配成功的数目。MOTP主要量化检测器的定位精度，几乎不包含与跟踪器实际性能相关的信息。
### IDF1 
$$
I D F_{1}=\frac{2 I D T P}{2 I D T P+I D F P+I D F N}
$$
- ID Precision $P=\frac{T P}{T P+F P}=\frac{T P}{C}$
- ID Recall $\quad R=\frac{T P}{T P+F N}=\frac{T P}{T}$
- $\mathrm{F}_{1}$-score $\quad F_{1}=2 \frac{P R}{P+R}=\frac{T P}{\frac{T+C}{2}}$

### AMOTA
和AP一样，使用多阈值MOTA计算曲线下积分得到
### AMOTP
和AP一样，使用多阈值MOTP计算曲线下积分得到