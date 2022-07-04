---
layout: post
title: 'Train in Germany, Test in The USA: Making 3D Object Detectors Generalize'
date: 2022-4-25
author: Poley
cover: '/assets/img/20220425/MLC.png'
tags: 论文阅读
---

本文和Train in Germany那篇文章得出了一样的结论，即domain shift的主要来源是几何统计的mismatch。但是和那篇文章不同，本方法并不需要使用目标域的几何先验，或者少量的标注信息。本文基于MeanTeacher模型以及分别在point, instance 和neural statistics level设置一致性损失来完成无监督的域适应学习。

和2D图像不同，由于3D物体的尺寸不随物体距离传感器的距离而发生变化，因此3D目标检测器在训练的过程中对于尺寸的预测往往会服从一个较窄的分布。这就造成了显著的domain gap。不同数据集预测的尺寸分布差异如下图所示。

传统的2D图像域适应并不是很适用于3D场景，因为其一般都针对光照、颜色、纹理的差异，而这些差异3D中都没有。

无监督域适应首先需要建立有意义的适应尺度的target来促进学习，这里使用的是Mean-Teacher架构，其中的Teacher相当于是学生网络的一个temporal ensenmble。

同时，本文使用三个level上的consistence，分别是point-level，instance-level和neural statistics-level。其大致意义分别是，point-level：对ts使用同样的点云输入，并通过索引获得点和点之间的对应关系（在数据增强后）；Instance-level：使用Transfering代替Matching来环节潜在的错误；Neural-level：Transfering学生的统计信息到Teacher。

其整体结构如下图所示
![](/assets/img/20220425/MLCF3.png)

+ point-level: 这里实际上是region proposal的consistance。由于点云的不规则性，其proposal的对应关系不像图像那样直接在feature map的对应位置进行（这里作者使用的是PointRCNN,从点产生proposal，而不是体素网格）。而直接匹配会不可避免的产生错误，因此这里直接通过点的对应关系（序号）来确定teacher和student proposal的对应关系。TS两者使用相同的点云输入已建立点之间的对应关系。在损失上，分类一致性使用DL散度，而BOX回归使用向量距离，如下
$$
\begin{equation}
L_{p t, c l s}=\frac{1}{\left|x_{t}\right|} \sum D_{K L}\left(\hat{R}_{t}^{c} \| R_{t}^{c}\right)
\end{equation}
$$
$$
\begin{equation}
L_{p t, b o x}=\frac{1}{\left|\mathbb{P}_{p o s}\right|} \sum_{p^{i} \in \mathbb{P}_{p o s}} d\left(\hat{R}_{t}^{c(i)}, h\left(R_{t}^{c(i)}\right)\right)
\end{equation}
$$

同时，和一般的目标检测损失也一样，这里只计算正样本的回归损失，其通过teacher和student各自的预测做NMS之后取交集得到。

+ Instance Level： 基本同上，只不过这里是对refinement结束之后的结果做上述的两个损失。

+ Neural Statistic level： 这里个人认为是本文的一个创新之一。作者认为对于这些不参与学习的参数（BN的参数不参与反向传播），其会导致网络数据的统计特性发生偏移。由于Student模型同时使用源域的标注数据和目标域的未标注数据进行训练，而Teacher只使用目标域的未标注数据进行训练，因此两者的Batch统计信息会产生差异。这里作者选择直接将student模型的BN参数给Teacher用，以保证两者统计特性的一致性。

## Experiment
本文的方法取得了较好的效果，如下图所示

![](/assets/img/20220425/MLCT1.png)

并作了若干的消融实验，以分别验证三个level consistence和EMA的作用。注意到，这里消融实验体现出，box回归的loss对于性能的提升远大于cls层面的，这说明SIZE的差异对于域差异有较大贡献，这也符合之前相关研究工作得出的结论。
![](/assets/img/20220425/MLCT2.png)

![](/assets/img/20220425/MLCT3.png)

![](/assets/img/20220425/MLCT4.png)


![](/assets/img/20220425/MLCT5.png)
