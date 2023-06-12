---
layout: post
title: 'Fully Sparse 3D Object Detection'
date: 2022-12-15
author: Poley
cover: '/assets/img/20221205/FS.jpg'
tags: 论文阅读  
---
目前3D检测器通常使用一个稀疏backbone提取特征，之后将其转换为dense并进行后续的密集预测。同样的，转换为dense map也存在问题，即浪费了体素的稀疏性，dense feature 在大range的时候吃不住。这个问题同样适用于多尺度。

dense map在预测时起到关键作用。主要是，如果没有dense卷积，物体中心的特征是空的！一般是通过卷积的扩散效应，将物体表面的voxel特征扩散到物体中心。如下图所示，物体边缘的特征通过dense卷积逐步扩散到中心。

![](/assets/img/20221205/FSF1.jpg)

自动驾驶场景一般需要大场景，空间分辨率不变的情况下，增大检测半径，feature map的大小平方增长。但是，实际上稀疏性则是进一步提高了。因此作者提出Fully Sparse，充分利用网络的稀疏性，避免产生过大的dense map，使得网络在检测范围上更加灵活。

本文提出的方法整体结构如下
![](/assets/img/20221205/FSF3.jpg)

本方法其实很类似于point-based的方法，只不过是使用体素网络来作为主干，再将特征分配到各个点上。进行后续的基于点的处理。

如上图所示，首先通过体素提取特征。之后，每个点的特征通过其对应的体素+点和体素中心的偏移构成。类似于VoteNet，进行center的投票。

这里后续的目的主要是通过点特征来构建instance-level feature。因此需要先group出对应的instance。这里提出通过center之间的距离来判断，通过阈值来判断center的连通性，并通过dfs高效的完成所有点的遍历。实现group。得到group之后，是本文的重点，即如何提取instance-level feature。

整体流程如下所示
![](/assets/img/20221205/FSF4.jpg)

总体来说，这里通过通过d ynamic 广播/pooling来实现group特征和point之间的快速转换。这要求group之间没有重叠，而以instance为group自然的满足这个要求。

基本的point-based模块分为三个部分：
1.group center。这里用同组的centroid来表示

2.Pair-wise feature,这里将group的特征和点特征cat得到

3.Group的特征通过dynamic pooling实现。

具体方法上，以当前组内相对位置做编码，得到新的特征。再pooling group内的特征，cat到点上二次更新特征。相当于是做了pooling再cast到所有点上。
$$
\begin{equation}
\begin{aligned}
F_l^{\prime} & =\operatorname{LinNormAct}\left(\operatorname{CAT}\left(F_l, X-p_{\mathrm{avg}}\left(X^{\prime}, I\right)[I]\right)\right) \\
F_{l+1} & =\operatorname{LinNormAct}\left(\operatorname{CAT}\left(F_l^{\prime}, p_{\max }\left(F_l^{\prime}, I\right)[I]\right)\right)
\end{aligned}
\end{equation}
$$

这样的模块称为SIR，可以多个叠加，类似于decoder。将每一层获得的group feature收集起来作为instance-level的特征并给出预测框。由于这里是通过聚类产生的group，并且group之间不存在重叠，因此理论上这里的目标检测也是end-to-end的，比较简洁。

同时，为了避免一定会存在的分组做错误，这里相当于引入了一个二阶段，用两个SIR模型。二阶段的SIR根据第一个SIR模块得到的proposal来重新划分分组。

网络的具体性能如下
![](/assets/img/20221205/FST1.jpg)

![](/assets/img/20221205/FST2.jpg)