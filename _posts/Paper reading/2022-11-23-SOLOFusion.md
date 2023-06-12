---
layout: post
title: 'TIME WILL TELL: NEW OUTLOOKS AND A BASELINE FOR TEMPORAL MULTI-VIEW 3D OBJECT DETECTION'
date: 2022-10-27
author: Poley
cover: '/assets/img/20221123/SOLO.jpg'
tags: 论文阅读
---

本文是此时的nuscenes上camera模态的sota，并且具有明显的性能优势。其提出从3D多视图重建的角度解决camera的3D检测问题，很有新意并有很好的效果。

作者认为，现有方法阻碍时序立体图像性能的关键：1、粗粒度的匹配2、由于有限的历史信息利用导致的次优的多视角设置。结果显示，对于不同的像素和深度，最优的时间差，差异巨大。这使得网络必须融合long term history。从3D重建的角度来说，长时间的History对于更好的深度估计非常关键，这也是camera-only workss的主要瓶颈。


目前常见的时序融合方法：
+ BEV-based的方法通过concatenate多个timesteps的volumes实现；
+ query-based方法通过将query投影到过去的图像上实现提取特征。

但是这些方法都只能从temporal fusion中获得有限的增益。

不同时序融合方式的对比。分别是基于多视图匹配、lss和query。如下表所示
![](/assets/img/20221123/SOLOT1.jpg)

作者从多视图重建的角度来看待这个问题，因为多视图立体匹配和时序3D检测的共同点在于，其都关心一个候选位置是否被占据，前者关心其是否被任何物体占据，而后者关心其是否被特定物体占据。

本文首先提出了一个概念： Localization potential，如下图所示
![](/assets/img/20221123/SOLOF1.jpg)


个人理解：在reference view里的深度估计误差，投影在图像中的误差随着时间的增大而增大（基线变长），因此更容易通过图像点之间的匹配实现对深度的准确估计。引入long term实现对深度的准确估计可以弥补相机在3D检测中的主要劣势，从而提升性能。

其详细定义如下

如下，一个视图中的点定义为$(x,y,d)$，共有a,b两个视图。则a中的点投影在b中需要的内外参表示如下

$$
\begin{equation}
K=\left[\begin{array}{ccc}
f & 0 & c_x \\
0 & f & c_y \\
0 & 0 & 1
\end{array}\right], K^{-1}=\left[\begin{array}{ccc}
1 / f & 0 & -c_x / f \\
0 & 1 / f & -c_y / f \\
0 & 0 & 1
\end{array}\right], R t_{A \rightarrow B}=\left[\begin{array}{ccc|c}
\cos \theta & 0 & -\sin \theta & t_x \\
0 & 1 & 0 & 0 \\
\sin \theta & 0 & \cos \theta & t_z
\end{array}\right]
\end{equation}
$$

其中，坐标系定义如下图所示

![](/assets/img/20221123/SOLOF3.jpg)

这里假设相机只在水平面内移动。那么投影后的坐标可以表示为

$$
\begin{equation}
\left[x_b, y_b\right]=\left[\frac{d_a x_a^{\prime} \cos \theta-d_a f \sin \theta+t_x f}{\frac{d_a x_a^{\prime} \sin \theta}{f}+d_a \cos \theta+t_z}+c_x, \frac{d_a y_a^{\prime}}{\frac{d_a x_a^{\prime} \sin \theta}{f}+d_a \cos \theta+t_z}+c_y\right]
\end{equation}
$$

Localization potential则表示物体坐标相对于深度d的变化率，如下
$$
\begin{equation}
\text { Localization Potential }=\left|\frac{\partial x_b}{\partial d_a}\right|=\frac{f \bar{t} \cos (\alpha)|\sin (\alpha-(\theta+\beta))|}{\left(d_a \cos (\alpha-\theta)+t_z \cos (\alpha)\right)^2}
\end{equation}
$$

将其和时间t结合并重写，得到

$$
\begin{equation}
\left|\frac{\partial x_b}{\partial d_a}\right|=\frac{f \bar{t} t \cos (\alpha)|\sin (\alpha-(\theta+\beta))|}{\left(d_a \cos (\alpha-\theta)+t_z t \cos (\alpha)\right)^2}
\end{equation}
$$

即在reference view中某一点的距离变化在source view中引起的像素位置变动率。这里的分析指针对x，因为y的分析相当于x中t,theta=0的特殊情况。

传统的室内场景方法会选择具有最小仿射变换的帧，而室外方法一般经验性的选取某个历史帧。但是本文发现，（为了实现最好的深度估计）的最优source和reference之间的旋转和平移，根据不同的像素、深度、相机和自车运动具有显著的变化。

基于上述Localization potential可以得到，对于不同的坐标位置和预设深度，其获得最佳的Localization potential所需要的Time difference是截然不同的，如下图所示

![](/assets/img/20221123/SOLOF4.jpg)

通过引入时序，获得的Localization potential增益如下所示（值越大，深度估计越准确）

![](/assets/img/20221123/SOLOF5.jpg)

注意到，低分辨会（特征提取过程中的降采样）降低potential（由定义可知，其和分辨率成正比），但是由于计算限制，这是不可避免的。但是通过引入时序可以弥补这一点。获得增益大部分都超过了降采样的scale。上述Localization potential最小要达到1才能对depth产生有效的纠正和预测，不同距离的物体产生1的Localization potential所需要的时间如下所示

![](/assets/img/20221123/SOLOF6.jpg)

基于上述理论和实验分析，本文提出如下模型

![](/assets/img/20221123/SOLOF7.jpg)

基本思路是，将长序列分为若干短序列。在短序列中使用重建方法构建cost volume生成depth map并投影为bev feauture。短序列之间通过简单的concat形成长序列特征（cost volume）。这里的长序列特征直接通过concatbev特征获得，相对简单。而短序列的MVS重建则引入了强化学习相关的知识，笔者暂时还没太看懂。

在最终的性能上，本文方法实现了非常显著的性能提升

![](/assets/img/20221123/SOLOT2T3.jpg)