---
layout: post
title: 'Waymo Open Dataset 中 Range Image 的存储方式及其转换'
date: 2021-11-04
author: Poley
cover: '/assets/img/mycat3.jpg'
tags: 总结
---

Range Image是最接近激光雷达2.5D成像视角的数据存储格式，相比其他形式，其具有regular, dense的特点，可以直接使用2D卷积进行处理。Range Image相当于点云从3D空间中像激光雷达所在位置做球面投影，相对复杂一些。Waymo Open Dataset使用Range Image作为点云数据的存储方式。笔者在这里记录一下其数据的具体意义以及和普通点云的转换方式，这部分内容均来自于 waymo open dataset toolkit的源码阅读。

# 坐标系

Waymo数据一共有四个坐标系，分别是：
+  Global frame : 全局坐标系，东北上坐标系，可以用于多帧之间的对齐。数据集中的frame pose和pixel pose都是针对世界坐标系的。似乎只有Top LiDAR才进行这种世界坐标系的转换；
+  Vehicle frame: 车辆的坐标系，也就是所有传感器统一转换的坐标系；vehicle pose是一个$4\times 4$的位姿矩阵，从车辆坐标系到全局坐标系。
+  Sensor Frame: 车上各个传感器相对于车辆坐标系的位姿矩阵。同样是一个$4\times 4$的位姿矩阵。
+  Image frame: 图像平面，略
+  LiDAR 球坐标系： 用于生成Range Image的球坐标系。
# 数据存储

Waymo中的Range Images是存在一起的，一共5个激光雷达，一个主激光雷达+4个盲区激光雷达。存储方式均为MatrixFloat，序列式的存储。读取之后首先要将其reshape成range image 应有的大小。对于waymo数据集使用的激光雷达来说，这个大小应该是$64 \times 2650$。

除了range image本身，为了将其投影到3D空间中，相应的姿态和投影矩阵是很重要的。因此还需要两个投影信息，分别是 camera_projections 以及 range_image_top_pose。后者同样是一个$64 \times 2650$大小的高维张量，每个位置存储range image 每个像素相对于top lidar的pose。


range_image_top_pose 的维度大小是$64 \times 2650\times 6$，其中前三个维度分别表示roll, pitch, yaw的值，即沿x, y, z的旋转角度。后三个维度代表平移translation。将三维姿态转换成姿态矩阵，之后和translation合并，可以得到$4\times 4$的齐次位姿矩阵。

range_image的存储方式是：
```
{laser_name,
         [range_image_first_return, range_image_second_return]}
```

其中每个range_image一共4个维度。分别是$[range, intensity, unknow*2]$

在转换的时候，一般使用单回波模式，即忽略二次回波。在转换过程中，另一个非常必要的信息就是**beam_inclination**，即激光雷达垂直分布的线束角度信息，如果没有这个信息，可以选择通过制定范围来获得均匀分布的线束角度。但是问题在于，激光雷达的线束基本都不是均匀分布的。对于多激光雷达，还需要知道激光雷达之间的旋转和位移情况，方便进行转换坐标和投影的转换。

## 数据转换
+ 确定vehicle range_image各个像素中的位姿信息；
+ 读取某一个激光雷达 range_image(每个激光雷达的range_image是不一样大的，取决于雷达的线束数和水平角分辨率以及可视角度)；
+ 读取其线束角度分布，以及和主激光雷达(range image)的位姿关系；
+ 由于range image中不全都是有效的点，可能有没有收到回波的角度。因此需要读取一个range image mask来过滤无用的点（range<0表示没用的点）。
+ 对于一个range_image，首先计算其极坐标。先计算其每个像素点对应的方位角,这个方位角是对于vehicle range image的，由于lidar的朝向不一定和vehicle坐标系对齐，因此需要；之后结合雷达自身的Inclination以及雷达的测距range就可以到达一组完整的极坐标。
+ 将极坐标投影到笛卡尔坐标系，得到每个点的坐标，之后在每个点坐标的基础上再使用每个点的Pose_rotation和pose_translation来得到最终的点坐标。**注意，这里转换到笛卡尔坐标系之后逐点修正的旋转和平移是对世界坐标系的**之后再从世界坐标系转换回vehicle坐标系。
+ 由此，得到最后输出的点云。