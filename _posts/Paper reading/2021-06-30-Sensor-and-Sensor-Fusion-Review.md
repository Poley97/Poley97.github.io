---
layout: post
title: 'Sensor and Sensor Fusion Technology in Autonomous Vehicles: A Review'
date: 2021-06-30
author: Poley
cover: '/assets/img/20210630/SSF.png'
tags: 论文阅读
---

> 论文链接: https://www.mdpi.com/1424-8220/21/6/2140

未来几十年，自动驾驶汽车市场预计将经历显著增长，并彻底改变交通和移动性的未来。自动驾驶汽车（Autonomous Vehicles，AV）是一种能够感知环境并在很少或没有人为干预的情况下安全有效地执行驾驶任务的车辆，预计最终将取代传统车辆。自动驾驶车辆使用各种传感器来感知周围环境，并依靠5G通信技术的进步来实现这一目标。传感器是感知周围环境的基础，近年来，与AVs相关的传感器技术发展迅猛。尽管取得了显著的进步，但由于硬件缺陷、噪音和环境条件等原因，传感器仍可能无法按要求运行。因此，任何自主驾驶任务都不希望依赖单个传感器。在最近的研究中显示的实用方法是结合多个互补的传感器，以克服独立操作的单个传感器的缺点。本文综述了适用于自主车辆的传感器的技术性能和能力，主要集中在视觉摄像机、激光雷达和雷达传感器。审查还考虑了传感器与各种软件系统的兼容性，使多传感器融合方法能够用于障碍物检测。这篇综述文章最后强调了一些挑战和未来可能的研究方向。

# Introduction
根据世界卫生组织(世卫组织)发布的《全球现状报告》，2018年报告的年度道路交通死亡人数达到135万人，使其成为世界上所有年龄人群中第八大非自然死亡原因[1]。就欧盟而言，虽然报告的年度道路死亡人数有所减少，但每年仍有40，000多人死亡，其中90%是由人为失误造成的。因此，为了改善交通流量，全球投资者投入巨资支持自动驾驶汽车的发展。此外，预计自动车辆(AVs)将有助于降低碳排放水平，从而有助于实现碳减排目标[2]。自主或无人驾驶车辆提供了传统车辆的运输能力，但在很大程度上能够感知环境并在最少或没有人为干预的情况下自动导航。根据优先研究公司发布的一份报告，2019年全球自动驾驶汽车市场规模达到约6500辆，预计在2020年至2027年期间将经历63.5%的复合年增长率[3]。2009年，谷歌秘密启动了自动驾驶汽车项目，目前名为Waymo(目前是谷歌母公司Alphabet的子公司)。2014年，Waymo展示了一款没有踏板和方向盘的100%自主汽车原型[4]。迄今为止，Waymo已经实现了一个重要的里程碑,其自动驾驶车辆已经在美国25个城市的公共道路上总共行驶了2000多万英里[5]。在爱尔兰的背景下，2020年，捷豹路虎(JLR)宣布与爱尔兰香农的自动汽车中心合作，并将使用450公里的道路来测试其下一代自动驾驶车辆技术[6]。
2014年，SAE国际(以前称为汽车工程师协会(SAE))为消费者推出了J3016“驾驶自动化水平”标准。J3016标准定义了六个不同级别的驾驶自动化，从SAE 0级(驾驶员完全控制车辆)到SAE 5级(车辆能够在没有人工干预的情况下控制动态驾驶任务的所有方面)。图1描述了这些水平，在高度自动化车辆(HAV)的安全设计、开发、测试和部署中，这些水平经常被行业引用和参考[7]。目前，特斯拉和奥迪(大众)等汽车制造商在开发其自动化功能时采用了SAE级自动化标准，即特斯拉的Autopilot[8]和奥迪A8的Traffic Jam Pilot[9，10]。另一方面，Alphabet的Waymo自2016年以来一直在评估一种基于4级自动驾驶出租车服务的商业模式，这种服务可以在美国亚利桑那州的有限区域内产生收入[11]。

![](/assets/img/20210630/SSFF1.png)

虽然各种自动驾驶车辆系统可能略有不同，但自动驾驶系统是一个非常复杂的系统，由许多子组件组成。在[12]中，从技术角度介绍了自动驾驶系统的体系结构，该体系结构结合了自动驾驶系统的硬件和软件组件，并从功能角度介绍了自动驾驶系统从数据收集到车辆控制所需的处理块。从技术角度来看，硬件和软件是两个主要层，每一层都包括代表整个系统不同方面的各种子组件。一些子组件作为其层内的主干，用于硬件和软件层之间的通信。从功能角度来看，自动驾驶车辆由四个主要功能块组成:感知、规划和决策、运动和车辆控制以及系统监控。这些功能块是根据处理阶段和到通过数据采集得到的用来控制车辆的信息流定义的。AV架构的技术和功能方面的描述如图2所示。AV架构的详细讨论超出了本文的范围(请参见[12]了解更详细的概述)。

![](/assets/img/20210630/SSFF2.png)

采用多种传感器的AV的传感能力是整个AV系统的一个基本要素；这些传感器的配合和性能直接决定了驾驶员的生存能力和安全性[13]。选择合适的传感器阵列及其最佳配置是任何自动数据采集系统的主要考虑因素之一，这些传感器阵列及其最佳配置本质上将用于模拟人类感知和描绘可靠环境图像的能力。从整体上考虑所选传感器组(即智能传感器和非智能传感器)的优点、缺点和局限性总是至关重要的。“智能传感器”的定义在过去几十年中随着物联网(IoT)的出现而演变，物联网是一种由相互关联的互联网连接对象(设备)组成的系统，可以在无线网络上收集和传输数据，而无需人工干预。在物联网环境中，**智能传感器是一种无需单独的计算机就能调节输入信号、处理和解释数据以及做出决策的设备[14]**。此外，在AV环境中，用于环境感知的距离传感器，例如摄像机、激光雷达和雷达，当传感器提供目标跟踪、**事件描述和其他信息作为其输出的一部分时，可以被认为是智能的”**。相比之下，“非智能”传感器是一种仅调节传感器原始数据或波形并传输数据进行远程处理的设备。它需要外部计算资源来处理和解释数据，以便提供关于环境的附加信息。最终，只有当计算机资源是物理传感器设计不可分割的一部分时，传感器才被认为是“智能的”。总的来说，AV系统的整体性能通过多种不同类型的传感器(智能/非智能)和模式(视觉、红外和无线电波)在不同的范围和带宽(数据速率)下运行，并结合每种传感器的数据产生融合输出而大大提高[16，17，18]。现在，多传感器融合是所有自动驾驶系统的必备过程，从而克服了单个传感器类型的缺点，提高了整个自动驾驶系统的效率和可靠性。

目前，在提高无人驾驶车辆模块的准确性、鲁棒性、性能和可靠性方面投入了大量努力，尤其是与网络安全和安全性相关的操作问题，这些问题在实际驾驶条件下至关重要[16]。CAV，也称为**互联自动车辆（Connected & Automated Vehicles）**，是一种变革性技术，在减少道路事故、提高运输系统效率和提高生活质量方面具有巨大潜力。连接的技术允许自动车辆和周围基础设施之间的相互作用，例如，CAV中的车载设备从路边单元(RSU)接收数据，并向驾驶员显示适当的警报消息。然而，CAV，与其他计算平台一样，容易受到网络攻击，并可能导致致命的碰撞事故[19]。

这篇综述文章将重点介绍用于环境感知的AV传感器模式，以及用于目标检测的当前发展和现有传感器融合技术的概述。主要目的是全面审查与自动车辆相关的传感器的技术性能和能力，以及感知子系统中用于物体检测的多传感器融合方法的当前发展。**第2节概述了自动车辆系统中使用的现有传感模式，主要侧重于视觉摄像机、激光雷达和雷达传感器，包括它们在不同环境条件下的优缺点和局限性。第3节讨论了AVs中传感器校准的必要性，概述了现有的校准包，该包解决了任何校准系统所需的主要方面，随后是用于物体检测的传感器融合方法的当前发展及其对安全可靠的环境感知的挑战。第4节总结回顾和展望，并对未来的研究提出建议。**

# Sensor Technology in Autonomous Vehicles

传感器是将检测到的事件或周围环境的变化映射到定量测量以供进一步处理的设备。一般来说，传感器根据其工作原理分为两类。本体感受传感器或内部状态传感器捕捉动态状态并测量动态系统的内部值，例如力、角速度、车轮负载、电池电压等。**本体感受传感器的例子包括惯性测量单元、编码器、惯性传感器(陀螺仪和磁力计)和定位传感器(全球导航卫星系统接收器)**。相反，**外部感知传感器或外部状态传感器感测并获取系统周围的信息，如距离测量或光强度。照相机、无线电探测和测距(雷达)、光探测和测距(激光雷达)以及超声波传感器是外部感受传感器的例子**。此外，传感器可以是被动传感器或主动传感器。无源传感器接收从周围环境发射的能量以产生输出，例如视觉摄像机。相反，主动传感器将能量发射到环境中，并测量环境对该能量的“反应”以产生输出，例如激光雷达和雷达传感器[20-22]。**在自主车辆中，传感器对于车辆周围环境的感知和定位以及路径规划和决策至关重要，是控制车辆运动的必要前提。AV主要利用多个视觉摄像头、雷达传感器、激光雷达传感器和超声波传感器来感知其环境。**此外，其他传感器，**包括全球导航卫星系统(GNSS)、惯性测量单元和车辆里程计传感器**用于确定车辆的相对和绝对位置[23]。AV的相对定位是指车辆参照其相对于周围地标的坐标，而绝对定位是指车辆参照其相对于全球参考系的位置[24]。典型反病毒应用中环境感知传感器的位置、覆盖范围和应用如图3所示。读者将会意识到，在行驶的车辆中，车辆周围有更全面的覆盖。多个传感器的单独和相对定位对于精确和准确的物体检测至关重要，因此可以可靠和安全地执行任何后续操作[25]。总的来说，在AV中从一个独立的来源生成足够的信息是有挑战性的。本节回顾了三种主要传感器的优点和缺点:摄像机、激光雷达和雷达，用于视听应用中的环境感知。

![](/assets/img/20210630/SSFF3.png)

## Camera

照相机是感知周围环境最常用的技术之一。照相机的工作原理是通过镜头(安装在传感器前面)检测光敏表面(图像平面)周围发出的光，以产生周围的清晰图像[23，26]。相机相对便宜，配有适当的软件，能够检测其视野内的移动和静止障碍物，并提供高分辨率的周围图像。这些能力允许车辆的感知系统在道路交通车辆的情况下识别道路标志、交通灯、道路车道标记和障碍物，在越野车辆的情况下识别大量其他物品。**AV中的摄像机系统可以采用单目摄像机或双目摄像机，或者两者的组合**。顾名思义，单目摄像机系统利用单个摄像机来创建一系列图像。传统的RGB单目相机从根本上比双目相机更受限制，因为它们缺乏原始深度信息，尽管在一些应用或采用双像素自动对焦硬件的更高级的单目相机中，深度信息可以通过使用复杂的算法来计算[27-29]。因此，两个摄像头通常并排安装，形成一个双目摄像机系统。双目相机，也称为立体相机，模仿动物对深度的感知，利用每只眼睛中形成的略有不同的图像之间的“差异”(潜意识)来提供深度感。立体摄像机包含两个图像传感器，由一条基线隔开。术语基线是指两个图像传感器之间的距离(通常在立体摄像机的规格中引用)，它根据摄像机的特定型号而变化。例如，在[30]中评论的用于自主智能车辆(AIV)的Orbbec 3D相机，对于Persee和Astra系列相机[31]都具有75毫米的基线。在动物视觉的情况下，根据立体摄像机图像计算的视差图允许使用极线几何和三角测量方法生成深度图(视差计算算法的详细讨论超出了本文的范围)。参考文献[32]使用机器人操作系统(ROS)中的“立体图像处理”模块，这是一个开源的机器人元操作系统[33]，在实现SLAM(同时定位和映射)和自主导航之前执行立体视觉处理。表1显示了不同制造商的双目摄像机的一般规格。

![](/assets/img/20210630/SSFT1.png)

AVs中其他常用的感知周围环境的摄像机包括鱼眼镜头摄像机[46-48]。鱼眼相机通常用于近场传感应用，如停车和交通堵塞辅助，只需要四个相机就可以提供360度的环境视图。参考文献[46]提出了一种鱼眼环绕视图系统和卷积神经网络(CNN)架构，用于在自主驾驶环境中分割运动对象，以每秒15帧的速度运行，精度为40% Overage over Union(IoU，粗略地说，是一种计算目标遮罩(地面真实)和预测遮罩之间重叠面积的评估指标)和69.5%的平均IOu。在所有实际的照相机上，数字图像是通过光穿过安装在传感器平面前面的照相机镜头形成的，照相机镜头将光线聚焦并导向传感器平面，以形成周围的清晰图像。**镜头几何形状与理想/标称几何形状的偏差会导致图像失真，因此在极端情况下，物理场景中的直线可能会在图像中变成轻微的曲线。这种空间失真可能在图像中检测到的障碍物或特征的估计位置中引入误差。因此，摄像机通常是“固有校准的”。所有相机的内在校准是至关重要的，以便纠正相机镜头造成的任何失真，否则会对深度感知测量的准确性产生不利影响[49]**。我们在第3.1.1节中详细讨论了摄像机内部校准和常用方法。此外，众所周知，摄像机捕获的图像质量(分辨率)可能会受到光照和不利天气条件的显著影响。照相机的其他缺点可能包括在分析图像数据时需要大的计算能力[26]。综上所述，相机是一种无处不在的技术，它提供高分辨率的视频和图像，包括感知环境的颜色和纹理信息。**AVs上摄像机数据的常见用途包括交通标志识别、交通灯识别和车道标志检测。**由于相机的性能和高保真图像的创建高度依赖于环境条件和照明，图像数据通常与其他传感器数据(如雷达和激光雷达数据)融合，以便在AD中生成可靠和准确的环境感知。

## LiDAR

光探测和测距(LiDAR)最早建立于20世纪60年代，广泛用于航空和航天地形测绘。在20世纪90年代中期，激光扫描仪制造商生产并交付了第一台商业激光雷达，用于地形测绘应用，脉冲数为每秒2000至25000次[50]。在过去的几十年里，激光雷达技术的发展一直在以显著的速度不断发展，**激光雷达目前是高级驾驶员辅助系统(ADAS)和AD车辆的核心感知技术之一**。激光雷达是一种遥感技术，其工作原理是发射反射目标物体的红外光束或激光脉冲。这些反射由仪器检测，光脉冲发射和接收之间的时间间隔能够估计距离。**当激光雷达扫描其周围环境时，它会以点云的形式生成场景的3D表示[26]**。与自主机器人、无人机、仿人机器人和自主车辆相关的研究和商业企业的快速增长对激光雷达传感器提出了很高的要求，因为它具有测量范围和精度、对周围变化的鲁棒性和高扫描速度(或刷新率)等性能属性。例如，目前使用的典型仪器每秒钟可记录200，000个点或更多，覆盖360°旋转和30°垂直视场。因此，近年来出现了许多激光雷达传感器公司，并一直在引入新技术来满足这些需求。**因此，汽车激光雷达市场的收入预计到2025年将达到69.1亿美元[51]**。**AVs中使用的当前最先进的激光雷达传感器的波长通常为905纳米**，这是最安全的激光器类型(1类)，其吸水率低于以前使用的1550纳米波长传感器[52]。参考文献[53]中的一项研究发现，在雾和雨等恶劣天气条件下，905纳米系统能够提供更高分辨率的点云。然而，905纳米激光雷达系统对雾和降水仍然部分敏感:最近的一项研究[54]中报告说，**像雾和雪这样的恶劣天气条件可能会使传感器的性能下降25%**。激光雷达传感器的三个主要变体可以应用于广泛的应用，包括1D、2D和3D激光雷达。激光雷达传感器将数据输出为一系列点，也称为1D、2D和3D空间中的点云数据以及对象的强度信息。对于3D激光雷达传感器，PCD包含场景或环境中障碍物的x、y、z坐标和强度信息。对于自动数据采集应用，通常采用**64或128通道**的激光雷达传感器来生成高分辨率的激光图像(或点云数据)[55，56]。

+ 1D或一维传感器仅测量周围物体的距离信息(x坐标)；
+ 2D或二维传感器提供关于目标物体角度(y坐标)的附加信息；
+ 3D或三维传感器沿垂直轴发射激光束，以测量周围物体的高度(z坐标)。

激光雷达传感器可以进一步分为机械激光雷达或固态激光雷达。**机械激光雷达是AV研发领域最受欢迎的远程环境扫描解决方案**。它使用由电机驱动的高级光学器件和旋转透镜来引导激光束，并捕捉自主车辆周围的期望视野(FoV)。旋转镜头可以实现覆盖车辆周围的360°水平FoV。相比之下，**固态硬盘消除了旋转镜头的使用，从而避免了机械故障**。SSL利用多种微结构波导引导激光束感知周围环境。近年来，这些激光雷达作为旋转激光雷达的替代产品越来越受到人们的关注，因为它们坚固、可靠，并且通常比机械激光雷达的成本低。**然而，与传统的机械激光雷达相比，它们的水平视场更小且有限**，通常为120°或更小[23，57]。参考文献[58]比较和分析了目前市场上各种制造商提供的12种旋转激光雷达传感器。在[58]中，不同的模型和激光配置在三种不同的场景和环境中进行评估，包括动态交通、天气模拟室内产生的恶劣天气和静态目标。结果表明，Ouster OS116激光雷达模型在反射目标上的平均点数最低，**旋转激光雷达的性能受强光照和恶劣天气的强烈影响，特别是在降雨量高、不均匀或大雾的地方**。表2显示了在研究[58]中测试的每个激光雷达传感器的一般规格(综合设备规格也在[59]中给出)。此外，我们用其他激光雷达扩展了[58，59]研究中总结的一般规范，包括来自Cepton、SICK和IBEO的Hokuyo 210旋转激光雷达和固态激光雷达，以及我们最初发现的用于数据采集的常用ROS传感器驱动器。激光回波是当激光脉冲被目标拦截和反射时记录的离散观测值。激光雷达可以从同一个激光脉冲中收集多个回波，现代传感器可以记录每个激光脉冲的多达五个回波。例如，Velodyne VLP-32C激光雷达分析多次回波，并根据激光回波模式配置报告最强回波、最后一次回波或双重回波。在单激光返回模式(最强返回或最后返回)下，传感器分析从一个方向的激光束接收的光，以确定距离和强度信息；随后使用该信息来确定最后的回报或最强的回报。相反，双返回模式下的传感器将返回最强和最后一次返回测量值。然而，如果最强的返回测量值与最后的返回测量值相似，则第二强的测量值将作为最强的返回。更不用说强度不够的点会被忽略[60]。

一般来说，**目前，3D旋转激光雷达由于其更宽的视野、更远的探测范围和深度感知，更常用于自动驾驶车辆，以提供可靠和精确的白天和夜晚感知**。以点云格式采集的数据提供了AVs周围环境的密集3D空间表示(或“激光图像”)。**与相机系统相比，激光雷达传感器不提供周围环境的颜色信息，这是点云数据通常使用传感器融合算法与来自不同传感器的数据融合的一个原因。**

![](/assets/img/20210630/SSFT2.png)

## Radar
无线电探测和测距(Radio Detection and Ranging)或雷达(Radar)在第二次世界大战之前首次建立，其工作原理是在感兴趣的区域内辐射电磁波，并接收目标的散射波(或反射波)，以进行进一步的信号处理并建立关于目标的距离信息。它利用电磁波的多普勒特性来确定探测到的障碍物的相对速度和相对位置[23]。多普勒效应，也称为多普勒频移，是指波源和目标之间的相对运动引起的波频率的变化或偏移。例如，当目标向雷达系统的方向移动时，接收信号的频率增加(短波)。雷达多普勒频移的一般数学方程可以表示为[90，91]:

$$
\begin{equation}
f_{D}=\frac{2 \times V_{r} \times f}{C}=\frac{2 \times V_{r}}{\lambda}
\end{equation}
$$

其中$f_D$是多普勒频率，单位为赫兹(Hz)；㼿㼿是目标的相对速度；f是发射信号的频率；c是光速(3×108米/秒)，λ是发射能量的波长。实际上，雷达中的多普勒频率变化发生两次；首先，当电磁波发射到目标时，其次，在多普勒频移能量反射到雷达(源)期间。**市场上可用的商用雷达目前工作在24 GHz(千兆赫)、60 GHz、77 GHz和79 GHz频率**。与79 GHz雷达传感器相比，24 GHz雷达传感器的距离、速度和角度分辨率更有限，导致在识别和应对多种危害方面存在问题，预计将在未来逐步淘汰[23]。电磁波(雷达)的传播不受不利天气条件的影响，雷达功能不受环境照明条件的影响；因此，它们可以在雾天、下雪天或阴天的白天或晚上工作。**雷达传感器的缺点之一是错误地检测周围感知的金属物体，如路标或护栏，以及区分静态、静止物体的挑战[92]。**例如，由于多普勒频移的相似性，动物尸体(静态物体)和道路之间的差异可能会给雷达带来挑战。在[94]中演示的设置中使用79 GHz汽车雷达传感器(SmartMiro[94])的当前研究中的初步发现显示，**感兴趣区域中的假阳性检测频率很高**。图4示出了在距离安装的传感器大约5-7米的距离处物体的假阳性检测的例子。

![](/assets/img/20210630/SSFF4.png)

AV车辆中的雷达传感器通常以不可见的方式集成在几个位置，例如挡风玻璃顶部附近的车顶上，车辆保险杠或品牌标志后面。**确保生产中雷达的安装位置和方向的精度至关重要，因为任何角度偏差都可能对车辆的运行产生致命后果，这种误差包括对周围障碍物的错误或延迟检测[95，96]**。**中程雷达(MRR)、远程雷达(LRR)和短程雷达(SRR)是汽车雷达系统的三个主要类别。反车辆制造商利用SRR进行包装辅助和碰撞临近警告，利用MRR进行侧/后碰撞避免系统和盲点检测，利用LRR进行自适应巡航控制和早期检测应用[23]。**我们回顾了来自不同制造商的几种雷达传感器的一般规格，如SmartMicro、Continental和Aptiv Delphi，表3给出了概述。

![](/assets/img/20210630/SSFT3.png)

一般来说，雷达传感器是自主系统中众所周知的传感器之一，并且通常用于自主车辆中，以在白天和晚上提供对障碍物的可靠和精确的感知，这是因为其能够在不受照明条件和不利天气条件影响的情况下工作。它提供额外的信息，如检测到的移动对象的速度，并可以根据配置模式在短、中或长距离范围内执行映射。**然而，雷达传感器通常不适合物体识别应用，因为与照相机相比，它们的分辨率较低。因此，AV研究人员经常将雷达信息与其他感官数据(如相机和激光雷达)融合，以弥补雷达传感器的局限性。**

# Sensor Calibration and Sensor Fusion for Object Detection

根据美国Lyft的自动驾驶部门Lyft Level 5[113]的一篇文章，传感器校准是自主系统开发中讨论最少的话题之一。**它是一个自治系统及其组成传感器的基础模块，是实现传感器融合深度学习算法之前必不可少的预处理步骤**。传感器校准通过比较雷达、摄像机和激光雷达检测到的**已知特征的相对位置，向系统通知传感器在真实坐标中的位置和方向**。通常，这是通过采用三个传感器之一的固有坐标系来完成的。精确的校准对于进一步的处理步骤至关重要，包括传感器融合和实现用于对象检测、定位和映射以及控制的深度学习算法。术语“对象检测”定义了从环境的图像或点云表示中的大量预定义类别(图像分类)中定位对象实例(对象定位)的存在的过程[114]。**不良的校准结果可能导致谚语所说的“垃圾-(数据)-输入和垃圾-(结果)-输出”**；导致对探测到的障碍物位置的错误或不准确的估计，并可能导致致命的事故。**校准又分为内部校准、外部校准和时间校准**。**内部校准处理特定于传感器的内部参数**，它首先执行，然后再执行外部校准和物体检测算法。一方面，**外部校准确定传感器相对于外部参考系的位置和方向(相对于3D空间的所有三个正交轴的旋转和平移)**。另一方面，**时间校准指的是具有潜在不同频率和延迟的各种传感器数据流的同步性[115]**。第3.1节回顾了三类校准，并概述了当前研究中使用的现有校准包。传感器融合是自主车辆的基本任务之一。与单独使用传感器相比，算法融合了从多个传感器获取的信息，以减少不确定性。传感器融合有助于建立一个一致的模型，可以在各种环境条件下准确感知周围环境[116]。传感器融合的使用提高了检测周围障碍物的精度和可信度。此外，它降低了复杂性和组件的总数，从而降低了整体系统成本[117]。传感器融合算法主要用于反车辆总体结构的感知模块，包括目标检测子过程。参考文献[118]提出了用于AV车辆感知任务的多传感器数据融合(MSDF)框架，如图5所示。MSDF框架由一个传感器校准过程和几个物体检测处理链(基于传感器数量)组成，校准过程涉及估计校准参数。MSDF过程融合了校准参数和目标检测信息，用于进一步的处理任务，如跟踪、规划和决策。第3.2节回顾了三种传感器方法，即用于目标检测的高级融合(HLF)、低级融合(LLF)和中级融合(MLF)，并总结了常用的算法，接下来是用于安全可靠的环境感知的传感器融合的挑战。

![](/assets/img/20210630/SSFF5.png)

## Sensor Calibration

### Intrinsic Calibration Overview

**固有校准是确定传感器固有参数或内部参数的过程，用于校正系统或确定性像差或误差**。这些参数是特定于传感器的，例如相机的焦距或失真系数，并且一旦估计了固有参数，这些参数就应该是恒定的。通过个人交流得知，Velodyne激光雷达被校准到国家标准与技术研究所(NIST)目标的10%反射率(缓解)。因此，低于10%反射率的障碍物反射率可能无法被激光雷达探测到[119]。传感器固有校准的算法和方法在过去几年中受到了相当大的关注，并取得了显著的进步，现在在文献中已经得到了很好的确立。这些算法和方法可能因传感器而异[120-127]。本小节旨在概述针孔摄像机模型最常用的校准目标和校准方法。针孔照相机模型是计算机视觉应用中众所周知且常用的模型(受最简单的照相机的启发[128])，其描述了3D空间中的点在2D图像平面上的投影的数学关系[129]。图6显示了摄像机针孔模型，该模型由一个封闭的盒子组成，盒子正面有一个小开口(针孔)，来自目标的光线通过该小开口进入并在相对的摄像机壁(图像平面)上产生图像[130]。从数学角度来看(图7)，该模型包括一个3D相机坐标系和一个2D图像坐标系，以使用透视变换方法校准相机[132，133]。校准过程包括利用外部参数(3×4矩阵，由旋转和平移[R | t]变换组成)将世界坐标空间(XW、YW、ZW)中的3D点变换成它们相应的3D摄像机坐标(XC、YC、ZC)。此外，它包括使用固有参数(也称为3×3固有矩阵，K [134])，将3D相机坐标转换成2D图像坐标(x，y)。

![](/assets/img/20210630/SSFF6.png)

![](/assets/img/20210630/SSFF7.png)

透视变换方法输出4x3相机矩阵(P)，也称为投影矩阵，其包括将3D世界坐标空间变换成2D图像坐标的内在和外在参数。应当强调的是，相机校准环境中的外部校准参数不同于一个或多个传感器相对于另一个传感器的外部校准过程。众所周知，相机矩阵不考虑任何镜头失真——缺少镜头的理想针孔相机。透视法的一般数学方程表示为[123，132，135，136]:

$$
\begin{equation}
P=K[\mathrm{R} \mid \mathrm{t}] \text { or } P=\left[\begin{array}{ccc}
f_{x} & s & c_{x} \\
0 & f_{y} & c_{y} \\
0 & 0 & 1
\end{array}\right]\left[\begin{array}{cccc}
r_{11} & r_{12} & r_{13} & t_{1} \\
r_{21} & r_{22} & r_{23} & t_{2} \\
r_{31} & r_{32} & r_{33} & t_{3}
\end{array}\right]\left[\begin{array}{c}
X_{w} \\
Y_{w} \\
Z_{w} \\
1
\end{array}\right]
\end{equation}
$$
其中P是4x3相机矩阵；[R|t]表示将3D世界点(XW、YW、ZW)转换为相机坐标的外部参数(旋转和平移)；K是针孔摄像机的固有矩阵，由摄像机的几何特性组成，如轴偏斜、光学中心或主点偏移和焦距。相机的焦距(f)是指针孔和像平面之间的距离，它决定了图像的投影比例。因此，较小的焦距将导致较小的图像和较大的视角[130]。3D世界点投影到2D图像平面的详细讨论、相机镜头失真的估计和实现超出了本文的范围(更全面的概述见[130，131])。摄像机校准(或摄像机重新分割[135])是确定构成摄像机矩阵的内在和外在参数的过程。摄像机标定是计算机视觉和摄影测量中的核心问题之一，在过去的几年里受到了广泛的关注。多种校准技术[122-124，131，137-140]仅举几个例子，已经被开发出来以适应各种应用，例如自动车辆导航系统、无人驾驶水面飞行器(USV)或水下3D重建。参考文献[139]将这些技术分为:

+ 摄影测量校准这种方法使用从校准对象(通常是平面图案)观察到的已知校准点，其中3D世界空间中的几何形状是高精度已知的。
+ 自校准。这种方法利用从静态场景中的运动摄像机捕获的图像之间的对应关系来估计摄像机的内部和外部参数。

著名的Zhang方法是最常用的照相机校准技术之一。它使用摄影测量校准和自校准技术的组合来估计相机矩阵。它使用从多个方向(至少两个)的平面图案(图8)观察到的已知校准点以及不同位置的校准点之间的对应关系来估计摄像机矩阵。此外，当照相机或平面图案相对于彼此移动时，用于照相机校准的zhang方法不需要运动信息[139]。

![](/assets/img/20210630/SSFF8.png)

ROS中流行的开源“摄像机校准”包提供了几个预实现的脚本(通过命令行执行)，以使用平面图案作为校准目标来校准单目、立体和鱼眼摄像机。校准结果包括失真图像的固有矩阵、失真参数、校正矩阵(仅限立体摄像机)、摄像机矩阵或投影矩阵，以及其他操作参数，如宁滨和感兴趣区域。标定包是基于OpenCV摄像机标定和三维重建模块开发的；标定算法是基于著名的Zhang方法和Bouguet, J.Y.[126，132]的MATLAB摄像机标定工具箱实现的。一般来说，如果相机的变焦(焦距)发生变化，相机校准结果将不再适用。应该注意的是，根据我们的经验，雷达和激光雷达传感器是工厂内部校准的。

# Extrinsic Calibration Overview

**外部校准是将点从一个3D坐标系映射到另一个坐标系的刚性变换(或欧几里德变换)**，例如，点从3D世界或3D激光雷达坐标系到3D相机坐标系的刚性变换。外部校准估计传感器相对于3D空间正交轴(也称为6自由度，6DoF)相对于外部参考系的位置和方向[141]。校准过程输出由传感器的旋转(R)和平移(t)信息组成的外部参数，通常以3×4矩阵表示，如等式2所示。本节旨在提供现有开源多传感器外部校准工具的比较概述，以及文献中针对摄像机、激光雷达和包含传感器融合系统的雷达传感器的外部校准提出的算法总结。外部校准和方法学的研究在文献中已经建立，例如参见参考文献[141-149]。然而，在多传感器系统中，具有不同物理测量原理的多个传感器的外部校准会带来挑战。**例如，匹配相机图像(以像素为单位的密集数据)和3D激光雷达或雷达点云(没有颜色信息的稀疏深度数据)之间的对应特征通常是具有挑战性的[142]**。基于目标的非本征校准方法采用特别设计的校准目标，例如无标记平面图案[45]，棋盘图案[143]，正交和三面反射器[45，141，144-145]，圆形图案来校准自主系统中的多个传感器模态。无目标外部校准方法利用单个传感器的估计运动，或者利用感知环境中的特征来校准传感器。然而，使用感知的环境特征需要多模态传感器来提取环境中的相同特征，并且对校准环境敏感[142，146]。[143]中对现有外部校准工具的比较概述报告称，可用工具仅解决最多两种传感模式的成对校准。例如，[141]中提出的框架使用从粗到细的外部校准方法，用威力登激光雷达校准RGB相机。该算法利用一种新的具有四个圆孔的3D标记来估计粗略校准参数，并使用密集搜索方法进一步细化这些参数，以在小的6自由度校准参数子空间中估计更精确的校准。参考文献[147]提出了一种非本征校准算法，该算法利用平面点到平面和平面边缘到反向投影平面的几何约束，使用无标记平面校准目标来估计3D激光雷达和立体摄像机的非本征参数。如前一段所强调的，每个感测模态具有不同的物理测量原理；因此，具有更多模态的传感器设置可能会重复校准工作，特别是在移动机器人中，在移动机器人中传感器经常被拆卸或重新定位。为此，参考文献[143]和[145]提出了一种新的校准方法，利用专门设计的校准目标从外部校准所有三种传感模式，即雷达、激光雷达和摄像机。下面的表4总结了开源的外部传感器校准工具，特别是针对相机、激光雷达传感器和雷达传感器的外部校准。


**[143]中对现有外部校准工具的比较概述报告称，可用工具仅解决最多两种传感模式的成对校准。**例如，[141]中提出的框架使用从粗到细的外部校准方法，用Velodyne激光雷达校准RGB相机。该算法利用一种新的具有四个圆孔的3D标记来估计粗略校准参数，并使用密集搜索方法进一步细化这些参数，以在小的6自由度校准参数子空间中估计更精确的校准。参考文献[147]提出了一种非本征校准算法，该算法利用平面点到平面和平面边缘到反向投影平面的几何约束，使用无标记平面校准目标来估计3D激光雷达和立体摄像机的非本征参数。**如前一段所强调的，每个感测模态具有不同的物理测量原理；因此，具有更多模态的传感器设置可能会重复校准工作，特别是在移动机器人中，在移动机器人中传感器经常被拆卸或重新定位**。为此，参考文献[143]和[145]提出了一种新的校准方法，利用专门设计的校准目标从外部校准所有三种传感模式，即雷达、激光雷达和摄像机。下面的表4总结了开源的外部传感器校准工具，特别是针对相机、激光雷达传感器和雷达传感器的外部校准。

![](/assets/img/20210630/SSFT4.png)

参考文献[143]提出了一种新颖的外部校准工具，该工具利用基于目标的校准方法和联合外部校准方法来促进三种感测模态的外部校准。建议的校准目标设计由位于大矩形板中心的四个圆形锥形孔和位于板后部四个圆圈之间的金属三面角反射器组成(图9)。角反射器提供强雷达反射，因为聚苯乙烯泡沫塑料板对电磁辐射基本透明。此外，圆形边缘为激光雷达(尤其是与较少的激光雷达光束相交时)和相机提供了精确和鲁棒的检测。该系统的作者为联合外部校准建立了三种可能的优化配置，即:
  + 姿态和结构估计(PSE)。它估计真实板位置的潜在变量，并使用估计的潜在变量优化到所有校准目标姿态的精确估计的变换。
  + 最小连接姿态估计(MCPE)。它依赖于一个参考传感器，并将多传感模式转换估计为一个单一的参考框架。
  + 全连接姿态估计(FCPE)。它“联合”估计所有感测模态之间的转换，并实施循环闭合约束以确保一致性。

![](/assets/img/20210630/SSFF9.png)


所提出的校准工具[150]与常用的ROS中间件绑定，并提供上述联合优化配置，以根据多个位置的同时校准板检测来估计传感器姿态。它输出一个变换矩阵(P)，可用于将检测从源参考帧变换到目标参考帧，以及传感器相对于父链接的姿态，以便可视化(在ROS中)。他们比较了基于多个变量的PSE、MCPE和FCPE联合优化结果，如所需的校准板位置数量和MCPE参考传感器选择。结果表明，当使用五个以上的板位置时，FCPE联合优化提供了比MCPE和PSE更好的性能。每个联合优化配置及其算法的详细讨论以及校准板的几何形状超出了本文的范围(更全面的概述请参见[143]和[150])。目前的作者利用和审查了参考文献[143]中的校准工具，在初始多传感器设置中对Velodyne VLP-32C激光雷达传感器、SmartMiro UMRR-96T-153雷达传感器和FalconIQ EZIP-T030(E)互联网协议(IP)工业变焦单目相机进行了外部校准[94]。这项工作产生的意见和建议包括:

+ 确保圆圈的边缘与背景有足够的颜色对比，特别是在室外校准相机时，这在我们的情况下是必要的。然而，在[143]中建议传感器的校准应在室内进行，以避免强风吹翻校准板。
+ 确保相机镜头不受雨滴影响，以便在室外校准传感器时降低噪音，尤其是在雨天和大风天气条件下。
+ 根据所采用的活性氧传感器驱动程序，可能需要额外的或修改的脚本来匹配板检测器节点的活性氧传感器消息类型。例如，在[143]中使用了大陆ARS430雷达传感器，并利用了自动调节缓冲提供的ROS消息，该消息以自动调节缓冲传感器消息阵列格式输出检测结果[97]。然而，智能微雷达的ROS驱动程序以点云2格式的ROS传感器消息类型输出检测结果[108]。表5总结了外部校准工具的每个板检测器节点的传感器消息类型(作为输入要求)。
+ 确保在激光雷达点云中用足够的点检测(覆盖)四个圆的边缘。我们检查并比较了velodyne VLP-32C和velodyne HDL-64E(在[143]中使用)的仰角。结果表明，**HDL-64E的垂直激光点在-24.9度至2度之间均匀分布。相比之下，Velodyne VLP-32C的垂直激光点集中在-25度到15度之间的光学中心的中间**，如图10所示。因此，激光雷达相对于校准板的位置和方向可能对报告的激光雷达数据内检测到的圆的位置有显著影响。
+ [143]中建议将校准板放置在一个宽敞的区域，并在所有传感器的FoV中捕捉至少十个校准板位置。但是，不建议握住校准板，这可能会影响角反射器(由雷达传感器)的检测。
+ [143]中使用的立体摄像机由两个单目摄像机构成；即IDS成像UI-3060CP第2版；并利用ROS [33]中的“立体图像处理”模块来创建感知环境的视差图像。
![](/assets/img/20210630/SSFT5.png)
![](/assets/img/20210630/SSFF10.png)

基于对研究团体可用的外部校准工具的这一修订，注意到它们中的大多数仅解决两个感测模态的成对校准，显著的例外是[143]中描述的外部校准工具，该工具促进了两个以上感测模态(雷达、照相机和激光雷达)的联合外部校准，并与ROS中间件绑定。其他开源外部校准工具包括Kalibr，它提供多摄像机校准或摄像机-IMU外部校准和Calirad，便于雷达、摄像机和LiDAR传感器的外部校准和时间校准。再次强调，在实施外部校准之前，单独的传感器是内部校准的。与基于目标的外部校准方法相比，无目标外部校准方法估计传感器的运动或感知环境中的特征，例如道路标记，以确定传感器的外部校准。

### Temporal Calibration Overview 

时间校准是在多传感器设置中估计来自传感器的数据的同步性(或相对时间延迟)的过程。同样重要的是要强调，传感器通常会在不同的频率下工作。例如，相机通常以30帧/秒或更低的速度捕捉图像，而激光雷达传感器可以以低至5 Hz的速度扫描。同步传感器的一种方法是在端点(计算机)接收的消息时间戳之间建立最接近的匹配。然而，原则上，基于消息时间戳的同步是次优的，因为传感器可能具有未知的延迟，例如传感器电路中的通信传输延迟或预处理延迟[118]。这些未知的延迟可能无法直接确定，并且可能因传感器而异。ROS消息过滤器模块[162]中的近似时间同步器方法基于来自每个感测模态(或ROS术语中的主题)的报头时间戳来匹配消息，作为使用自适应算法的时间同步的手段。自适应算法首先确定特定主题队列头中的最新消息作为参考点，并基于估计的参考点在给定阈值内近似同步这些消息。我们利用[162]中的近似时间同步器方法来同步初始多传感器设置中的传感器数据[94]。结果显示，平均86.6%的传感器信息以不同的操作频率在50毫秒的阈值内同步。此外，相机和激光雷达之间最长的非同步周期为850毫秒；在激光雷达和雷达之间，是870毫秒；相机和雷达之间的时间是880毫秒。另一种基于ROS中消息头时间戳的同步方法是精确时间同步器[162]，它要求传入的消息具有精确的时间戳进行同步。近似时间同步器方法中采用的自适应算法的全面概述以及这些方法的使用超出了本文的范围(更详细的概述见[162])。

时间校准通常被忽略，在多传感器融合应用中至关重要，例如自动驾驶车辆必须实时执行复杂的传感和估计任务，例如状态估计和障碍物检测[118]。有两种方法可以暂时校准传感器:外部同步，利用外部硬件进行时间同步和内部同步，利用每个传感器测量值上的附加时间戳进行同步[163，164]。外部同步方法使用中央硬件时钟作为外部时间源或参考时钟来对传感器进行时间同步，并且与实时标准(如世界协调时标准时间)精确相关。例如，参考文献[165]利用外部诺华智能6-L全球定位系统(GPS)作为参考时钟，并利用GPS时间戳信息来同步系统(或计算机)时钟。相反，内部同步方法在没有外部时间源的情况下，基于相关时间戳同步传感器，以获得所有传感器网络的一致时间视图。参考文献[166]提出了被动同步算法，以确定设备和传感器时钟漂移时的时间偏移，并可以显著减少同步误差，即使存在未知的延迟和具有显著时钟误差的传感器。完整的传感器到传感器校准，也称为时空校准，包括传感器到统一坐标空间的外部校准和时间校准，以估计传感器数据流之间的相对时间延迟。参考文献[167]提出了一种时空校准方法，该方法使用来自高斯过程(GPs)的估计连续时间运动物体轨迹和基于目标的方法来相对于彼此校准传感器。它利用估计的物体速度来估计传感器之间的相对时间延迟。这些[167]实验表明，所提出的算法能够可靠地确定时间延迟，最大延迟为最快传感器采样速率的一小部分。在[167]中提出的方法的实现在[168]中是开源的，并且它与ROS中间件绑定。此外，一旦使用的多传感器能够确定移动“目标”的3D位置，它适用于任何多传感器设置。对所采用的遗传算法的深刻讨论超出了范围(更全面的概述见[167-169])。此外，通过个人通信，目标检测在6米或更远的距离变得不稳定，这取决于跟踪器的大小。构成校准跟踪器的材料包括(图11) [170]:
+ 制造三角形平面图案的聚苯乙烯泡沫塑料或纸板；
+ 长度约为17厘米的印刷AprilTag标记，位于三角形平面的前面；
+ 组装三面角反射器的纸板，其中反射器的三个内侧覆盖有铝箔，并附着在三角形平面的后部。

![](/assets/img/20210630/SSFF11.png)

其他时空校准方法包括采用基于目标的方法和目标测量(位置)的时空关系来估计时间延迟和传感器外部参数[171]。在[172]中，使用多同步总线(外部硬件)向所有计算机发布基于IEEE 1588精确时间协议(PTP)的同步时间戳，作为数据采集过程中的时间同步手段。总之，估计以不同频率工作的多个传感器之间的时间延迟是至关重要的，特别是在时间关键的自主系统中，以便实时精确地执行自主任务，例如物体检测、车辆状态估计以及最终防止碰撞。

传感器融合是自动驾驶汽车和无人驾驶地面车辆(UGV)等大多数自主系统的一个重要方面。传感器融合提供了对周围环境的准确、鲁棒和可靠的感知。它在多传感器设置中集成了从多种传感模式(如相机、雷达、激光雷达)获取的数据，以减少检测不确定性的数量，并克服单个传感器独立运行的缺点。例如，摄像机和雷达融合可以提供高分辨率图像和感知环境中检测到的障碍物的相对速度。下表6根据其技术特点和其他外部因素，如天气和照明条件，定性总结了自动车辆中常用的外部感受传感器的优缺点。

![](/assets/img/20210630/SSFT6.png)

在文献[16，23，92，165，173-175]中，对用于环境感知和目标检测的自动车辆中的多传感器融合系统的研究已经建立。目前，文献中主要有三种用于障碍物检测的传感器组合，包括相机-激光雷达；摄像雷达；和照相机-激光雷达-雷达传感器组合。[92]进行的一项调查显示，在环境感知的多传感器融合系统中，最常用的是认知无线电传感器组合，其次是认知无线电和认知无线电。铬传感器组合提供高分辨率图像，同时获得周围障碍物的额外距离和速度信息。例如，特斯拉利用铬传感器组合和其他传感器，如超声波传感器，来感知车辆周围的环境[8]。同样，CLR传感器组合可以提供更大范围的分辨率，并通过激光雷达点云和深度图信息精确了解周围环境。它还提高了整个自治系统的安全冗余度。例如，Waymo和Navya [176]在他们的AVs中使用了CLR传感器组合来感知环境。

### Sensor Fusion Approaches 

在MSDF框架中，有三种主要方法来组合来自各种传感模式的传感数据:**高级融合(HLF)、低级融合(LLF)和中级融合(MLF)** [177]。在HLF方法中，每个传感器独立地执行目标检测或跟踪算法，随后执行融合。例如，参考文献[23]利用HLF方法融合处理后的数据，即雷达信号和激光雷达点云，随后采用非线性卡尔曼滤波方法检测障碍物和状态跟踪。由于相对复杂性低于LLF和MLF，经常采用HLF办法。

然而，HLF信息不充分，因为如果有几个重叠的障碍，置信度较低的分类将被丢弃。**相反，使用LLF方法，来自每个传感器的数据在最低抽象层(原始数据)被集成(或融合)**。因此，所有的信息都被保留，并且可以潜在地提高障碍物检测精度。参考文献[178]提出了一种两阶段3D障碍物检测架构，称为3D交叉视图融合(3D-CVF)。在第二阶段，他们利用LLF方法，使用基于3D感兴趣区域(RoI)的汇集方法，将从第一阶段获得的联合相机-激光雷达特征图与低层相机和激光雷达特征融合。他们在KITTI和nuScenes数据集上评估了所提出的方法，并报告说对象检测结果优于KITTI排行榜中最先进的3D对象检测器；(更全面的总结见参考文献[178])。在实践中，LLF方法面临诸多挑战，尤其是在实施方面。**它需要对传感器进行精确的外部校准，以准确融合它们对环境的感知。传感器还必须平衡自我运动(环境中系统的3D运动)，并进行时间校准[177]**。MLF，也称为特征级融合，是LLF和高层框架之间的一个抽象层。它融合从相应传感器数据(原始测量)中提取的多目标特征，例如来自图像的颜色信息或雷达和激光雷达的位置特征，并随后对融合的多传感器特征执行识别和分类。参考文献[179]提出了一种特征级传感器融合框架，用于在通信能力有限的动态背景环境中检测目标。他们利用符号动态滤波(SDF)算法从多个不同方向的红外传感器中提取低维特征，并且在存在变化的环境光强度的情况下；随后，将提取的特征作为聚类与用于运动目标检测的凝聚层次聚类算法融合。**然而，由于其对环境的有限感知和上下文信息的丢失，多层框架似乎不足以实现安全等级4或5的自动驾驶系统[180]**。

### Sensor Fusion Techniques and Algorithms

传感器融合技术和算法在过去几年中已经被广泛研究，并且现在在文献中已经被很好地建立。然而，最近的一项研究[181，182]表明，由于文献中提出的融合算法的多学科和变体，获得当前最先进的融合技术和算法是一项艰巨而具有挑战性的任务。[16]的研究将这些技术和算法分为经典传感器融合算法和深度学习传感器融合算法。一方面，经典的传感器融合算法，例如基于知识的方法、统计方法、概率方法等，利用来自数据缺陷的不确定性理论，即不准确性和不确定性来融合传感器数据。参考文献[183]介绍了一种道路环境中的实时环形交叉口检测和导航系统，该系统结合了用于检测障碍物的“激光模拟器”算法和用于决策的经典基于知识的模糊逻辑(FL)算法。

另一方面，深度学习传感器融合算法涉及生成各种多层网络，使其能够处理原始数据和提取特征，以执行具有挑战性和智能的任务，例如在城市环境中进行AV目标检测。在视听环境中，算法，如卷积神经网络(CNN)和递归神经网络(RNN)是感知系统中最常用的算法。参考文献[184]提出了一种先进的加权平均只看一次(YOLO) CNN算法来融合RGB相机和LiDAR点云数据，以提高对象检测的实时性能。[185]于2016年首次创建了YOLO探测器，并在过去几年中实现了一个重要的里程碑。它是一个单阶段检测器，可以预测包围盒，并在单个神经网络(仅一次评估)中的图像上产生带有置信度得分的类概率。基于YOLO的模型在挥发性有机化合物2007数据集上提供了45 FPS的快速检测速度和59.2%的平均精度(AP，一种测量对象检测或信息检索模型性能的评估指标)[185]。此外，由[186]于2020年4月发布的最新YOLOv4在NVIDIA特斯拉V1V上以大约65 FPS的MS COCO数据集和43.5%的AP(和65.7%的AP50 - IoU高于50%)获得了真实时间速度的最新结果此外，随着3D传感器和用于理解周围AV的3D环境的多种应用的出现，3D对象检测的研究重点增加。**参考文献[187]利用了他们先前在[188]中提出的VoxelNet框架，并提出了两种称为点融合和体素融合的特征级融合方法，以结合用于3D对象检测的RGB和点云数据。**根据[188]，VoxelNet是一种通用的3D对象检测网络，它将特征提取和包围盒预测过程统一到一个单阶段、端到端可训练的深度网络中。点融合方法使用已知的校准矩阵将3D点投影到图像上，随后从预先训练的2D CNN中提取图像特征，并在点级别连接它们。随后，他们利用VoxelNet架构来联合处理连接的要素和相应的点。相比之下，体素融合方法将VoxelNet创建的非空3D体素投影到图像上，并提取2D感兴趣区域内的特征，从而在体素级别连接合并的图像特征。参考文献[189]提出了一种点融合框架，该框架利用图像数据和原始点云数据进行3D对象检测。他们利用CNN和PointNet [190]架构独立处理图像和点云，然后组合结果输出来预测多个3D盒假设及其相应的置信度。PointNet架构是一种新颖的神经网络，它为从3D分类到场景语义解析的应用程序提供了统一的架构，以处理原始点云数据。其他基于深度学习的传感器融合算法，举几个例子，包括:
+ ResNet，或残差网络，是一个残差学习框架，便于深度网络训练[195]。
+ SSD或Single-Shot Multibox Detector是一种将边界框离散为一组框的方法，每个feature map位置具有不同的大小和纵横比，以检测不同大小的对象[196]–它克服了YOLO小尺寸和不同比例对象检测精度的限制。
+ CenterNet[197]代表了最先进的单目摄像机3D对象检测算法，该算法利用关键点估计来找到边界框的中心点，并将中心点回归到所有其他对象属性，包括大小、3D位置、方向和姿态。有兴趣详细讨论传感器融合技术和算法的读者可参考[16，26，181，187-199]，其应用范围从环境感知，包括2D或3D物体检测和车道跟踪，到定位和制图。

### Challenges of Sensor Fusion for Safe and Reliable Environment Perception

毫无疑问，在广泛研究的基础上，多传感器融合技术在从仿人机器人到AVs的自主系统中取得了相对全面的优势。这些系统通常配备有传感器阵列，每小时可以生成大量数据。例如，一个AV软件每小时可以生成大约383 GB到5.17 TB(TB)的数据[200]。因此，处理这些数据需要很大的计算能力。参考文献[201]回顾了一家领先的自动驾驶公司的L4 AV计算平台的实现，并研究了几种现有的AD处理解决方案。此外，他们提出并原型化了一个安全、模块化、动态、节能和高性能的广告计算架构和软件堆栈。他们的原型系统平均消耗11瓦的功率，并且可以使用ARM移动片上系统(SoC)以每小时8公里的速度驾驶移动车辆。从软件的角度来看，**将强化学习技术与监督学习算法相结合有助于降低计算能力**、训练数据要求和训练时间。

强化学习是一种机器学习方法，它使用来自他们的行动和经验的反馈，在交互式环境中训练反向学习模型。相比之下，监督学习算法利用标记数据来训练最大似然模型(参考文献[202]获得更详细的概述)。然而，它对来自所有可能场景的数据进行训练和注释具有挑战性，包括但不限于AV在现实世界中可能遇到的位置、地形和天气。尽管数据的协作和共享可能有利于自治系统的发展，但由于担心削弱竞争优势，研究自治系统的公司不愿意共享资源[203]。此外，目标检测、定位和映射的最大似然/最小似然算法的性能受所用数据集质量的影响；因此，糟糕的数据质量可能导致众所周知的“垃圾-(数据)-输入和垃圾-(数据)-输出”。Roboflow 的创始人和首席技术官写道，在Udacity Dataset 2的15000个样本(或图像)中，33%没有注释，注释的边界框(或感兴趣的对象)过大[204]。
由于DL算法的不透明性，在多传感器AV中使用的DL模型的功能安全性也是一个挑战。参考文献[202]强调，在道路上部署DL模型之前，**进一步研究可用的安全验证方法和神经网络的可解释性至关重要**。此外，利用DL架构的自治系统容易受到攻击。攻击者在典型的图像上覆盖了敌对的样本(或扰动的图像)，**这些样本代表了数据链系统输入的细微变化，但导致了对具有高置信度的对象的错误分类[203]**。其他传感器融合挑战包括收集数据集的偏差、训练数据集的过拟合、数据测量的不精确性和不确定性，例如与校准误差、量化误差、精度损失、缺失值等相关的噪声。将多传感器数据转换成标准参考系也可能对传感器融合实施提出挑战。**从环境角度来看，传感器融合在可靠和安全感知方面仍然面临的挑战之一是视觉传感器在恶劣天气条件下的性能，如雪、雾、沙尘暴或暴雨**。这种情况会影响视觉传感器的视觉和距离测量，导致可见距离减小，并导致错误和误导性的输出。在**最坏的情况下，传感器可能会出现部分或完全的传感器故障，这对自动车辆及其周围环境可能是灾难性的。**因此，根据学习到的经验和历史数据，重要的是在过程的早期评估故障风险，并使驾驶员能够中断或完全脱离自主系统[16]。一般来说，质量数据是安全可靠的环境感知的关键。DL/ML模型使用这些数据来了解环境的特征并执行对象检测。因此，在实现DL/ML算法之前，对数据进行清理和预处理是至关重要的。然而，DL算法容易受到恶意攻击，这在安全关键系统(如AVs)中可能是灾难性的。自主系统的进一步研究和广泛测试对于评估所有可能的解决方案以防止恶意攻击和评估所有可能的传感器和系统故障风险至关重要；传感器情况下的替代解决方案

# Conclusions and Future Research Recommendations
 AV的领域很广，涵盖广泛的技术学科和技术，从电子、传感器和硬件到控制和决策算法，以及经济、法律和社会方面。传感器是感知环境、定位和绘图以及车辆状态控制的基础。目前，AV主要包括多个互补传感器，如惯性测量单元、雷达、激光雷达和摄像机，以克服单个传感器独立工作的缺点。本次审查调查和评估了各制造商传感器的技术性能和能力，主要侧重于视觉传感器，包括立体摄像机、机械和固态激光雷达以及雷达。**在执行数据处理算法之前，校准传感器至关重要**。精确的传感器校准使反车辆能够理解其在现实世界坐标中的位置和方向。我们回顾了**传感器校准的三个主要类别**，每个类别都是必要的:即**内部校准、外部校准、时间校准**和相关算法。此外，我们还提供了几个已成功用于近期研究的现有开源校准包的比较概述。很明显，大多数现有的用于外部和时间校准的开源校准工具仅解决最多两个感测模态的成对校准。近年来，传感器标定方法主要集中在离线标定方法上。传感器校准的离线方法利用专门设计的校准目标来提供精确的校准结果，但它不灵活。例如，如果传感器之间的几何形状发生变化，车辆需要重新校准。此外，外部因素，如温度和振动，可能会影响校准精度，因为多传感器通常在工厂校准。因此，**进一步研究在线和离线校准技术以自动检测和改进校准参数来提供自主操作中物体存在和位置的精确估计是至关重要的。自动驾驶车辆中可靠且高效的障碍物检测的发展对于实现自主驾驶至关重要**。在最近的研究中，用于安全和可靠的障碍物检测的实用方法是组合来自多模态传感器的信息，例如距离信息、速度、颜色分布等，以便提供准确、鲁棒和可靠的检测结果。我们回顾了**传感器融合的三种主要方法:高级融合、中级融合和低级融合**；随后回顾了最近提出的用于障碍物检测的多传感器融合技术和算法。同样，我们强调了多传感器融合在可靠和安全的环境感知方面的几个挑战。主要挑战是:环境条件、对数据链模型中恶意攻击的抗毁性、低质量数据集或不能解决反病毒所有可能环境的数据集，以及实时处理大量数据集的计算成本。因此，公司和研究人员必须评估失败的风险，并为驱动程序实施替代解决方案，以处理最坏的情况。在所有可能的情况下，包括恶劣的天气条件下，**进一步改进目标检测性能对于提供安全可靠的场景感知至关重要**。这对于开发精确、鲁棒和可靠的目标检测算法来区分环境中的障碍物至关重要。**提供更可靠和准确的障碍物检测的一种方法是通过深度学习方法或深度强化学习方法来增强现有的传感器融合算法[210]。另一种方法是投资传感器硬件技术，以提供更高的环境分辨率[16]**。