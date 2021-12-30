---
layout: post
title: 'Server环境配置笔记'
date: 2021-05-19
author: Poley
cover: '../assets/img/mycat3.jpg'
tags: 环境配置
---

由于来了新的服务器需要进行配置，在此进行简单记录。

# 网络桥接
由于某些原因，服务器不好直接联网，因此使用现有的笔记本电脑来对服务器进行网络桥接，从而实现服务器联网配置环境的需要。

## 桥接主机：Window10 Laptop

1. 确保Windows机器可以正常联网
2. 打开Windows网络连接，如下
![](/assets/img/20210615/SENV1.png)

右键属性打开目前正在联网的适配器，在这里是无线网WLAN。在共享栏中选择电脑以太网相连的接口，这里是以太网2。配置如下

![](/assets/img/20210615/SENV2.png)

之后在设置中选择对应的端口即可。

## 桥接从机：Ubuntu Server

>参考博客：https://blog.csdn.net/sunqian666888/article/details/85093576

注：如果不用了，记得吧桥接关上，降低优先级。把要使用的网络优先级调高。

# 远程连接(SSH)

## 远程服务器：Ubuntu Server
```[shell]
sudo apt install openssh-server #安装shh服务端
dpkg -l | grep ssh #检查ssh服务端安装
ps -e | grep ssh #确认shh服务启动
ifconfig #获取本机IP信息，然后就可以通过ssh登陆本机了
```

## 远程客户端： Windows10 Laptop

直接使用Xshell设置对应的局域网IP和帐号密码即可，略。

# apt和conda换源

这部分直接复制清华镜像站的即可，参考

apt源
> https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

conda源
> https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

修改conda超时阈值
> https://blog.csdn.net/Arthur_Holmes/article/details/105095088

pip换源

> https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

仍然有些包需要使用默认源安装，可以使用
```
pip install [module] -i https://pypi.org/simple
```
```
conda config --set remote_connect_timeout_secs 40
conda config --set remote_read_timeout_secs 100
```
# 显卡驱动

## NVIDIA显示驱动
本次安装直接使用Ubuntu系统中**Software & update**中的附加驱动安装。

使用.run安装的时候需要问题，提示已经加载了模块'nvidia-drm'，无法安装。
故卸载此模块
```
sudo systemctl isolate multi-user.target

sudo modprobe -r nvidia-drm
```

即可

上面的方法安装完了还是有错，因此求助于强大apt，直接在线安装

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
ubuntu-drivers devices
sudo apt-get install nvidia-XX你需要的版本号XX
或安装全部驱动
sudo ubuntu-drivers autoinstall
重启
```
## CUDA & cudnn安装
> https://blog.csdn.net/ashome123/article/details/105822040
> https://blog.csdn.net/chch2010523/article/details/107929168
注：安装cuda11不用降级gcc。

# torch安装

个人用torch 1.8

```
 conda install pytorch=1.8 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

```

# 配置多CUDA版本

> https://blog.csdn.net/Maple2014/article/details/78574275




