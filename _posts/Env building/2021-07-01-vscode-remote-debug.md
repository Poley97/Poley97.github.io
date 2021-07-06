---
layout: post
title: 'vscode远程调试环境配置笔记'
date: 2021-07-01
author: Poley
cover: '../assets/img/mycat3.jpg'
tags: 环境配置
---

由于老板大气给配了服务器，为了方便使用，配置一下远程调试环境。

虽然我一直用的Python编辑器是pycharm，但是pycharm的远程调试要钱，所以选择了vscode。

## 远程调试配置

1. **Remote-SSH** 如下所示
![](/assets/img/20210701/vs1.png)

安装之后左边会新增一个远程资源管理器，在里面添加自己的主机，命令如下
```
shh 用户名@host
```
之后按照它的提示输入密码等，提示连接成功即可

2. **Python调试**

通过远程资源管理器打开服务器上的一个项目，之后选择打开要调试的文件。左边栏中有一个运行与调试，开启调试会让你安装相关的调试插件（比如python相关的），都按上即可。

![](/assets/img/20210701/vs2.png)


vscode调试主要需要两个文件，tasks.json和launch.json，tasks用于在launch前执行任务，launch用于读取执行文件
选择创建launch.json文件，这个文件决定了你在调试中的种种参数设置。示例如下

```
#launch.json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [


        {
            "name": "Debug remote", # 起个名字
            "type": "python", # 程序类别
            "request": "launch",
            "program": "${file}", #程序文件
            "python": "/home/zdk19/anaconda3/envs/mmdetaugfpn/bin/python", #这是conda虚拟环境对应的解释器
            "console": "integratedTerminal",
            "args": [
                "./resample_aug_training_log/final_more_anchor_refine/cascade_rcnn_x101_fpn_1x_more_anchor_globalroi_seesaw_openBrand_dataaug_resample_refine_higher_resample_thr_7_softnms_filp.py",
                "./resample_aug_training_log/final_more_anchor_refine/epoch_1.pth",
                "--format-only",
                "--options", "jsonfile_prefix=./resample_aug_training_log/test_remote_ssh",
                "--cfg-options", "gpu_ids=7"
              ],  #输入参数
            "cwd": "${workspaceFolder}" #调试所处的目录
        }
    ]
}
```

实际上python似乎不是很需要task步骤，c++中一般用来做编译步骤，示例如下，如果要用tasks，只需在launch.json中加入一行
```
"preLaunchTask": "(task的label名)"
```
```
#tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "python",
            "type": "shell",
            "command": "/home/ml/anaconda3/envs/py36/bin/python",  #这个是虚拟环境 conda info --envs 可以看虚拟环境的地址
            "args": [
                "${file}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": [
                "$eslint-compact"
            ]
        }
    ]
}
```

3. GPU的选择
在程序运行之前，单独运行一行代码，可以改变该终端的gpu可见状况

```
export CUDA_VISIBLE_DEVICES=7
```
这样合理的调度自己使用的Gpu，高效的利用资源，也防止和别人冲突。