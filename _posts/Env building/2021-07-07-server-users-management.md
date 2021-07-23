---
layout: post
title: '服务器用户管理'
date: 2021-07-01
author: Poley
cover: '../assets/img/mycat3.jpg'
tags: 环境配置
---

# 新建/删除用户分组
新建

```
sudo groupadd 组名 -g GID
```
帮助
```
sudo groupadd -h #查看帮助
```

查看
```
cat /etc/group
```

删除
```
sudo groupdel 组名
```

# 新建/删除用户/添加用户到组中
已有用户
```
adduser 用户名 组名
```

新用户
```
adduser --group GROUP [--gid ID] 用户名
```

注：在CentOs下useradd与adduser是没有区别的都是在创建用户，在home下自动创建目录，没有设置密码，需要使用passwd命令修改密码。

**而在Ubuntu下useradd与adduser有所不同**

1、useradd在使用该命令创建用户是不会在/home下自动创建与用户名同名的用户目录，而且不会自动选择shell版本，也没有设置密码，那么这个用户是不能登录的，需要使用passwd命令修改密码。

2、adduser在使用该命令创建用户是会在/home下自动创建与用户名同名的用户目录，系统shell版本，会在创建时会提示输入密码，更加友好

删除用户
```
userdel -r 用户名 # -r代表连其目录一起删了
```

提升用户到管理员（sudo权限）
```
sudo usermod -a -G adm wyx
sudo usermod -a -G sudo wyx
```
用adduser也可以


# 用户权限

## 基本概念
用户是Linux系统工作中重要的一环用户管理包括 用户与组管理
在Linux系统中，不论是由本机或是远程登录系统，每个系统都必须拥有一个账号，并且对于不同的系统资源拥有不同的使用权限
在Linux 中可以指 **每一个用户**针对 **不同的文件或者目录** 的 **不同权限**
对 文件／目录 的权限包括：
读写执行 rwx

![](/assets/img/20210707/rwx.png)

## 组
为了方便用户管理，提出了**组**的概念，在实际应用中，可以预先针对**组**设置好权限，然后**将不同的用户添加到对应的组中，从而不用依次为每一个用户设置权限**

## ls拓展

ls -l 可以查看文件夹下文件的详细信息，从左到右依次是：
+ 权限，第 1 个字符如果是 d 表示目录
+ 硬链接数，通俗地讲，就是有多少种方式，可以访问到当前目录／文件
+ 拥有者，家目录下 文件／目录 的拥有者通常都是当前用户组，在 Linux 中，很多时候，会出现组名和用户名相同的情况，
+ 大小
+ 时间
+ 名称

![](/assets/img/20210707/lsl.png)

## chmod
chmod 可以修改 用户／组 对 文件／目录 的权限
命令格式如下,rwx可以用 4 2 1  5 6 7 等代替：
```
chmod +/-rwx 文件名|目录名
```

## 超级用户

Linux 系统中的 root 账号通常 用于系统的维护和管理，对操作系统的所有资源 具有所有访问权限
在大多数版本的 Linux 中，都不推荐 直接使用 root 账号登录系统

在 Linux 安装的过程中，系统会自动创建一个用户账号，而这个默认的用户就称为“标准用户”

### sudo
**su** 是 substitute user 的缩写，表示 使用另一个用户的身份
**sudo** 命令用来以其他身份来执行命令，预设的身份为 root
用户使用 sudo 时，必须先输入密码，之后有 **5 分钟的有效期限**，超过期限则必须重新输入密码

## 查看用户信息

![](/assets/img/20210707/userinfo.png)

![](/assets/img/20210707/usermod.png)

## which
![](/assets/img/20210707/which.png)

## 修改文件/目录权限
![](/assets/img/20210707/chown.png)

![](/assets/img/20210707/chmod.png)
