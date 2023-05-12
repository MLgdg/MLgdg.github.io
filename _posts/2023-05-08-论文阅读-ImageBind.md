---
layout:     post
title:      大模型训练
subtitle:   
date:       2023-05-04
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - 多模态
    - audio
---


## 模型结构
1、6个Transformer 分别对应六种模态  
2、每个模态有个hear  
3、对每种模态输出加一个参数  

## video模态
使用3d卷积展开
## audio模态
对音频分段然后提取mel特征，使用2d卷积扩展，
## text模态
自回归模型
## depth模态
## 红外模态
## 坐标模态
