---
layout:     post
title:      Loss
subtitle:   
date:       2022-09-07
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - Transformer
---

## Transformer
出自google的经典论文 Attention is All YouNeed  
除了未来5年内所有的技术都是在这个技术基础之上实现的  

![Transformer](/img/20230313/attentiton.png) 
结构之简单性能之强大可以说牛逼  
介绍一下这个结构
#### Encode  
首先embedding不用多介绍主要是word embedding 和 位置embedding  
计算求和后输出输出到attention QKV 交互 Q是索引，表示一句话的一个字  
K 是一句话所有的字，用一个字计算所有的关系，然后将这个关系体现在V上
然后计算和输入的残差，再标准化，再经过FFD再计算残差和标准化  
如此就是一个模块，多个模块串联就是encode结构   
经典的BERT就是Encode结构，再损失上增加了MLM和句子判断模型
#### Decode
相比Encode更复杂一些 多了两个核心的一个是Mask的Attention一个是来自  
Encode的输出也VK和Decode输出Q进行计算也就是cross Attention
Mask的Attention就是对qk矩阵进行mask 避免看到序列每个字后面的字的结果  
GPT用的就是去掉cross Attention