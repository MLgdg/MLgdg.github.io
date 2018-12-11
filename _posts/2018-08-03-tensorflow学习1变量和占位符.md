---
layout:     post
title:      tensorflow学习(1)变量和占位符
subtitle:  
date:       2018-08-03
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - 变量
    - 占位符
    - tensorflow

---

## 变量
```
import tensorflow as tf 
import numpy as np

a=tf.Variable(1) #变量可以是任意大小，形状，随机，分布
b=tf.constant(1) #常量乐意是任意大小形状的
new_1=tf.add(a,b) #加法
upda=tf.assign(a,new_1)#将a跟新成new_1
init=tf.initialize_all_variables()#初始化所有变量常量
seen=tf.Session()
seen.run(init) #启动构建好的图，图中的每一个节点都要启动才能输出
seen.run(upda)
print(seen.run(a))
```
## 占位符
```
input1=tf.placeholder(tf.float32)
#创建一个变量可以是任意形状的输入的时候需要feed喂入相应格式的数据
input2=tf.placeholder(tf.float32)
#占位符用来定义输入输出时用，这些变量需要手动输入的
out=tf.multiply(input1,input2)
seen=tf.Session()

b=seen.run(out,feed_dict={input1:2,input2:3})
#占位符需要输入数据，数据的输入是以定义的格式来输入
print(b)
seen.close()。#启动图后要关闭它，
```



