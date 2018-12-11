---
layout:     post
title:      tensorflow学习(3)神经网络
subtitle:  
date:       2018-08-05
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - 神经网络
    - 学习tensorflow

---

### 创建简单的神经网络算法模型

```
import tensorflow as tf
import numpy as np


#定义一层网络（就是定义一层网络权重矩阵和偏置向量）
def layer_(inputs,insize,outsize,af=None):
#af是激活函数，
    weight=tf.Variable(tf.random_normal([insize,outsize]))
    b=tf.Variable(tf.zeros([1,outsize])+0.1)
    a=tf.matmul(inputs,weight)+b
    if af is None:
        outputs=a
    else:
        outputs=af(a)
    return outputs
#返回时网络的输出

x_data=np.linspace(-1,1,300,dtype=np.float32)[:, np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)-0.5+noise
#定义一个拟合问题的数据，加上noise噪音数据

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
#定义一个输入输出占位符
l1=layer_(xs,1,10,af=tf.nn.relu)
p=layer_(l1,10,1,af=tf.nn.relu)
#定义网络结构

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - p),reduction_indices=[1]))
#定义损失函数，最后那个reduction_indices没什么用

train_=tf.train.GradientDescentOptimizer(0.1)

x=train_.minimize(loss)
init=tf.global_variables_initializer()
#初始化所有变量

seen=tf.Session()
seen.run(init)


for i in range(100):
    seen.run(x,feed_dict={xs:x_data,ys:y_data})
    print("训练次数",str(i))
    print(seen.run(loss,feed_dict={xs: x_data, ys: y_data}))
#最终更新的事变量，这个变量随着更新改变，
seen.close()

```