---
layout:     post
title:      tensorflow学习(4)卷积网络
subtitle:  
date:       2018-09-01
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - 卷积网路
    - 学习tensorflow
    - dropout
    - 滤波器

---

### 实现一个简单的卷积神经网络

```
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#导入手写数字数据集

def weight_variable(shape):
    init=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)
#定义滤波器也就是权重矩阵

def bias_variable(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)
#定义偏置向量

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#定义卷积运算

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#定义池化计算

xs=tf.placeholder(tf.float32,[None,784])#不确定输出图像的个数所以写None
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
#定义输入的数据包括输入输出数据和dropout参数

x_image=tf.reshape(xs,[-1,28,28,1])
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#正向传播过程，需要手动计算卷积后的数据大小尺寸

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
#转化成全链接网络
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob=keep_prob)
#在全链接上加dorpout

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
pr=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#定义输出结果

loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pr+1e-10),reduction_indices=[1]))
#定义损失函数加一个数防止log中的数小于等于0
train_=tf.train.AdamOptimizer(0.001).minimize(loss)

seen=tf.Session()
seen.run(tf.global_variables_initializer())
#启动图初始化

for i in range(100):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    seen.run(train_,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    print(seen.run(loss,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5}))
    print(i)
```








