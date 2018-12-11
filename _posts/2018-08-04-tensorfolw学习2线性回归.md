---
layout:     post
title:      tensorflow学习(2)线性回归
subtitle:  
date:       2018-08-04
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - 线性回归
    - 学习tensorflow
    - 变量

---
## 线性回归


#### 整个工作过程是
1创建需要拟合的数据

2创建需要学习的变量（变量是学习出来的）

3设置预测函数，如何输出回归结果

4创建损失函数

5创建学习算法

6创建最优化的方法

7初始化所有的变量和整个图

8启动图
 

```
import tensorflow as tf
import  numpy as np


x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3
#创建数据，随机生成一个数据
weights=tf.Variable(tf.random.uniform([1],-1.0,1.0))
#创建权重变量，变量是一个数，这个数是一个分布在-1到1之间的均匀分布
biases=tf.Variable(tf.zeros([1]))#创建偏执变量
#创建一个偏置
y=x_data*weights+biases
#创建一个预测函数


loss=tf.reduce_mean(tf.square(y-y_data))  
#创建损失函数，reduce_mean()函数是指定轴上求平均如果未指定则求所           
 有值平均
op=tf.train.GradientDescentOptimizer(0.5) #创建学习方法
train=op.minimize(loss)#创建最优化的方式
init=tf.global_variables_initializer()#初始化参数变量


seen=tf.Session()#启动图
seen.run(init)

for i in range(50):
    seen.run(train)
    print(i,seen.run(weights),seen.run(biases))
    #输出变量的值

```

