---
layout:     post
title:      tensorflow学习(5)参数存取
subtitle:  
date:       2018-09-06
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - 参数存取
    - tensorflow学习

---

### 参数存取

需要注意的的是保存的是所有变量取的也是所有变量
取的时候一定要按存的顺去来取

这个涉及到fintune问题

```
import tensorflow as tf
import numpy as np
# 保存参数
w=tf.Variable([[1,2,3],[4,5,6]],dtype=tf.float32,name="weight")
b=tf.Variable([[1,2,3]],dtype=tf.float32,name='bias')

init=tf.global_variables_initializer()
seen=tf.Session()

save=tf.train.Saver()
seen.run(init)

save_path=save.save(seen,"Download/save.ckpt")

#提取参数
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

saver = tf.train.Saver()
with tf.Session() as sess:
    # 提取变量
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))
```
