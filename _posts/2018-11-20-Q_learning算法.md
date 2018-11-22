---
layout:     post
title:      Q-learning算法
subtitle:  
date:       2018-11-20
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - 强化学习
    - python
    - 损失函数
    - Q-learning
---

## 强化学习
核心是两个矩阵 Q和两个矩阵的维度相同，R矩阵表示表示可以每个动作     
和状态的矩阵，Q矩阵表示每个动作下学习到的经验。

Agent表示具备行为能力的物体
Action表示物体可以的动作，动作从动作集中选择，动作集影响任务求解
Reward表示agent执行了动作影响了环境，环境变化的用reward表示

Q的迭代计算中是需要下一个Q值的，实际上不可能知道下一个Q值所以
再迭代计算的时候需要勇敢Q的上一个值，

## 代码
    import numpy as np
    import random
    #r举证相当于一场游戏中的规则有6个状态每个状态间移动获得的收益
    r = np.array([[-1, -1, -1, -1, 0, -1], [-1, -1, -1, 0, -1, 100], [-1, -1, -1, 0, -1, -1], [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100], [-1, 0, -1, -1, 0, 100]])
    #初始化Q表都是0
    q = np.zeros([6,6],dtype=np.float32)
    gamma = 0.8
    print(random.randint(0,5))
    step = 0
    #迭代100次
    while step < 100:    
         state = random.randint(0,5)
         #判断起点是否是终点
         if state != 5:
              next_state_list=[]
              for i in range(6):
                    #找这个状态下可行的状态，
                    if r[state,i] != -1:
                next_state_list.append(i)
        #随机找一个可行的状态
        next_state = next_state_list[random.randint(0,len(next_state_list)-1)]
        #计算更新Q表
        qval = r[state,next_state] + gamma * max(q[next_state])
        q[state,next_state] = qval
        step=step+1
        print(step)
    print(q)

## 验证

    for i in range(10):
        #随机初始化一个位置
        print("第{}次验证".format(i + 1))
        state = random.randint(0, 5)
        print('机器人处于{}'.format(state))
        count = 0
        while state != 5:
            if count > 20:
                print('fail')
                break
            # 选择最大的q_max
            q_max = q[state].max()
            #选择Q值最大的那一步，作为下一步的位置
            q_max_action = []
            for action in range(6):
                if q[state, action] == q_max:
                    q_max_action.append(action)
            next_state = q_max_action[random.randint(0, len(q_max_action) - 1)]
            print("the robot goes to " + str(next_state) + '.')
            state = next_state
            count += 1





