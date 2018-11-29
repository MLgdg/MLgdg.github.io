---
layout:     post
title:      论文实现：python计算MACD值以及ADF检验及数据分析
subtitle:  
date:       2018-11-28
author:     高庆东
header-img: img/ceshi.jpg
catalog: true
tags:
    - pandas
    - pandas数据组合
    - ADF
    - MACD
    - 股票数据分析
    
--


这段代码主要是实现了一篇关于交易股票算法的论文上提到的方法，论文上说这个方法基本上可以赚钱，好像很吊的样子
## 数据读取

    import pandas as pd
    import numpy as np
    import datetime
    import time

    #获取数据
    def csv_read():
        df=pd.read_csv('mcad.csv')
        df.columns=['date','time','open','high','low','close','volume','amt']
        #df=df[['date','open','high','low','close','volume','amt']]
        #print(df.head(5))
        return df

    #划分为一天为单位groupby的作用是给某列分组，后面的是其他列在一那列分组为基础的前提下，取值
    def time2date(df):
        z = df.groupby(['date']).size()
        df_=df.groupby('date').agg({'open':'first', 'high': 'max', 'low': 'min','close':'last'})
        return pd.DataFrame(df_)
        
## MACD计算
    import pandas as pd
    import numpy as np
    import datetime
    import time
    import csv_read as cr

    def get_EMA(df,N):
        for i in range(len(df)):
            if i==0:
                df.ix[i,'ema']=df.ix[i,'close']
            if i>0:
                df.ix[i,'ema']=(2*df.ix[i,'close']+(N-1)*df.ix[i-1,'ema'])/(N+1)
        ema=list(df['ema'])
        return ema

    def get_MACD(df,short=12,long=26,M=9):
        a=get_EMA(df,short)
        b=get_EMA(df,long)
        df['diff']=list(pd.Series(a)-pd.Series(b))
        #print(df.head(5))
        for i in range(len(df)):
            if i==0:
                df.ix[i,'dea']=df.ix[i,'diff']
            if i>0:
                df.ix[i,'dea']=(2*df.ix[i,'diff']+(M-1)*df.ix[i-1,'dea'])/(M+1)
        df['macd']=2*(df['diff']-df['dea'])
        return df

    def get_hat_MACD(df,long=26,M=9):
        for i in range(len(df)):
            if i<long+M:
                df.ix[i,'h_macd']=0
            if i>=long+M:
                df.ix[i,'h_macd']=df.ix[i,'macd']/df.ix[i-(long+M),'close']
        return df


## ADF检验
    import numpy as np
    import statsmodels.tsa.stattools as st
    import macd_cal as mc 
    import csv_read as cr


    df=cr.csv_read()
    df=cr.time2date(df)
    macd=mc.get_MACD(df,12,26,9)
    h_mc=mc.get_hat_MACD(macd,26,9)

    res=st.adfuller(h_mc['h_macd'],1)
    print(res)
