# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:46:48 2016

@author: yiran.zhou
@description: 基于主要是Pandas和numpy的数据处理函数库
"""

import pandas as pd
import numpy as np
from pandas.tseries.offsets import Day
from matplotlib.pylab import date2num
import matplotlib.pyplot as plt
import matplotlib.finance as mpf

# 计算一组数据对应的所属区间，用来做分组的histogram
# 输入:待处理的series, 分组的间距，分组的最大最小值，左闭右开还是左开右闭
# 输出:一个新的数列cut，作为分组的key, 每个key的格式为'< X' or '<= X'、'X ~ Y'、'> Y' or '>= Y'
def histoCut(s, step, maxN = None, minN = None, leftClose = True):
    cut = []
    maxN = np.ceil(s.max() / step) * step if maxN == None else maxN
    minN = np.floor(s.min() / step) * step if minN == None else minN
    for e in s:
        cut.append(histoCutKey(e, step, maxN, minN, leftClose))
    return cut
    
# 计算某个值所属的区间
# 输入:单个数字
# 输出: string，作为分组的key, 格式为'< X' or '<= X'、'X ~ Y'、'> Y' or '>= Y'   
def histoCutKey(e, step, maxN = None, minN = None, leftClose = True):
    key = ''
    maxN = e + 1 if maxN == None else maxN #如果不指定max、min，必然生成'X ~ Y'格式
    minN = e - 1 if minN == None else minN
    if leftClose == True: #左闭右开 [ )
        if e >= maxN:
            key = '>= ' + str(maxN)
        elif e < minN:
            key = '< ' + str(minN)
        else:
            upper = np.ceil(e / step) * step
            lower = np.floor(e / step) * step
            upper = upper + step if upper == lower else upper
            key = str(lower) + ' ~ ' + str(upper)
    else:   #左开右闭 ( ]
        if e > maxN:
            key = '> ' + str(maxN)
        elif e < minN:
            key = '<= ' + str(minN)
        else:
            upper = np.ceil(e / step) * step
            lower = np.floor(e / step) * step
            lower = lower - step if upper == lower else lower
            key = str(lower) + ' ~ ' + str(upper)
    return key
    
# 给histogram的索引排序
# 输入:是一个以histogram cut为索引的dataframe，索引的格式是'< X' or '<= X'、'X ~ Y'、'> Y' or '>= Y'
# 输出:是排完序的dataframe
# 相关func：histoCut
def histoSort(df, ascending = True):
    tmpIdx = 'tmpIdx'
    maxi = None
    mini = None
    i = 0
    # 新建一个列把index的str转化成可以排序的数字
    df[tmpIdx] = None
    while i < df.index.size:
        if df.index[i][0] == '>':
            maxi = i
        elif df.index[i][0] == '<':
            mini = i
        else:
            df[tmpIdx][i] = float(df.index[i].split(' ')[0])
        i += 1
    if maxi != None:
        df[tmpIdx][maxi] = df[tmpIdx].max() + 1
    if mini != None:
        df[tmpIdx][mini] = df[tmpIdx].min() - 1
    # 用新的列排序
    df = df.sort_values(tmpIdx, ascending = ascending)
    df = df.drop(tmpIdx, axis = 1)
    return df

# 计算一组数据中最后一个数据的Z-score
# 输入:是series（dataframe的列）
# 输出：最后一个数据的Z-score
def zscore(s):
    zs = (s[-1] - s.mean()) / s.std()
    return zs
    
# 计算一组数据中正值的比率（包括0） 
# 输入：series（dataframe的列）
# 输出：正值比率
def posPct(s):
    pct = s[s>=0].count() / s.count()    
    return pct

# dataframe的rolling函数, 对指定的列做rolling，生产新的列，返回增加列之后的原df
# 输入：colNames指定df里面需要rolling的列的list，newColNames是新生成的列的名字的list，funcList是处理rolling的函数
# funcList里面的每个函数输入是一个Series，输出是单一值。如果要输入多个Series，使用rollingND函数
# funcList,colNames和newColNames的数量需相等
# span有正负，正表示未来的x days，负表示过去的x days，均包含当前日期
# 输出：增加新生成列之后的原df，会产生nan
def rolling(df, span, funcList, colNames, newColNames):
    i = 0
    # 对每一个需要处理的列进行循环
    while i < len(colNames):
        col = df[colNames[i]]
        newCol = newColNames[i]
        # 根据span设定初始rolling范围
        if span < 0: # 往前rolling过去的数据
            idx = 0 - span - 1 # 指向当前的日期
            startIdx = 0 # rolling开始的日期
            endIdx = idx # rolling结束的日期
        else: # 往后rolling未来的数据
            idx = 0
            startIdx = idx
            endIdx = startIdx + span - 1
        # 对列进行rolling操作
        while endIdx < len(col):
            interval = np.arange(startIdx, endIdx + 1)
            rollingData = col.ix[col.index[interval]]
            df.ix[idx, newCol] = funcList[i](rollingData)
            idx +=1
            startIdx +=1
            endIdx +=1
        # 继续下一列
        i += 1
    return df
    
# dataframe的rolling函数, 对指定的复数列做rolling，生产新的单一列，返回增加列之后的原df
# 输入：colNames指定df里面需要rolling的列，newColName是新生成的列的名字，func是处理rolling的函数
# func输入是dataframe(复数Series)，输出是单一值
# span有正负，正表示未来的x days，负表示过去的x days，均包含当前日期
# 输出：增加新生成列之后的原df，会产生nan
# 和rolling的区别：
#rolling可以同时处理多个列，每个列应用不同的函数，生成同样多的列，但是每个rolling函数只能读入一个列。
#rollingND(N-Dimension)只能生成一列，应用一个rolling函数，但是这个函数可以读入多个列。
def rollingND(df, span, func, colNames, newColName):
    cols = df[colNames]
    # 根据span设定初始rolling范围
    if span < 0: # 往前rolling过去的数据
        idx = 0 - span - 1 # 指向当前的日期
        startIdx = 0 # rolling开始的日期
        endIdx = idx # rolling结束的日期
    else: # 往后rolling未来的数据
        idx = 0
        startIdx = idx
        endIdx = startIdx + span - 1
    # 对列进行rolling操作
    while endIdx < len(df.index):
        interval = np.arange(startIdx, endIdx + 1)
        rollingData = cols.ix[cols.index[interval]]
        df.ix[idx, newColName] = func(rollingData)
        idx +=1
        startIdx +=1
        endIdx +=1
    return df
        
# 计算一组任意日期的附近时间范围内的指定日期，比如找出这组日期之后一周内的周五，类似于偏移时间数列
# 输入：一组pd.DatetimeIndex， 周围的日期范围，指定的日期,如'W-FRI'
# 输出： 一组pd.DatetimeIndex
def findNearbyDate(timeIdx, shiftDate, freq):
    timeIdxShift = timeIdx.shift(shiftDate, 'D')     
    l = []
    i = 0
    while i < len(timeIdx):
        if shiftDate > 0:
            l.append(pd.date_range(timeIdx[i], timeIdxShift[i], freq = freq)[0])
        else:
            l.append(pd.date_range(timeIdxShift[i], timeIdx[i], freq = freq)[0])
        i += 1
    l = pd.DatetimeIndex(l)
    return l
        

# 找到给定一组时间之后一段时间内的符合某个条件的日期和数值，比如找之后一个月内的最大跌幅
# 输入： 一个dataframe，包含日期和价格，pxColName, 价格列的名称， 一个pd.DatetimeIndex(需要是dataframe.index的子集),
#  一个条件函数（min（）），一组日期偏移量（[7,14,28]），列名称后缀('Low')
# 输出：一个新的dataframe包含找到的数值和日期
def findConditionalValue(df, pxColName, timeIdx, func, dateSftLst, suffixStr):
    
    # 设定columns的列名称, 包含3个数据：价格，时间，变化百分比
    i = 0
    cols = [pxColName]
    while i < len(dateSftLst):
        cols.append(str(dateSftLst[i]) + 'D ' +  suffixStr)
        cols.append(str(dateSftLst[i]) + 'D ' +  suffixStr + 'Date')
        cols.append(str(dateSftLst[i]) + 'D ' +  suffixStr + 'Pct Chg')
        i += 1
    
    # 把时间序列和对应的价格放到新的结果的df里面  
    res = pd.DataFrame(columns = cols, index = timeIdx)
    res[pxColName] = df.ix[timeIdx, pxColName]
    
    # 找出给定条件的时间和价格
    k = 0
    while k < len(timeIdx):
        
        StartDay = res.index[k]
        j = 0
        while j < len(dateSftLst):
            endDate = min(StartDay + Day(dateSftLst[j]), df.index[-1])
            pxCond = func(df.ix[StartDay:endDate, pxColName])
            pxCondDate = df[df[pxColName] == pxCond].index[0]
            pxPctChg = pxCond / res.ix[k, pxColName] - 1        
            
            res.ix[k, j*3] = pxCond
            res.ix[k, j*3+1] = pxCondDate
            res.ix[k, j*3+2] = pxPctChg
            
            j += 1
        
        k += 1
    return res
        
# 把dataframe转换成list格式。matplotlib可以用这个list来画蜡烛图
# 输入：dataframe的index是日期，columns是'Open', 'Close','Low','High'
# 输出：list，每个元素是一个tuple，顺序是(t,open,high,low,close)
def df2OHLC(df):
    
    data_lst = []
    i = 0
    while i < len(df.index):
        dt = df.index[i].to_datetime()
        t = date2num(dt)
        Open = df.ix[i, 'Open']
        Close = df.ix[i, 'Close']
        Low = df.ix[i, 'Low']
        High = df.ix[i, 'High']
        data = (t, Open, High,Low, Close)
        data_lst.append(data)        
        
        i += 1
    return data_lst

# 画蜡烛图
# 输入：df2OHLC的结果
def darwCandleChart(lst):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    # 设置X轴刻度为日期时间
    ax.xaxis_date()
    plt.xticks(rotation=45)
    plt.yticks()
    plt.title("candle stick")
    plt.xlabel("Date")
    plt.ylabel("Price")
    mpf.candlestick_ohlc(ax,lst,width=0.5,colorup='r',colordown='g')
    plt.grid()
    
# 通过月份找合约的月份代码
def month2ticker(m):
    idx = [1,2,3,4,5,6,7,8,9,10,11,12]
    data = list('FGHJKMNQUVXZ')    
    df = pd.DataFrame(data, index = idx, columns = ['symbol'])
    return df.ix[m, 'symbol']

# 通过月份代码找合约月份
def ticker2month(s):    
    data = [1,2,3,4,5,6,7,8,9,10,11,12]
    idx = list('FGHJKMNQUVXZ')    
    df = pd.DataFrame(data, index = idx, columns = ['month'])
    return df.ix[s, 'month']