# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:46:48 2016

@author: yiran.zhou
"""

import pandas as pd
import numpy as np

# 计算一组数据对应的区间，用来做分组的histogram
# 输入待处理的series, 分组的间距，分组的最大最小值，左闭右开还是左开右闭
# 输出一个新的数列cut，作为分组的key, 格式为'< X' or '<= X'、'X ~ Y'、'> Y' or '>= Y'
def histoCut(s, step, maxN = None, minN = None, leftClose = True):
    cut = []
    maxN = np.ceil(s.max() / step) * step if maxN == None else maxN
    minN = np.floor(s.min() / step) * step if minN == None else minN
    if leftClose == True: #左闭右开 [ )
        for e in s:
            if e >= maxN:
                cut.append('>= ' + str(maxN))
            elif e < minN:
                cut.append('< ' + str(minN))
            else:
                upper = np.ceil(e / step) * step
                lower = np.floor(e / step) * step
                upper = upper + step if upper == lower else upper
                cut.append(str(lower) + ' ~ ' + str(upper))    
    else:   #左开右闭 ( ]
        for e in s:
            if e > maxN:
                cut.append('> ' + str(maxN))
            elif e < minN:
                cut.append('<= ' + str(minN))
            else:
                upper = np.ceil(e / step) * step
                lower = np.floor(e / step) * step
                lower = lower - step if upper == lower else lower
                cut.append(str(lower) + ' ~ ' + str(upper))     
    return cut
    
# 给histogram的索引排序
# 输入是一个以histogram cut为索引的dataframe，索引的格式是'< X' or '<= X'、'X ~ Y'、'> Y' or '>= Y'
# 输出是排完序的dataframe
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

# 计算一组数据中最后一个数据的Z-score, 输入是series（dataframe的列）
def zscore(s):
    zs = (s[-1] - s.mean()) / s.std()
    return zs

# dataframe的rolling函数, 对指定的列做rolling，生产新的列，返回一个新的df
# colNames指定df里面需要rolling的列，newColNames是新生成的列的名字，funcList是处理rolling的函数
# funcList里面的每个函数输入是Series，输出是单一值
# funcList,colNames和newColNames的数量需相等
# span有正负，正表示未来的x days，负表示过去的x days，均包含当前日期
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