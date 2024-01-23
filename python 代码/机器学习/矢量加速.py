import math
import time 
import numpy as np
import torch
from d2l import torch as d2l

n = 10000
a = torch.ones(n)
b = torch.ones(n)

class Timer:
    def __init__(self):
        self.times = []   # 创建一个空的时间列表
        self.start()     # 初始化时启动计时器
    
    def start(self):    # 启动计时器
        self.tik = time.time()   # 记录当前时间为起始时间

    def stop(self):   # 停止计时器，并将时间记录在列表中
        self.times.append(time.time() - self.tik)   # 计算当前时间与起始时间的差值，并将差值添加到时间列表中

    def avg(self):  # 返回平均时间
        return sum(self.times) / len(self.times)   # 计算时间列表中所有时间的平均值，并返回结果
    
    def total(self):  # 返回时间总和
        return sum(self.times)   # 计算时间列表中所有时间的总和，并返回结果
    
    def cumsum(self):  # 返回累计时间
        return np.array(self.times).cumsum().tolist()   # 计算时间列表中时间的累计和，并返回一个列表

c = torch.zeros(n)
timer = Timer()
"""
for i in range(n):
    c[i] = a[i] + b[i]
"""
d = a + b
timer.stop()
print(f'Time elapsed: {timer.times[-1]:.5f} sec')
print(d)


    









        
