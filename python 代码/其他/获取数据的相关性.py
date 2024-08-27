import pandas as pd  
import numpy as np  
  
# 读取Excel文件  
# 假设你的Excel文件名为'data.xlsx'，并且你想要读取第一个sheet（索引为0）  
file_path = 'C:\\Users\\A208-11\\Desktop\\data.xlsx'
df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl',header=None)  

# 将DataFrame转换为numpy数组  
numpy_array = df.values  # 或者使用 df.to_numpy()  

# 其中H的每一行分别表示1月最大值，7月最大值到7月平均
# H的每一列表示的是整区一直到12区
H = np.zeros((6,14))
for i in range(6):
    for j in range(14):
        H[i,j] = np.corrcoef(numpy_array[2:,j+1].astype(float),numpy_array[2:,j+1+(i+1)*15].astype(float))[0,1]
print(H)
