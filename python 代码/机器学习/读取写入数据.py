import os
import pandas as pd
os.makedirs(os.path.join("data"),exist_ok= True)
data_file = os.path.join('data','2house_tiny.csv')
with open(data_file,'w') as f:
    f.write('nujo,polo,llkj\n')
    f.write('NA,pave,1452\n')
    f.write('2,NA,10026\n')
    f.write('NA,541,4569\n')
    f.write('NA,NA,2564\n')
data = pd.read_csv(data_file)
print(data)
