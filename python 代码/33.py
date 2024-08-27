import numpy as np
import matplotlib.pyplot as plt

# 定义网格范围
x =2 # np.linspace(-5, 5, 20)
y =3 #np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x, y)

# 定义向量函数 F(x, y) = (y, -x)
U = Y  # x 方向的分量
V = -X  # y 方向的分量

# 绘制向量场
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', scale=1)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.title('Vector Field: F(x, y) = (y, -x)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.show()

