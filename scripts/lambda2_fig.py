# 折线图
# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt

# datas
x1 = np.arange(10, 110, 10)
bleu1 = [41.53, 41.52, 41.49, 41.13, 41.56, 41.33, 41.82, 41.46, 42.22, 42.00]

x2 = np.arange(10, 110, 10)
bleu2 = [99.14, 99.24, 99.11, 99.08,  99.16, 99.08 , 99.24, 99.15, 99.27, 99.21]

x3 = np.arange(10, 110, 10)
bleu3 = [28.23, 31.05, 29.84, 19.71, 19.45, 23.07, 19.97, 18.52, 18.81, 14.18]

x4 = np.arange(10, 110, 10)
bleu4 = [59.00,  58.95, 59.14, 58.63, 59.08, 58.81, 59.20, 59.04, 59.38, 59.49]

# curves
plt.plot(x1, bleu1, '*-', color='r', label='BLEU')
plt.plot(x2, bleu2, '*-', color='lime', label='Caption Accuracy')
plt.plot(x3, bleu3, '*-', color='gold', label='Label Precision')
plt.plot(x4, bleu4, '*-', color='b', label='METEOR')

# 设置数字标签
for x, bleu in zip([x1, x2, x3, x4], [bleu1, bleu2, bleu3, bleu4]):
    for a, b in zip(x, bleu):
        plt.text(a, b, b, ha='center', va='bottom')

# 设置横坐标轴的刻度(纵坐标不需要设置)
# my_x_ticks = np.arange(0.01, 0.1, 0.01)
# plt.xticks(my_x_ticks)
# 可以设置坐标字
scale_ls = np.arange(10, 110, 10)
index_ls = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
plt.xticks(scale_ls,index_ls)

# 横坐标的名字 # Iterations   x1000  纵坐标的名字
plt.xlabel('Weight of MMH-Align Loss')
plt.ylabel('Metric Values')

# 图标题 #图例 , loc=”best”是自动选择放图例的合适位置
plt.legend(bbox_to_anchor=(0.7, 1.15), ncol=2)
# plt.legend(loc='best')

#保存图
path = r'G:\NMT_Code\AAAI2022\align-mmt'
# 去除图片周围的白边
plt.savefig(path+"\labmda2_fig.eps", bbox_inches='tight', dpi=1000, pad_inches=0.0)

# 显示图
plt.show()