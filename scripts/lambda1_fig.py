# 折线图
# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt

# datas
x1 = np.arange(10, 110, 10)
bleu1 = [41.22, 41.83, 41.53, 41.41, 41.50, 41.73, 41.83, 42.22, 41.64, 42.02]

x2 = np.arange(10, 110, 10)
bleu2 = [93.04, 96.09, 97.59, 98.14,  98.48, 98.93 , 98.89,  99.27, 99.27, 99.34 ]

x3 = np.arange(10, 110, 10)
bleu3 = [20.20, 17.13, 19.02,19.89, 19.31, 25.25, 20.16, 18.81,19.07, 21.27 ]

x4 = np.arange(10, 110, 10)
bleu4 = [59.17,  59.28 , 58.89 , 58.78 , 59.14, 59.33, 59.26,59.38,  59.18, 59.31]

# curves
plt.plot(x1, bleu1, 'o-', color='r', label='BLEU')
plt.plot(x2, bleu2, 'o-', color='lime', label='Caption Accuracy')
plt.plot(x3, bleu3, 'o-', color='gold', label='Label Precision')
plt.plot(x4, bleu4, 'o-', color='b', label='METEOR')

# 设置数字标签
for x, bleu in zip([x1, x2, x3, x4], [bleu1, bleu2, bleu3, bleu4]):
    for a, b in zip(x, bleu):
        plt.text(a, b, b, ha='center', va='bottom')

# 设置横坐标轴的刻度(纵坐标不需要设置)
# my_x_ticks = np.arange(0.01, 0.1, 0.01)
# plt.xticks(my_x_ticks)
# 可以设置坐标字
scale_ls = np.arange(10, 110, 10)
index_ls = ['0.01', '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.10']
plt.xticks(scale_ls,index_ls)

# 横坐标的名字 # Iterations   x1000  纵坐标的名字
plt.xlabel('Weight of CMC-Align Loss')
plt.ylabel('Metric Values')

# 图标题 #图例 , loc=”best”是自动选择放图例的合适位置
plt.legend(bbox_to_anchor=(0.7, 1.15), ncol=2)
# plt.legend(loc='best')

#保存图
path = r'G:\NMT_Code\AAAI2022\align-mmt'
# 去除图片周围的白边
plt.savefig(path+"\labmda1_fig.eps", bbox_inches='tight', dpi=1000, pad_inches=0.0)

# 显示图
plt.show()