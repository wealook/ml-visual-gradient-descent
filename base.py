import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import seaborn as sns
from ipywidgets import *
import math

# 绘图元素比例  比较小
sns.set_context('paper', font_scale=2)
# 设置显示中文字体
font = {'family': 'SimHei',
        #         'weight' : 'bold',
        'size': '15'}
plt.rc('font', **font)  # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）


def f_2d(x1, x2):
    '''优化的目标函数 有2个变量'''
    return 0.5 * x1 ** 2 + x2 ** 2


def f_grad(x1, x2):
    '''x1和x2的梯度'''
    dfdx1 = x1
    dfdx2 = 2 * x2
    return dfdx1, dfdx2


def train_2d(trainer, lr, epoch=50, init_x1=-4, init_x2=-4):
    """自定义训练器:trainer
       lr:学习率
       epoch:轮次
       init_x1:x1的初始值 便于观察
       init_x2:x1的初始值
    """
    x1, x2 = init_x1, init_x2
    s_x1, s_x2 = 0, 0
    res = [(x1, x2)]
    for i in range(epoch):
        x1, x2, s_x1, s_x2, lr = trainer(x1, x2, s_x1, s_x2, lr)
        res.append((x1, x2))
    return res


def plot_2d(res, figsize=(10, 8), title=None):
    x1_, x2_ = zip(*res)
    fig = plt.figure(figsize=figsize)
    plt.plot([0], [0], 'r*', ms=15)  # 画出终点（五角星）
    plt.text(0.0, 0.25, '最小值', color='k')
    plt.plot(x1_[0], x2_[0], 'ro', ms=10)  # 画出起点（大红圆点）
    plt.text(x1_[0] + 0.2, x2_[0] - 0.15, '起点', color='k')
    plt.plot(x1_, x2_, '-o', color='r')  # 画出优化过程中的轨迹及上面的点(点的稀疏与学习率相关，)

    plt.plot(x1_[-1], x2_[-1], 'bo', ms=10)  # 画出通过优化器更新后最后的坐标点（蓝色圆点）
    plt.text(x1_[-1] + 0.1, x2_[-1] - 0.4, '终点', color='k')

    x1 = np.linspace(-5, 5, 100)
    x2 = np.linspace(-5, 5, 100)  # linespace(起始值，结束值，值的个数)
    x1, x2 = np.meshgrid(x1, x2)  # 生成网格点,# 把x1,x2数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
    # print(x1)
    # print(x2)

    x3 = [1, 2, 3]
    x4 = [4, 5]
    x3, x4 = np.meshgrid(x3, x4)  # 该函数将x3的行数copy到len(x4),将x4的列数copy到len(x3)
    # print(x3)
    # print(x4)
    # 画出等高线区域，并用彩虹颜色填充，设置透明度
    cp = plt.contourf(x1, x2, f_2d(x1, x2), 7, alpha=0.75, cmap=cm.rainbow)  # 把损失函数值相同区域涂成相同色，数字指定分为多少个区域
    # 画出等高线
    C = plt.contour(x1, x2, f_2d(x1, x2), 7, colors='black')  # 根据损失函数值的画出等高线，数字指定分为多少个区域
    # print(f_2d(x1, x2).shape)
    plt.clabel(C, inline=True, fontsize=15)
    cbar = fig.colorbar(cp)
    cbar.set_label('损失值')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.show()
    x1, x2 = res[-1]
    loss = f_2d(x1, x2)
    print('最小损失值:', loss, 'x1:', x1, ' x2:', x2)


def sgd(x1, x2, s1, s2, lr):
    dfdx1, dfdx2 = f_grad(x1, x2)
    return (x1 - lr * dfdx1, x2 - lr * dfdx2, 0, 0, lr)







