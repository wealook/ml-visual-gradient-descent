from base import *


# 定义了4个变量控件，可以随时调节，查看效果 (最小值，最大值，步长)

def visualize_gradient_descent(lr=0.05, epoch=50, init_x1=-4, init_x2=-2.4):
    res = train_2d(sgd, lr, epoch, init_x1, init_x2)
    plot_2d(res, (12, 8), title='原始GD')


if __name__ == '__main__':  # 运行
    visualize_gradient_descent()
