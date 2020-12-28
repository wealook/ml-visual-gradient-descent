from base import *


def visualize_sgd_inertia(lr=0.1, gamma=0.1, epoch=10, init_x1=-4, init_x2=-2.4):
    '''lr: learning rate
    gamma: parameter for inertia sgd'''

    def inertia(x1, x2, v1, v2, lr):
        dfdx1, dfdx2 = f_grad(x1, x2)
        v1 = gamma * v1 + (1 - gamma) * dfdx1
        v2 = gamma * v2 + (1 - gamma) * dfdx2
        x1 = x1 - lr * v1
        x2 = x2 - lr * v2
        return (x1, x2, v1, v2, lr)

    res = train_2d(inertia, lr, epoch, init_x1, init_x2)
    plot_2d(res, title='inertia')


if __name__ == '__main__':  # 运行
    visualize_sgd_inertia()
