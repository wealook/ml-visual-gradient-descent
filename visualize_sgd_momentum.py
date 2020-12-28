from base import *


def visualize_sgd_momentum(lr=0.1, gamma=0.1, epoch=10, init_x1=-4, init_x2=-2.4):
    '''lr: learning rate
    gamma: parameter for momentum sgd
    每次都会根据上一次的动量来进行更新
    '''

    def momentum(x1, x2, v1, v2, lr):
        dfdx1, dfdx2 = f_grad(x1, x2)
        v1 = gamma * v1 + lr * dfdx1
        v2 = gamma * v2 + lr * dfdx2
        x1 = x1 - v1
        x2 = x2 - v2
        return (x1, x2, v1, v2, lr)

    def momentum_1(x1, x2, v1, v2, lr):
        dfdx1, dfdx2 = f_grad(x1, x2)
        v1 = gamma * v1 - lr * dfdx1
        v2 = gamma * v2 - lr * dfdx2
        x1 = x1 + v1
        x2 = x2 + v2
        return (x1, x2, v1, v2, lr)

    res = train_2d(momentum, lr, epoch, init_x1, init_x2)
    plot_2d(res, title='momentum')


if __name__ == '__main__':  # 运行
    visualize_sgd_momentum()
