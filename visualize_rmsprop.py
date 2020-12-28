from base import *


@interact(lr=(0, 4, 0.001),
          gamma=(0, 0.99, 0.001),
          continuous_update=False, epoch=(0, 100, 1), init_x1=(-5, 5, 0.1), init_x2=(-5, 5, 0.1))
def visualize_rmsprop(lr=0.1, gamma=0.9, epoch=50, init_x1=-4, init_x2=-2.4):
    '''lr: learning rate,
       gamma: momentum'''

    def rmsprop_2d(x1, x2, s1, s2, lr):
        eps = 1e-6
        g1, g2 = f_grad(x1, x2)
        s1 = gamma * s1 + (1 - gamma) * g1 ** 2
        s2 = gamma * s2 + (1 - gamma) * g2 ** 2
        x1 -= lr / math.sqrt(s1 + eps) * g1
        x2 -= lr / math.sqrt(s2 + eps) * g2
        return x1, x2, s1, s2, lr

    res = train_2d(rmsprop_2d, lr, epoch, init_x1, init_x2)
    plot_2d(res, title='RMSProp')


if __name__ == '__main__':  # 运行
    visualize_rmsprop()
