from base import *


@interact(lr=(0, 4, 0.01),
          continuous_update=False, epoch=(0, 100, 1), init_x1=(-5, 5, 0.1), init_x2=(-5, 5, 0.1))
def visualize_adagrad(lr=0.1, epoch=10, init_x1=-4, init_x2=-2.4):
    '''lr: learning rate'''

    def adagrad_2d(x1, x2, s1, s2, lr):
        g1, g2 = f_grad(x1, x2)
        eps = 1e-6
        s1 += g1 ** 2
        s2 += g2 ** 2
        x1 -= lr / math.sqrt(s1 + eps) * g1
        x2 -= lr / math.sqrt(s2 + eps) * g2
        return x1, x2, s1, s2, lr

    res = train_2d(adagrad_2d, lr, epoch, init_x1, init_x2)
    plot_2d(res, title='Adagrad')


if __name__ == '__main__':  # 运行
    visualize_adagrad()
