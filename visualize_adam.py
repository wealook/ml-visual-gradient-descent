
from base import *
@interact(lr=(0, 1, 0.001),
          beta1=(0, 0.999, 0.001),
          beta2=(0, 0.999, 0.001),
          continuous_update=False, epoch=(0, 100, 1), init_x1=(-5, 5, 0.1), init_x2=(-5, 5, 0.1))
def visualize_adam(lr=0.1, beta1=0.9, beta2=0.999, epoch=10, init_x1=-4, init_x2=-2.4):
    '''lr: learning rate
    beta1: parameter for E(g)
    beta2: parameter for E(g^2)
    '''

    def Deltax(m, n, g, t):
        eps = 1.0E-6
        m = beta1 * m + (1 - beta1) * g
        n = beta2 * n + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        n_hat = n / (1 - beta2 ** t)
        dx = lr * m_hat / (math.sqrt(n_hat) + eps)
        return m, n, dx

    def adam_2d(x1, x2, m1, n1, m2, n2, lr, t):
        '''m1, m2: E(g1), E(g2)
           n1, n2: E(g1^2), E(g2^2) where E() is expectation
           lr: learning rate
           t: time step'''
        eps = 1e-6
        g1, g2 = f_grad(x1, x2)  # 得到x1和x2的梯度
        m1, n1, dx1 = Deltax(m1, n1, g1, t)
        m2, n2, dx2 = Deltax(m2, n2, g2, t)
        x1 -= dx1
        x2 -= dx2
        return x1, x2, m1, n1, m2, n2, lr

    def train_adam(trainer, lr, epoch=10, init_x1=-4, init_x2=-4):
        """Train a 2d object function with a customized trainer"""
        x1, x2 = init_x1, init_x2
        m1, n1, m2, n2 = 0, 0, 0, 0
        res = [(x1, x2)]
        for i in range(epoch):
            x1, x2, m1, n1, m2, n2, lr = trainer(x1, x2, m1, n1, m2, n2, lr, i + 1)  # i初始为0，但时间步初始为1，所以传入i+1
            res.append((x1, x2))
        return res

    res = train_adam(adam_2d, lr, epoch, init_x1, init_x2)
    plot_2d(res, title='adam')


if __name__ == '__main__':  # 运行
    visualize_adam()
