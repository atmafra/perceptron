import math

import matplotlib.pyplot as plt
import numpy as np


class Activation:

    def name(self):
        raise NotImplemented('Method should be implemented in the subclass')

    def activation(self, x):
        raise NotImplemented('Method should be implemented in the subclass')

    def derivative(self, x):
        raise NotImplemented('Method should be implemented in the subclass')

    def plot(self, x_from: float = -3.0, x_to: float = 3.0):
        resolution = 1000
        x = np.linspace(x_from, x_to, 1000)
        y = np.vectorize(self.activation)(x)
        dy_dx = np.vectorize(self.derivative)(x)
        plt.plot(x, y, label=self.name())
        plt.plot(x, dy_dx, label='{} derivative'.format(self.name()))
        plt.title(self.name() + ' activation function')
        plt.xlabel('input')
        plt.ylabel('activation')
        plt.grid()
        plt.legend()
        plt.show()


class Linear(Activation):

    def name(self):
        return 'Linear'

    def activation(self, x):
        return x

    def derivative(self, x):
        return 1.


class Sigmoid(Activation):

    def name(self):
        return 'Sigmoid'

    def activation(self, x):
        return 1. / (1. + math.exp(-x))

    def derivative(self, x):
        a = self.activation(x)
        return a * (1. - a)


class TanH(Activation):

    def name(self):
        return 'Hyperbolic Tangent'

    def activation(self, x):
        return math.tanh(x)

    def derivative(self, x: float) -> float:
        return 1. - math.tanh(x) ** 2


class ReLU(Activation):

    def name(self):
        return 'Rectified Linear Unit (ReLU)'

    def activation(self, x):
        return max(0., x)

    def derivative(self, x):
        if x >= 0:
            return 1.
        else:
            return 0.


class LeakyReLU(Activation):

    def __init__(self, alpha: float = 0.2):
        self.__alpha = alpha

    @property
    def alpha(self):
        return self.__alpha

    def name(self):
        return 'Leaky Rectified Linear Unit (ReLU)'

    def activation(self, x):
        return max(self.alpha * x, x)

    def derivative(self, x):
        if x >= 0:
            return 1.
        else:
            return self.alpha


class ELU(Activation):

    def __init__(self, alpha: float = 1.0):
        self.__alpha = alpha

    @property
    def alpha(self):
        return self.__alpha

    def name(self):
        return 'Exponential Linear Unit (ELU)'

    def activation(self, x):
        if x >= 0.:
            return x
        else:
            return self.alpha * (math.exp(x) - 1.)

    def derivative(self, x):
        if x >= 0.:
            return 1
        else:
            return self.alpha * math.exp(x)
