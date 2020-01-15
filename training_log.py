from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytz as pytz
from mpl_toolkits.mplot3d import Axes3D


class TrainingSample:
    """ A training sample records the state of the model after a single sample is trained
    """

    def __init__(self,
                 epoch: int,
                 weights: np.array,
                 loss: float):
        """ Creates a new training sample

        Args:
            epoch (int): training epoch
            weights (np.array): weight vector after training the sample
            loss (float): value of the loss function

        """
        self.__epoch: int = epoch
        self.__timestamp: datetime = datetime.now(tz=pytz.utc)
        self.__weights: np.array = weights
        self.__loss: float = loss

    @property
    def epoch(self):
        return self.__epoch

    @property
    def timestamp(self):
        return self.__timestamp

    @property
    def weights(self) -> np.array:
        return self.__weights

    @property
    def loss(self) -> float:
        return self.__loss

    @classmethod
    def keys(cls):
        return ['epoch', 'timestamp', 'weights', 'loss']

    def as_array(self) -> np.array:
        return np.array([self.epoch, self.timestamp, self.weights, self.loss])

    def as_dict(self) -> dict:
        return {'epoch': self.epoch, 'timestamp': self.timestamp, 'weights': self.weights, 'loss': self.loss}


class TrainingLog:
    """ The log is a list of training samples
    """

    def __init__(self, keys: list):
        self.__training_log = {}
        for key in keys:
            self.training_log[key] = []

    @property
    def training_log(self):
        return self.__training_log

    @property
    def size(self):
        return len(self.training_log)

    def add_sample(self, sample: TrainingSample):
        """ Adds a new training sample to the log
        """
        if sample is None:
            raise ValueError('Empty training sample')

        sample_dict: dict = sample.as_dict()
        for key in sample_dict.keys():
            self.training_log[key].append(sample_dict[key])

    def x_series(self):
        return range(1, len(self.training_log['epoch']) + 1)

    def loss_series(self):
        return self.training_log['loss']
        # loss_history: list = []
        # for sample in self.training_log:
        #     loss_history.append(sample.loss)
        # return loss_history

    def weights_series(self):
        weights = self.training_log['weights']
        if len(weights) < 1:
            return
        count = len(weights)
        dimension = len(weights[0])
        weights = np.concatenate(weights)
        return np.reshape(weights, (count, dimension))

    def plot_loss(self, log_scale: bool = True):
        x = self.x_series()
        y = self.training_log['loss']
        plt.plot(x, y, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if log_scale:
            plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_bias(self):
        x = self.x_series()
        weights = self.weights_series()
        y = weights[:, 0]
        plt.plot(x, y, label='Bias')
        plt.title('Bias evolution')
        plt.xlabel('epoch')
        plt.ylabel('bias')
        plt.grid()
        plt.show()

    def plot_weights(self):
        weights = self.weights_series()
        plt.scatter(x=weights[:, 1], y=weights[:, 2])
        plt.title('Weights evolution')
        plt.xlabel('w1')
        plt.ylabel('w2')
        plt.grid()
        plt.show()

    def plot_weights_3d(self):
        weights = self.weights_series()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(weights[:, 1], weights[:, 2], weights[:, 0])
        ax.set_title('Weights and bias evolution')
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        ax.set_zlabel('bias')
        plt.show()

    def plot_weights_loss_3d(self):
        weights = self.weights_series()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(weights[:, 1], weights[:, 2], self.training_log['loss'])
        ax.set_title('Loss evolution')
        ax.set_xlabel('w1')
        ax.set_ylabel('w2')
        ax.set_zlabel('loss')
        plt.show()
