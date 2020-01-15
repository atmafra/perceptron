import numpy as np

from activation import Activation
from training_log import TrainingSample, TrainingLog

decimal_format: str = '{: > 7.4f}'
decimal_formatter = {'float': decimal_format.format}


class Perceptron:
    """ Implements a Perceptron neuron model
    """

    def __init__(self, input_dimension: int,
                 activation_function: Activation):
        """ Instantiates a new Perceptron

        Args:
            input_dimension (int): dimension of the input vector, number of input connections
            activation_function (Activation): activation function

        """
        # status variables
        self.__input_dimension: int = input_dimension
        self.__activation_function: Activation = activation_function
        self.__input = np.zeros(shape=(self.input_dimension,))
        self.__weights = np.zeros(shape=(self.input_dimension + 1,))
        self.__activation: float = 0.
        self.__output: float = 0.

        # learning variables
        self.__error: float = 0.
        self.__loss: float = 0.
        self.__gradient: np.array = np.zeros(self.input_dimension)

        self.initialize()

    @property
    def input_dimension(self):
        return self.__input_dimension

    @property
    def activation_function(self):
        return self.__activation_function

    @property
    def input(self):
        return self.__input

    @input.setter
    def input(self, input_vector: list):
        if input_vector is None:
            raise ValueError('null input vector')
        if len(input_vector) != self.input_dimension:
            raise ValueError('input vector dimension ({}) differs from perceptron input dimension ({})'
                             .format(len(input_vector), self.input_dimension))
        self.__input = np.array(input_vector)
        self.update_activation()

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights: np.array):
        if weights is None:
            raise ValueError('no weight vector passed')
        if len(weights) != self.input_dimension + 1:
            raise ValueError('weight vector dimension ({}) is incompatible with input dimension ({})'
                             .format(len(weights), self.input_dimension))
        self.__weights = weights

    @property
    def bias(self):
        return self.weights[0]

    @property
    def activation(self):
        return self.__activation

    @activation.setter
    def activation(self, activation: float):
        self.__activation = activation
        self.update_output()

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, output: float):
        self.__output = output

    @property
    def error(self):
        return self.__error

    @error.setter
    def error(self, error: float):
        self.__error = error
        self.update_loss()
        self.update_gradient()

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss: float):
        self.__loss = loss

    @property
    def gradient(self):
        return self.__gradient

    @gradient.setter
    def gradient(self, gradient: np.array):
        self.__gradient = gradient

    def initialize_weights(self):
        """ Attributes random initial values for the weights and the bias
        """
        self.weights = np.random.normal(size=self.input_dimension + 1)

    def initialize(self):
        """ Puts the perceptron in a consistent initial state
        """
        self.initialize_weights()
        self.input = np.zeros(self.input_dimension)

    def update_activation(self):
        """ Updates the activation potential
        """
        # the 0-th dimension is padded with a '1' because the 0-th weight is the bias
        x = np.concatenate((np.ones(1), self.input))
        w = self.weights
        self.activation = np.float(np.dot(w, x))

    def update_output(self):
        """Updates the output value by applying the activation function to the activation potential
        """
        # self.output = math.tanh(self.activation)
        self.output = self.activation_function.activation(x=self.activation)

    def propagate_forward(self, x: list) -> float:
        """ Applies the input vector and propagates the signal forward

        Args:
            x (np.array): input vector

        Return:
            the neuron output value

        """
        self.input = x
        return self.output

    def update_error(self, y_desired: float):
        """ Update the value of the error with the current output
        """
        self.error = self.output - y_desired

    def update_loss(self):
        """ Updates the value of the loss from the error, applying the loss function
        """
        self.loss = self.error ** 2

    def update_gradient(self):
        """ Update the gradient vector applying the chain rule
        """
        d_loss_d_output = 2. * self.error
        d_output_d_activation = self.activation_function.derivative(x=self.activation)
        d_activation_d_weights = np.concatenate((np.array([1.]), self.input))
        self.gradient = d_loss_d_output * d_output_d_activation * d_activation_d_weights

    def propagate_backward(self, y: float, learning_rate: float):
        """ Executes one step of error propagation backwards, updating the weights
        """
        self.update_error(y_desired=y)
        self.weights -= learning_rate * self.gradient

    def train_element(self, x: list, y: list, learning_rate: float):
        """ Executes one step of training, for one element from the training set

        Args:
            x (np.array): input vector
            y (float): desired output value
            learning_rate (float): learning rate for backpropagation

        """
        self.propagate_forward(x=x)
        self.propagate_backward(y=y, learning_rate=learning_rate)

    def epoch(self, dataset: dict, learning_rate: float) -> float:
        """ Executes one pass of all the elements in the dataset, returning the epoch total loss

        Args:
            dataset (dict): a pair of lists, one for the input vectors and the other the desired outputs
            learning_rate (float): learning rate for backpropagation

        """
        inputs = dataset['inputs']
        outputs = dataset['outputs']
        epoch_loss = 0.
        for i in range(0, len(inputs)):
            self.train_element(x=inputs[i], y=outputs[i], learning_rate=learning_rate)
            epoch_loss += self.loss

        return epoch_loss

    def train(self, dataset: dict,
              epochs: int,
              learning_rate: float) -> TrainingLog:
        """ Execute multiple passes over the training set to minimize the loss function

        Args:
            dataset (dict): training set
            epochs (int): number of passes over the training set
            learning_rate (float): learning rate for backpropagation

        """
        training_log = TrainingLog(keys=TrainingSample.keys())
        for epoch in range(1, epochs + 1):
            average_epoch_loss = self.epoch(dataset=dataset, learning_rate=learning_rate) / len(dataset['inputs'])
            training_sample = TrainingSample(epoch=epoch, weights=self.weights.copy(), loss=average_epoch_loss)
            training_log.add_sample(training_sample)

        return training_log

    def evaluate(self, dataset: dict):
        """ Evaluates the performance of the model in the dataset, comparing desired and estimated outputs

        Args:
            dataset (dict): evaluation dataset

        """
        for input_vector, output_value in zip(dataset['inputs'], dataset['outputs']):
            output_estimated = self.propagate_forward(x=input_vector)
            self.update_error(y_desired=output_value)
            with np.printoptions(formatter=decimal_formatter):
                print('x = {}'.format(np.array(input_vector)), end=',  ')
                print('y = ' + decimal_format.format(output_value), end=',  ')
                print('y_est = ' + decimal_format.format(output_estimated), end=',  ')
                print('error = ' + decimal_format.format(self.error))

    def print_forward_status(self):
        with np.printoptions(formatter=decimal_formatter):
            print('Input       : {}'.format(self.input))
            print('Output      : ' + decimal_format.format(self.output))
            print('Weights     : {}'.format(self.weights[1:]))
            print('Bias        : ' + decimal_format.format(self.bias))
            print('Activation  : ' + decimal_format.format(self.activation))

    def print_backward_status(self):
        with np.printoptions(formatter=decimal_formatter):
            print('Error       : ' + decimal_format.format(self.error))
            print('Loss        : ' + decimal_format.format(self.loss))
            print('Gradient    : ' + decimal_format.format(self.gradient))
