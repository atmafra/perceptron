from activation import *
from perceptron import Perceptron

T = 1.0
F = 0.0
inputs = [[F, F], [T, F], [F, T], [T, T]]
dataset_and = {'inputs': inputs, 'outputs': [F, F, F, T]}
dataset_or = {'inputs': inputs, 'outputs': [F, T, T, T]}
dataset_1 = {'inputs': inputs, 'outputs': [F, T, F, T]}
dataset_not1 = {'inputs': inputs, 'outputs': [T, F, T, F]}
dataset_2 = {'inputs': inputs, 'outputs': [F, F, T, T]}
dataset_not2 = {'inputs': inputs, 'outputs': [T, T, F, F]}
dataset_xor = {'inputs': inputs, 'outputs': [F, T, T, F]}

if __name__ == '__main__':
    training_dataset = dataset_and
    perceptron = Perceptron(input_dimension=2, activation_function=LeakyReLU())

    print('\n--> Before training')
    perceptron.evaluate(dataset=training_dataset)

    training_log = perceptron.train(dataset=training_dataset, epochs=1000, learning_rate=0.01)

    print('\n--> After training')
    perceptron.evaluate(dataset=training_dataset)

    print()
    print('Final loss: {:0.6f}'.format(perceptron.loss))
    print('Final weights:', perceptron.weights)

    # training_log.plot_loss(log_scale=False)
    # training_log.plot_bias()
    # training_log.plot_weights()
    # training_log.plot_weights_3d()
    training_log.plot_weights_loss_3d()
