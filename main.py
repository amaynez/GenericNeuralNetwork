# Fully connected Generic Neural Network with sigmoid activation for supervised learning
#
# This program creates a neural network programmatically with the following parameters:
# - number of inputs
# - number of neurons in hidden layer 1, ..., number of neurons in hidden layer n
# - number of outputs
# - learning rate
#
# Once created the Neural Network has two functions:
# - Forward Propagation: to generate a prediction or guess based on the inputs
# - Train: to modify the inner weights and biases based on given inputs and target outputs
#
# For testing purposes the XOR algorithm is implemented in the main.py script

import numpy as np
import Neural_Network as nn

inputs = 2
hidden_layers = [32]
outputs = 1
learning_rate = 0.1

NN = nn.NeuralNetwork(inputs, hidden_layers, outputs, learning_rate)

training_data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

print('before training:')
print('0 XOR 0: ', np.round(NN.forward_propagation([0, 0]), 0).reshape(1))
print('0 XOR 1: ', np.round(NN.forward_propagation([0, 1]), 0).reshape(1))
print('1 XOR 0: ', np.round(NN.forward_propagation([1, 0]), 0).reshape(1))
print('1 XOR 1: ', np.round(NN.forward_propagation([1, 1]), 0).reshape(1))

# training
for i in range(10000):
    j = np.random.randint(4)
    NN.train(training_data[j:j+1, :2].reshape(2), training_data[j:j+1, 2:].reshape(1))

print('\nafter training:')
print('0 XOR 0: ', np.round(NN.forward_propagation([0, 0]), 0).reshape(1))
print('0 XOR 1: ', np.round(NN.forward_propagation([0, 1]), 0).reshape(1))
print('1 XOR 0: ', np.round(NN.forward_propagation([1, 0]), 0).reshape(1))
print('1 XOR 1: ', np.round(NN.forward_propagation([1, 1]), 0).reshape(1))

