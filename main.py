import numpy as np
import Neural_Network as nn

inputs = 2
hidden_layers = [3, 2]
outputs = 2
learning_rate = 0.1

NN = nn.NeuralNetwork(inputs, hidden_layers, outputs, learning_rate)

inputs_data = np.arange(inputs)
targets_data = np.arange(outputs)
print(NN.forward_propagation(inputs_data), '\n')

NN.train(inputs_data, targets_data)
# for weights in NN.weights:
#     print(weights, '\n')
# print(NN.bias, '\n')
