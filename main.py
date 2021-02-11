import numpy as np
import Neural_Network as nn

inputs = 5
hidden_layers = [10, 2, 2]
outputs = 1

NN = nn.NeuralNetwork(inputs, hidden_layers, outputs)

inputs_fp = np.arange(inputs)

print(NN.forward_propagation(inputs_fp), '\n')
# for weights in NN.weights:
#     print(weights, '\n')
# print(NN.bias, '\n')
