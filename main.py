import numpy as np
import Neural_Network as nn

NN = nn.NeuralNetwork(728, [16, 16], 8)

inputs = np.arange(728)

print(NN.forward_propagation(inputs), '\n')
# for weights in NN.weights:
#     print(weights, '\n')
# print(NN.bias, '\n')
