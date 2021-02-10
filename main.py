import numpy as np
import Neural_Network as nn

NN = nn.NeuralNetwork(2, 2, 1)

inputs = [2, 3]

result = NN.forward_propagation(inputs)

