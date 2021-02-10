import numpy as np
import Neural_Network as nn

NN = nn.NeuralNetwork(2, 2, 1)

inputs = [2, 3]

print(NN.forward_propagation(inputs), '\n')
for weights in NN.weights:
    print(weights, '\n')
print(NN.output_weights, '\n')
print(NN.bias, '\n')
