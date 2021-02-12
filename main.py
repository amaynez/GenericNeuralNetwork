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
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

inputs = 2
hidden_layers = [32, 24, 16, 8]
outputs = 1
learning_rate = 0.1
num_surface_points = 32

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

learning_rounds = 8

fig = plt.figure()
fig.canvas.set_window_title('Learning XOR Algorithm')
fig.set_size_inches(11, 6)
x = np.linspace(0, 1, num_surface_points)
y = np.linspace(0, 1, num_surface_points)
x, y = np.meshgrid(x, y)

def animate(t):
    # training
    for i in range(learning_rounds):
        j = np.random.randint(4)
        NN.train(training_data[j:j + 1, :2].reshape(2), training_data[j:j + 1, 2:].reshape(1))

    fig.clear()
    fig.suptitle('Learning iterations: ' +
                 str(learning_rounds * t),
                 fontsize=12
                 )
    axs1 = fig.add_subplot(1, 2, 1, projection='3d')

    z = np.array(NN.forward_propagation([x, y])).reshape(num_surface_points, num_surface_points)
    axs1.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', vmin=0, vmax=1, antialiased=True)
    axs1.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axs1.set_zticks([0, 0.25, 0.5, 0.75, 1])
    axs1.xaxis.set_ticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=8, color='.5')
    axs1.yaxis.set_ticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=8, color='.5')
    axs1.zaxis.set_ticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=8, color='.5')
    axs1.set_xlabel('Input 1', fontsize=10, color='.25')
    axs1.set_ylabel('Input 2', fontsize=10, color='.25')
    axs1.set_zlabel('Predicted result', fontsize=10, color='.25')
    axs2 = fig.add_subplot(1, 2, 2)
    axs2.axis("off")
    z = z.reshape(num_surface_points**2)
    scatter = axs2.scatter(x, y, marker='o', s=40, c=z.astype(float), cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(scatter, shrink=0.5)


ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()

print('\nafter training:')
print('0 XOR 0: ', np.round(NN.forward_propagation([0, 0]), 0).reshape(1))
print('0 XOR 1: ', np.round(NN.forward_propagation([0, 1]), 0).reshape(1))
print('1 XOR 0: ', np.round(NN.forward_propagation([1, 0]), 0).reshape(1))
print('1 XOR 1: ', np.round(NN.forward_propagation([1, 1]), 0).reshape(1))

print('\nafter training without rounding:')
print('0 XOR 0: ', NN.forward_propagation([0, 0]))
print('0 XOR 1: ', NN.forward_propagation([0, 1]))
print('1 XOR 0: ', NN.forward_propagation([1, 0]))
print('1 XOR 1: ', NN.forward_propagation([1, 1]))
