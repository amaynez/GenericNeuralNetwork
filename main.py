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

learning_rounds = 64

fig = plt.figure()
fig.canvas.set_window_title('Learning XOR Algorithm')
fig.set_size_inches(9, 6)
x = np.linspace(0, 1, num_surface_points)
y = np.linspace(0, 1, num_surface_points)
x, y = np.meshgrid(x, y)

def animate(t):
    # training
    for i in range(learning_rounds):
        j = np.random.randint(4)
        NN.train(training_data[j:j + 1, :2].reshape(2), training_data[j:j + 1, 2:].reshape(1))

    fig.clear()
    fig.suptitle('Learned ' +
                 str(learning_rounds * t),
                 fontsize=12
                 )
    axs_ = Axes3D(fig)
    z_ = np.array(NN.forward_propagation([x, y])).reshape(num_surface_points, num_surface_points)
    surface = axs_.plot_surface(x, y, z_, rstride=1, cstride=1, cmap='viridis', vmin=0, vmax=1, antialiased=True)
    axs_.set_xticks([0, 0.25, 0.5, 0.75, 1])
    axs_.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axs_.set_zticks([0, 0.25, 0.5, 0.75, 1])
    axs_.xaxis.set_ticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=8, color='.5')
    axs_.yaxis.set_ticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=8, color='.5')
    axs_.zaxis.set_ticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=8, color='.5')
    axs_.set_xlabel('Input 1', fontsize=10, color='.25')
    axs_.set_ylabel('Input 2', fontsize=10, color='.25')
    axs_.set_zlabel('Predicted result', fontsize=10, color='.25')
    fig.colorbar(surface, shrink=0.7, aspect=20, pad=0.12)


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


z = np.array(NN.forward_propagation([x, y])).reshape(num_surface_points**2)
fig = plt.figure()
fig.canvas.set_window_title('Learning XOR Algorithm')
fig.suptitle('Final XOR plot')
fig.set_size_inches(9, 6)
axs = fig.add_subplot(1, 1, 1)
axs.axis("off")
scatter = axs.scatter(x, y, marker='o', s=40, c=z.astype(float), cmap='viridis', vmin=0, vmax=1)
fig.colorbar(scatter, shrink=0.7, aspect=20, pad=0.12)
plt.show()
