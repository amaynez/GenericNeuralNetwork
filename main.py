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
from matplotlib import cm
import Neural_Network as nn
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

inputs = 2
hidden_layers = [32, 24, 16, 8]
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

learning_rounds = 8

fig = plt.figure()
axs = Axes3D(fig)
fig.canvas.set_window_title('Learning XOR Algorithm')
fig.set_size_inches(9, 6)
# axs.spines['top'].set_color('none')
# axs.spines['bottom'].set_color('none')
# axs.spines['left'].set_color('none')
# axs.spines['right'].set_color('none')
# axs.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

def animate(t):
    # training
    for i in range(learning_rounds):
        j = np.random.randint(4)
        NN.train(training_data[j:j + 1, :2].reshape(2), training_data[j:j + 1, 2:].reshape(1))

    fig.suptitle('Learned ' +
                 str(learning_rounds * t),
                 fontsize=12
                 )

    axs.clear()
    num_surface_points = 64
    x = np.linspace(0, 1, num_surface_points)
    y = np.linspace(0, 1, num_surface_points)
    x, y = np.meshgrid(x, y)
    z = np.array(NN.forward_propagation([x, y])).reshape(num_surface_points, num_surface_points)
    axs.plot_surface(x, y, z,  rstride=1, cstride=1, cmap='viridis', antialiased=True)
    axs.set_zlim(0, 1.01)
    axs.zaxis.set_major_locator(LinearLocator(10))
    axs.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()

print('\nafter training without rounding:')
print('0 XOR 0: ', np.round(NN.forward_propagation([0, 0]), 0).reshape(1))
print('0 XOR 1: ', np.round(NN.forward_propagation([0, 1]), 0).reshape(1))
print('1 XOR 0: ', np.round(NN.forward_propagation([1, 0]), 0).reshape(1))
print('1 XOR 1: ', np.round(NN.forward_propagation([1, 1]), 0).reshape(1))

print('\nafter training:')
print('0 XOR 0: ', NN.forward_propagation([0, 0]))
print('0 XOR 1: ', NN.forward_propagation([0, 1]))
print('1 XOR 0: ', NN.forward_propagation([1, 0]))
print('1 XOR 1: ', NN.forward_propagation([1, 1]))
