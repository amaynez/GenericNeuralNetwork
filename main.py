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
import random as rnd

inputs = 3
hidden_layers = [2]
outputs = 1
activation = ['linear', 'linear']
learning_rate = 0.000001
learning_pool_size = 100000
training_size = 1024
EPOCHS = 1
batch_size = 1
num_surface_points = 32  # for plotting

NN = nn.NeuralNetwork(inputs, hidden_layers, outputs, activation)
NN.set_optimizer('ADAM', beta1=0.9, beta2=0.999, epsilon=.0001, ADAM_bias_correction=True)
NN.set_learning_rate(learning_rate, decay_rate=0.0001, cycling=True, max_lr=0.1, cycle=256)

training_data = []
for n in range(learning_pool_size):
    x__ = rnd.random()
    y__ = rnd.random()
    training_data.append([x__, y__, x__ * y__, 0 if (x__ < 0.5 and y__ < 0.5) or (x__ >= 0.5 and y__ >= 0.5) else 1])

print('before training:')
print('0 XOR 0: ', np.round(NN.forward_propagation([0, 0, 0]), 0)[0])
print('0 XOR 1: ', np.round(NN.forward_propagation([0, 1, 0]), 0)[0])
print('1 XOR 0: ', np.round(NN.forward_propagation([1, 0, 0]), 0)[0])
print('1 XOR 1: ', np.round(NN.forward_propagation([1, 1, 1]), 0)[0])

px = 1/72
fig = plt.figure(dpi=72, figsize=(1024*px, 600*px))
fig.canvas.set_window_title('Learning XOR Algorithm')
ticks = [0, 0.25, 0.5, 0.75, 1]
axs1 = fig.add_subplot(1, 2, 1, projection='3d')
axs2 = fig.add_subplot(1, 2, 2)

x = np.linspace(0, 1, num_surface_points)
y = np.linspace(0, 1, num_surface_points)
x, y = np.meshgrid(x, y)


def animate(t):
    # training
    batch_data = rnd.sample(training_data, training_size)
    batch_data = np.array(batch_data)

    NN.fit(t, batch_data[0:, :3], batch_data[0:, 3:], epochs=EPOCHS, batch_size=batch_size, shuffle=False, verbose=False)

    fig.suptitle('Learning iteration: ' + str(t) +
                 ' ; total points learned: ' +
                 str(training_size * t), fontsize=14)

    z = np.array(NN.forward_propagation([x, y, x * y])).reshape(num_surface_points, num_surface_points)

    axs1.clear()
    axs1.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', vmin=0, vmax=1, antialiased=True)
    axs1.set_xticks(ticks)
    axs1.set_yticks(ticks)
    axs1.set_zticks(ticks)
    axs1.xaxis.set_ticklabels(ticks, fontsize=10, color='.5')
    axs1.yaxis.set_ticklabels(ticks, fontsize=10, color='.5')
    axs1.zaxis.set_ticklabels(ticks, fontsize=10, color='.5')
    axs1.set_xlabel('Input 1', fontsize=11, color='.25')
    axs1.set_ylabel('Input 2', fontsize=11, color='.25')
    axs1.set_zlabel('Predicted result', fontsize=11, color='.25')

    z = z.reshape(num_surface_points ** 2)

    axs2.clear()
    axs2.axis("off")
    scatter = axs2.scatter(x, y,
                           marker='o',
                           s=40,
                           c=z.astype(float),
                           cmap='viridis',
                           vmin=0,
                           vmax=1
                           )


ani = animation.FuncAnimation(fig, animate, interval=100, save_count=150)
# with open("XOR_video.html", "w") as f:
#     print(ani.to_html5_video(), file=f)
plt.show()

print('\nafter training:')
print('0 XOR 0: ', np.round(NN.forward_propagation([0, 0, 0]), 0).reshape(outputs))
print('0 XOR 1: ', np.round(NN.forward_propagation([0, 1, 0]), 0).reshape(outputs))
print('1 XOR 0: ', np.round(NN.forward_propagation([1, 0, 0]), 0).reshape(outputs))
print('1 XOR 1: ', np.round(NN.forward_propagation([1, 1, 1]), 0).reshape(outputs))

print('\nafter training without rounding:')
print('0 XOR 0: ', NN.forward_propagation([0, 0, 0]))
print('0 XOR 1: ', NN.forward_propagation([0, 1, 0]))
print('1 XOR 0: ', NN.forward_propagation([1, 0, 0]))
print('1 XOR 1: ', NN.forward_propagation([1, 1, 1]))
