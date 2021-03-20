import numpy as np
import math
import json
import json_numpy
import random
import time


def act_function(x, function):
    if function in ['sigmoid', 'Sigmoid', 'SIGMOID']:
        return 1 / (1 + np.exp(-x))
    elif function in ['ReLU', 'relu', 'RELU']:
        return x * (x > 0)
    elif function in ['linear', 'Linear', 'line']:
        return x
    elif function in ['tanh', 'TanH', 'tanH', 'Tanh']:
        return (2 / (1 + np.exp(-2 * x))) - 1


def d_act_function(x, function):
    if function in ['sigmoid', 'Sigmoid', 'SIGMOID']:
        return x * (1 - x)
    elif function in ['ReLU', 'relu', 'RELU']:
        return 1 * (x > 0)
    elif function in ['linear', 'Linear', 'line']:
        return np.ones_like(x)
    elif function in ['tanh', 'TanH', 'tanH', 'Tanh']:
        return 1 - x ** 2


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, activation):
        self.inputs = inputs
        if isinstance(hidden, int):
            hidden = [hidden]
        self.hidden = np.array(hidden)
        self.outputs = outputs
        self.learning_rate = 0.01
        self.activation = activation
        self.optimizer = 'vanilla'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = math.pow(10, -7)
        self.cycling = False
        self.max_lr = 0.1
        self.cycle = 1000
        self.decay_rate = 0.0001
        self.nag_coefficient = 0.9
        self.ADAM_bias_correction = True

        # create weights and gradients for first hidden layer (defined based on number of inputs)
        self.weights = [np.random.uniform(-1, 1, size=(self.hidden[0], self.inputs)).astype(np.float128)]
        self.gradients = [np.zeros((self.hidden[0], self.inputs), np.float128)]

        # create weights and gradients for interior hidden layers (defined based on previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_col in enumerate(self.hidden[1:]):
                self.weights.append(np.random.uniform(-1, 1, size=(hidden_col, self.hidden[idx])).astype(np.float128))
                self.gradients.append(np.zeros((hidden_col, self.hidden[idx]), np.float128))

        # create weights and gradients for output layer (defined based on number of outputs)
        self.weights.append(np.random.uniform(-1, 1, size=(self.outputs, self.hidden[-1],)).astype(np.float128))
        self.gradients.append(np.zeros((self.outputs, self.hidden[-1],), np.float128))

        # create bias and bias_gradients lists of matrices (one per hidden layer and one for the output)
        self.bias = []
        self.bias_gradients = []
        for idx, hidden_col in enumerate(self.hidden):
            self.bias.append(np.random.uniform(-1, 1, size=(hidden_col, 1)).astype(np.float128))
            self.bias_gradients.append(np.zeros((hidden_col, 1), np.float128))
        self.bias.append(np.random.uniform(-1, 1, size=(self.outputs, 1)).astype(np.float128))
        self.bias_gradients.append(np.zeros((self.outputs, 1), np.float128))

        # Create the dictionary variables for the optimization methods
        self.v = {}
        self.s = {}
        for i in range(self.hidden.size + 1):
            self.v["dW" + str(i)] = 0
            self.v["db" + str(i)] = 0
            self.s["dW" + str(i)] = 0
            self.s["db" + str(i)] = 0

    def set_optimizer(self, optimizer='vanilla', **kwargs):
        '''
        :param optimizer: 'vanilla', 'SGD_momentum', 'NAG', 'RMSProp', 'ADAM'
        :param kwargs: beta1, beta2, epsilon, nag_coefficient, ADAM_bias_correction
        '''
        self.optimizer = optimizer
        if 'beta1' in kwargs.keys():
            self.beta1 = float(kwargs.get('beta1'))
        if 'beta2' in kwargs.keys():
            self.beta2 = float(kwargs.get('beta2'))
        if 'epsilon' in kwargs.keys():
            self.epsilon = float(kwargs.get('epsilon'))
        if 'nag_coefficient' in kwargs.keys():
            self.nag_coefficient = float(kwargs.get('nag_coefficient'))
        if 'ADAM_bias_correction' in kwargs.keys():
            self.ADAM_bias_correction = bool(kwargs.get('ADAM_bias_correction'))

    def set_learning_rate(self, learning_rate=0.01, **kwargs):
        '''
        :param learning_rate: float
        :param kwargs: cycling, max_lr, cycle, decay_rate
        '''
        self.learning_rate = learning_rate
        if 'cycling' in kwargs.keys():
            self.cycling = bool(kwargs.get('cycling'))
        if 'max_lr' in kwargs.keys():
            self.max_lr = float(kwargs.get('max_lr'))
        if 'cycle' in kwargs.keys():
            self.cycle = int(kwargs.get('cycle'))
        if 'decay_rate' in kwargs.keys():
            self.decay_rate = float(kwargs.get('decay_rate'))

    def gradient_zeros(self):
        self.gradients = [np.zeros((self.hidden[0], self.inputs), np.float128)]
        self.bias_gradients = [np.zeros((self.hidden[0], 1), np.float128)]
        if self.hidden.ndim > 0:
            for idx, hidden_col in enumerate(self.hidden[1:]):
                self.gradients.append(np.zeros((hidden_col, self.hidden[idx]), np.float128))
                self.bias_gradients.append(np.zeros((hidden_col, 1), np.float128))
        self.gradients.append(np.zeros((self.outputs, self.hidden[-1]), np.float128))
        self.bias_gradients.append(np.zeros((self.outputs, 1), np.float128))

    def copy_from(self, neural_net):
        self.weights = neural_net.weights
        self.bias = neural_net.bias

    def forward_propagation(self, input_values, **kwargs):
        # create hidden results list for results matrices per hidden layer
        hidden_results = []

        # prepare the input values for matrix multiplication
        input_values = np.array(input_values)[np.newaxis].T

        # calculate results for the first hidden layer (depending on the inputs)
        hidden_results.append(act_function(np.matmul(self.weights[0], input_values) + self.bias[0], self.activation[0]))

        # calculate results for subsequent hidden layers if any (depending on the previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                hidden_results.append(act_function(np.matmul(self.weights[idx + 1],
                                                             hidden_results[idx]) +
                                                   self.bias[idx + 1], self.activation[idx + 1]))

        # calculate final result and return, if explicit is set then return all the intermediate results as well
        output = []
        if 'explicit' in kwargs.keys():
            if kwargs.get('explicit') in ['yes', 'y', 1]:
                output = hidden_results
        output.append(act_function(
            np.matmul(self.weights[-1], hidden_results[-1])
            + self.bias[-1], self.activation[-1]))
        return output

    def train_once(self, inputs, targets):
        # get the results including the hidden layers' (intermediate results)
        results = self.forward_propagation(inputs, explicit='yes')

        # prepare the targets and inputs for matrix operations
        targets = np.array(targets)[np.newaxis].T
        input_values = np.array(inputs)[np.newaxis].T

        # calculate the error (outputs vs targets), index 0
        error = [results[-1] - targets]

        # calculate the error of the hidden layers from last to first but insert in the correct order
        for idx in range(len(results) - 2, -1, -1):
            error.insert(0, np.matmul(self.weights[idx + 1].T, error[0]))

        # modify weights and biases (input -> first hidden layer)
        self.weights[0] -= np.matmul((error[0] * d_act_function(results[0], self.activation[0])
                                      * self.learning_rate), input_values.T)
        self.bias[0] -= ((error[0]
                          * d_act_function(results[0], self.activation[0]))
                         * self.learning_rate)

        # modify weights and biases (all subsequent hidden layers and output)
        for idx, weight_cols in enumerate(self.weights[1:]):
            weight_cols -= np.matmul((error[idx + 1]
                                      * d_act_function(results[idx + 1], self.activation[idx + 1])
                                      * self.learning_rate),
                                     results[idx].T)
            self.bias[idx + 1] -= ((error[idx + 1]
                                    * d_act_function(results[idx + 1], self.activation[idx + 1]))
                                   * self.learning_rate)

    def calculate_gradient(self, inputs, targets, batch_size):
        # get the results including the hidden layers' (intermediate results)
        results = self.forward_propagation(inputs, explicit='yes')

        # prepare the targets and inputs for matrix operations
        input_values = np.array(inputs)[np.newaxis].T
        targets = np.array(targets)[np.newaxis].T

        # calculate the error (outputs vs targets), index 0
        error = [(results[-1] - targets) / batch_size]

        loss = (np.sum((targets - results[-1]) ** 2) / len(targets))

        # calculate the error of the hidden layers from last to first but insert in the correct order
        for idx in range(len(results) - 2, -1, -1):
            error.insert(0, np.matmul(self.weights[idx + 1].T, error[0]))

        # modify weights and biases gradients (input -> first hidden layer)
        self.gradients[0] += np.matmul((error[0] * d_act_function(results[0], self.activation[0])), input_values.T)
        self.bias_gradients[0] += (error[0] * d_act_function(results[0], self.activation[0]))

        # modify weights and biases gradients (all subsequent hidden layers and output)
        for idx, gradient_cols in enumerate(self.gradients[1:-1]):
            gradient_cols += np.matmul((error[idx + 1]
                                        * d_act_function(results[idx + 1], self.activation[idx + 1])),
                                       results[idx].T)
            self.bias_gradients[idx + 1] += (error[idx + 1]
                                             * d_act_function(results[idx + 1], self.activation[idx + 1]))

        self.gradients[-1] += np.matmul((error[-1]
                                         * d_act_function(results[-1], self.activation[-1])),
                                        results[-2].T)
        self.bias_gradients[-1] += (error[-1] * d_act_function(results[-1], self.activation[-1]))

        return loss

    def cyclic_learning_rate(self, learning_rate, epoch):
        cycle = np.floor(1 + (epoch / (2 * self.cycle)))
        x = np.abs((epoch / self.cycle) - (2 * cycle) + 1)
        return learning_rate + (self.max_lr - learning_rate) * np.maximum(0, (1 - x))

    def apply_gradients(self, iteration, batch_size):
        eta = self.learning_rate * (1 / (1 + self.decay_rate * iteration))

        if self.cycling:
            eta = self.cyclic_learning_rate(eta, iteration)

        for i, weight_col in enumerate(self.weights):

            if self.optimizer == 'vanilla':
                weight_col -= eta * np.array(self.gradients[i]) / batch_size
                self.bias[i] -= eta * np.array(self.bias_gradients[i]) / batch_size

            elif self.optimizer == 'SGD_momentum':
                self.v["dW" + str(i)] = ((self.beta1 * self.v["dW" + str(i)])
                                         + (eta * np.array(self.gradients[i])))
                self.v["db" + str(i)] = ((self.beta1 * self.v["db" + str(i)])
                                         + (eta * np.array(self.bias_gradients[i])))

                weight_col -= self.v["dW" + str(i)] / batch_size
                self.bias[i] -= self.v["db" + str(i)] / batch_size

            elif self.optimizer == 'NAG':
                v_prev = {"dW" + str(i): self.v["dW" + str(i)], "db" + str(i): self.v["db" + str(i)]}

                self.v["dW" + str(i)] = (self.nag_coefficient * self.v["dW" + str(i)]
                                         - eta * np.array(self.gradients[i]))
                self.v["db" + str(i)] = (self.nag_coefficient * self.v["db" + str(i)]
                                         - eta * np.array(self.bias_gradients[i]))

                weight_col += -1 * ((self.beta1 * v_prev["dW" + str(i)])
                                    + (1 + self.beta1) * self.v["dW" + str(i)]) / batch_size
                self.bias[i] += ((-1 * self.beta1 * v_prev["db" + str(i)])
                                 + (1 + self.beta1) * self.v["db" + str(i)]) / batch_size

            elif self.optimizer == 'RMSProp':
                self.s["dW" + str(i)] = ((self.beta1 * self.s["dW" + str(i)])
                                         + ((1 - self.beta1) * (np.square(np.array(self.gradients[i])))))
                self.s["db" + str(i)] = ((self.beta1 * self.s["db" + str(i)])
                                         + ((1 - self.beta1) * (np.square(np.array(self.bias_gradients[i])))))

                weight_col -= (eta * (np.array(self.gradients[i])
                                      / (np.sqrt(self.s["dW" + str(i)] + self.epsilon)))) / batch_size
                self.bias[i] -= (eta * (np.array(self.bias_gradients[i])
                                        / (np.sqrt(self.s["db" + str(i)] + self.epsilon)))) / batch_size

            if self.optimizer == "ADAM":
                # decaying averages of past gradients
                self.v["dW" + str(i)] = ((self.beta1 * self.v["dW" + str(i)])
                                         + ((1 - self.beta1) * np.array(self.gradients[i])))
                self.v["db" + str(i)] = ((self.beta1 * self.v["db" + str(i)])
                                         + ((1 - self.beta1) * np.array(self.bias_gradients[i])))

                # decaying averages of past squared gradients
                self.s["dW" + str(i)] = ((self.beta2 * self.s["dW" + str(i)])
                                         + ((1 - self.beta2) * (np.square(np.array(self.gradients[i])))))
                self.s["db" + str(i)] = ((self.beta2 * self.s["db" + str(i)])
                                         + ((1 - self.beta2) * (np.square(np.array(self.bias_gradients[i])))))

                if self.ADAM_bias_correction:
                    # bias-corrected first and second moment estimates
                    self.v["dW" + str(i)] = self.v["dW" + str(i)] / (1 - (self.beta1 ** true_epoch))
                    self.v["db" + str(i)] = self.v["db" + str(i)] / (1 - (self.beta1 ** true_epoch))
                    self.s["dW" + str(i)] = self.s["dW" + str(i)] / (1 - (self.beta2 ** true_epoch))
                    self.s["db" + str(i)] = self.s["db" + str(i)] / (1 - (self.beta2 ** true_epoch))

                # apply to weights and biases
                weight_col -= ((eta * (self.v["dW" + str(i)]
                                       / (np.sqrt(self.s["dW" + str(i)]) + self.epsilon)))) / batch_size
                self.bias[i] -= ((eta * (self.v["db" + str(i)]
                                         / (np.sqrt(self.s["db" + str(i)]) + self.epsilon)))) / batch_size

        self.gradient_zeros()

    def fit(self, inputs, labels, **kwargs):
        """

        :param inputs: a list of inputs
        :param labels: a list of corresponding labels (same size as inputs)
        :param kwargs: epochs = 1, batch_size = 8, shuffle = True
        :return: loss as a float
        """
        epochs = 1
        if 'epochs' in kwargs.keys():
            epochs = int(kwargs.get('epochs'))

        batch_size = 8
        if 'batch_size' in kwargs.keys():
            batch_size = int(kwargs.get('batch_size'))

        shuffle = True
        if 'shuffle' in kwargs.keys():
            shuffle = bool(kwargs.get('shuffle'))

        loss = 0
        iteration = 1

        for i in range(math.ceil(len(inputs)/batch_size)):
            if batch_size*(i+1) > len(inputs):
                training_data = inputs[(batch_size * i):-1]
                training_targets = labels[(batch_size * i):-1]
            else:
                training_data = inputs[(batch_size*i):(batch_size*(i+1))]
                training_targets = labels[(batch_size*i):(batch_size*(i+1))]
            for j in range(epochs):
                print('Epoch ', j, '/', epochs, ' [', end='')
                if shuffle:
                    random.shuffle(training_data)
                    random.shuffle(training_targets)
                ti = time.perf_counter()
                for k in range(len(training_data)):
                    loss = self.calculate_gradient(training_data[k], training_targets[k], batch_size)
                    if k % (math.floor(batch_size/16)):
                        print('-', end='')
                self.apply_gradients(iteration, batch_size)
                tf = time.perf_counter()
                print(f"] {tf - ti:0.4f}s", ' | loss: ', loss)
                iteration += 1

    def save_to_file(self, file_name='NeuralNet.json'):
        json_file = {
            'weights': self.weights,
            'biases': self.bias}
        try:
            with open(file_name, 'w') as file:
                json.dump(
                    json_file,
                    file,
                    ensure_ascii=False,
                    cls=json_numpy.EncodeFromNumpy)
                print('weights saved to file')
        except:
            print('cannot save to ', file)

    def load_from_file(self, file_name='NeuralNet.json'):
        try:
            with open(file_name) as file:
                json_file = json.load(file, cls=json_numpy.DecodeToNumpy)
                print('weights loaded from file')
                self.weights = json_file['weights']
                self.bias = json_file['biases']
        except:
            print('cannot open ', file_name)
