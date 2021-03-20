import numpy as np
import json
import json_numpy


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
    def __init__(self, inputs, hidden, outputs, activation, learning_rate):
        self.inputs = inputs
        if isinstance(hidden, int):
            hidden = [hidden]
        self.hidden = np.array(hidden)
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.activation = activation

        # create weights for first hidden layer (defined based on number of inputs)
        self.weights = [np.random.uniform(-1, 1, size=(self.hidden[0], self.inputs))]

        # create weights for interior hidden layers (defined based on previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_col in enumerate(self.hidden[1:]):
                self.weights.append(np.random.uniform(-1, 1, size=(hidden_col, self.hidden[idx])))

        # create weights for output layer (defined based on number of outputs)
        self.weights.append(np.random.uniform(-1, 1, size=(self.outputs, self.hidden[-1],)))

        # create bias list of matrices (one per hidden layer and one for the output)
        self.bias = []
        for idx, hidden_col in enumerate(self.hidden):
            self.bias.append(np.random.uniform(-1, 1, size=(hidden_col, 1)))
        self.bias.append(np.random.uniform(-1, 1, size=(self.outputs, 1)))

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
                                                   self.bias[idx + 1], self.activation[idx+1]))

        # calculate final result and return, if explicit is set then return all the intermediate results as well
        output = []
        if 'explicit' in kwargs.keys():
            if kwargs.get('explicit') in ['yes', 'y', 1]:
                output = hidden_results
        output.append(act_function(
            np.matmul(self.weights[-1], hidden_results[-1])
            + self.bias[-1], self.activation[-1]))
        return output

    def train(self, inputs, targets):
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