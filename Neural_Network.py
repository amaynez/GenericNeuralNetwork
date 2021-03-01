import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, learning_rate):
        self.inputs = inputs
        if isinstance(hidden, int):
            hidden = [hidden]
        self.hidden = np.array(hidden)
        self.outputs = outputs
        self.learning_rate = learning_rate

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
        hidden_results.append(sigmoid(np.matmul(self.weights[0], input_values) + self.bias[0]))

        # calculate results for subsequent hidden layers if any (depending on the previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                hidden_results.append(sigmoid(np.matmul(self.weights[idx + 1],
                                                        hidden_results[idx]) +
                                              self.bias[idx + 1]))

        # calculate final result and return, if explicit is set then return all the intermediate results as well
        output = []
        if 'explicit' in kwargs.keys():
            if kwargs.get('explicit') in ['yes', 'y', 1]:
                output = hidden_results
        output.append(sigmoid(np.matmul(self.weights[-1], hidden_results[-1]) + self.bias[-1]))
        return output

    def train(self, inputs, targets, batch_size):
        # get the results including the hidden layers' (intermediate results)
        results = self.forward_propagation(inputs, explicit='yes')

        # prepare the targets and inputs for matrix operations
        targets = np.array(targets)[np.newaxis].T
        input_values = np.array(inputs)[np.newaxis].T

        # calculate the derivative error_matrix (targets vs outputs), index 0
        d_error_matrix = [((targets - results[-1]) * d_sigmoid(results[-1]))/batch_size]

        # calculate the derivative error_matrix of the hidden layers from last to first but insert in the correct order
        for idx in range(len(results) - 1, 0, -1):
            d_error_matrix.insert(0, np.matmul(self.weights[idx].T, d_error_matrix[0] * d_sigmoid(results[idx])))

        # calculate the gradient for all hidden layers
        for idx, weight_col in enumerate(self.weights[1:]):
            weight_col -= self.learning_rate * np.matmul(d_error_matrix[idx + 1], results[idx].T)
            self.bias[idx + 1] -= self.learning_rate * d_error_matrix[idx + 1]

        # calculate the gradient for the first layer weights (input -> first hidden layer)
        self.weights[0] -= self.learning_rate * np.matmul(d_error_matrix[0], input_values.T)
        self.bias[0] -= self.learning_rate * d_error_matrix[0]
