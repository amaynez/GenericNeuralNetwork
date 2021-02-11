import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        if isinstance(hidden, int):
            hidden = [hidden]
        self.hidden = np.array(hidden)
        self.outputs = outputs

        # create weights for first hidden layer (defined based on number of inputs)
        self.weights = [np.random.uniform(-1, 1, size=(self.hidden[0], self.inputs))]

        # create weights for interior hidden layers (defined based on previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_col in enumerate(self.hidden[1:]):
                self.weights.append(np.random.uniform(-1, 1, size=(hidden_col, self.hidden[idx])))

        # create weights for output layer (defined based on number of outputs)
        self.weights.append([np.random.uniform(-1, 1, size=(self.outputs, self.hidden[-1],))])

        # create bias matrices (one per hidden layer and one for the output)
        self.bias = []
        for idx, hidden_col in enumerate(self.hidden):
            self.bias.append(np.random.uniform(-1, 1, size=(hidden_col, 1)))
        self.bias.append(np.random.uniform(-1, 1, size=(self.outputs, 1)))

    def forward_propagation(self, input_values, **kwargs):
        # create hidden results matrix
        hidden_results = []

        # prepare the input values for matrix multiplication
        input_values = np.array(input_values)[np.newaxis]
        input_values = input_values.T

        # calculate results for the first hidden layer (depends on the inputs)
        hidden_results.append(sigmoid(np.matmul(self.weights[0], input_values) + self.bias[0]))

        # calculate results for subsequent hidden layers if any (depending on the previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                hidden_results.append(sigmoid(np.matmul(self.weights[idx + 1],
                                                        hidden_results[idx]) +
                                              self.bias[idx + 1]))
        # calculate final result and return
        output = []
        if 'explicit' in kwargs.keys():
            if kwargs.get('explicit') == 'yes':
                output = hidden_results
        output.append(sigmoid(np.matmul(self.weights[-1], hidden_results[-1]) + self.bias[-1]))
        return output

    def train(self, inputs, targets):
        # get the results including the hidden layers'
        results = self.forward_propagation(inputs, explicit='yes')
        # prepare the targets for matrix operations
        targets = np.array(targets)[np.newaxis]
        targets = targets.T
        # calculate the error of the outputs vs the targets, index 0
        error = [targets-results[-1]]
        print('output error:', error, '\n')

        # calculate the error of the hidden layers
        for idx, weight_matrix in enumerate(self.weights):
            print('weight_matrix:', idx, '\n', weight_matrix)
        for idx, results_matrix in enumerate(results):
            print('results matrix:', idx, '\n', results_matrix)

        for idx in range(len(results)-2, -1, -1):
            print('\n', idx)
            weights_matrix = np.array(self.weights[idx+1])
            print('weights matrix', idx+1, ': ', weights_matrix.shape)
            if weights_matrix.ndim == 3:
                weights_matrix = weights_matrix.reshape(weights_matrix.shape[1], weights_matrix.shape[2])
            weights_matrix = weights_matrix.T
            print('weights reshaped and transposed', weights_matrix.shape, '\n', weights_matrix)
            print('results\n', results[idx])
            print(results[idx].shape)
            error.append(np.matmul(weights_matrix, results[idx]))
            print('calculated error:\n', error[-1])
        print('error:', error)