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
        # calculate final result and return, if explicit is set then return all the intermediate results as well
        output = []
        if 'explicit' in kwargs.keys():
            if kwargs.get('explicit') == 'yes':
                output = hidden_results
        output.append(sigmoid(np.matmul(self.weights[-1], hidden_results[-1]) + self.bias[-1]))
        return output

    def train(self, inputs, targets):
        # get the results including the hidden layers' (intermediate results)
        results = self.forward_propagation(inputs, explicit='yes')
        # prepare the targets and inputs for matrix operations
        targets = np.array(targets)[np.newaxis].T
        input_values = np.array(inputs)[np.newaxis].T
        # calculate the error (targets vs the outputs), index 0
        error = [results[-1] - targets]
        # calculate the error of the hidden layers from last to first
        for idx in range(len(results) - 2, -1, -1):
            error.insert(0, np.matmul(self.weights[idx + 1].T, error[0]))
        # modify weights
        self.weights[0] -= np.matmul((error[0] * d_sigmoid(results[0]) * self.learning_rate), input_values.T)
        self.bias[0] -= (error[0] * d_sigmoid(results[0])) * self.learning_rate
        for idx, weight_cols in enumerate(self.weights[1:]):
            weight_cols -= np.matmul(np.array((error[idx + 1] * d_sigmoid(results[idx + 1]) * self.learning_rate)),
                                     results[idx].T)
            self.bias[idx+1] -= (error[idx+1] * d_sigmoid(results[idx+1])) * self.learning_rate
