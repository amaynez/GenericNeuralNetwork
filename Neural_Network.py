import numpy as np

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

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

        # create a bias matrix
        self.bias = []
        for idx, hidden_col in enumerate(self.hidden):
            self.bias.append(np.random.uniform(-1, 1, size=(hidden_col, 1)))
        self.bias.append(np.random.uniform(-1, 1, size=(self.outputs, 1)))

    def forward_propagation(self, input_values):
        # create hidden results matrix based on the maximum number of cells in the hidden layers
        hidden_results = np.zeros((np.max(self.hidden), self.hidden.shape[0]))
        # prepare the input values
        input_values = np.array(input_values)[np.newaxis]
        input_values = input_values.T

        # calculate results for the first hidden layer (depends on the inputs)
        hidden_results[:self.hidden[0] + 1, :1] = \
            sigmoid(np.matmul(self.weights[0], input_values) + self.bias[0])

        # calculate results for subsequent hidden layers if any (depending on the previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                hidden_results[:self.hidden[idx + 1], idx + 1:idx + 2] = \
                    sigmoid(np.matmul(self.weights[idx + 1],
                                      hidden_results[:self.hidden[idx],
                                      idx:idx + 1]
                                      )
                            + self.bias[idx + 1])

        # calculate final result and return
        return sigmoid(np.matmul(self.weights[-1],
                                 hidden_results[:self.hidden[-1], -1:]
                                 )
                       + self.bias[-1])
