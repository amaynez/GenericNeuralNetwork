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
        # create weights and bias matrices
        self.weights = [np.random.uniform(-1, 1, size=(self.hidden[0], self.inputs))]
        if self.hidden.ndim > 0:
            for idx, hidden_cols in enumerate(self.hidden[1:]):
                self.weights.append(np.random.uniform(-1, 1, size=(hidden_cols, self.hidden[idx])))
        self.weights.append([np.random.uniform(-1, 1, size=(self.outputs, self.hidden[-1],))])
        self.bias = np.random.uniform(-1, 1, size=(self.hidden.shape[0] + 1))

    def forward_propagation (self, input_values):
        # create results matrix
        hidden_results = np.zeros((np.max(self.hidden), self.hidden.shape[0]))

        # calculate results for the first hidden layer because it depends on the inputs
        input_values = np.array(input_values)[np.newaxis]
        input_values = input_values.T
        hidden_results[:self.hidden[0], :1] =\
            sigmoid(np.matmul(self.weights[0], input_values) +
                    self.bias[0])

        # calculate results for subsequent hidden layers if any
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                hidden_results[:self.hidden[idx+1], idx+1:idx+2] =\
                    sigmoid(np.matmul(self.weights[idx+1], hidden_results[:self.hidden[idx], idx:idx+1]) +
                            self.bias[idx+1])
        # calculate results for output
        output_results =\
            sigmoid(np.matmul(self.weights[-1], hidden_results[:self.hidden[-1], -1:]) +
                    self.bias[-1])

        return output_results


