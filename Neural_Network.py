import numpy as np

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

class NeuralNetwork:

    def __init__(self, inputs, hidden, outputs):
        # print(inputs, hidden, outputs)
        self.inputs = inputs
        if isinstance(hidden, int):
            hidden = [hidden]
        self.hidden = np.array(hidden)
        self.outputs = outputs
        self.weights = [np.random.uniform(-1, 1, size=(self.hidden[0], self.inputs))]
        if self.hidden.ndim > 0:
            # create weights matrix based on a list (multiple hidden layer)
            for idx, hidden_cols in enumerate(self.hidden[1:]):
                self.weights.append(np.random.uniform(-1, 1, size=(hidden_cols, self.hidden[idx])))
        self.output_weights = [np.random.uniform(-1, 1, size=(self.outputs, self.hidden[-1],))]
        self.bias = np.random.uniform(-1, 1, size=(self.hidden.shape[0] + 1))
        # print('bias', self.bias)

    def forward_propagation (self, input_values):
        # create results matrix
        hidden_results = np.zeros((np.max(self.hidden), self.hidden.shape[0]))
        # print('hidden results matrix\n', hidden_results)
        # print('hidden results matrix split\n', hidden_results[:self.hidden[0], :1])
        # calculate results for the first hidden layer because it depends on the inputs
        # print('weights \n', self.weights)
        input_values = np.array(input_values)[np.newaxis]
        input_values = input_values.T
        # print('inputs: \n', input_values)

        # print('weights index split: \n', self.weights[0])

        hidden_results[:self.hidden[0], :1] = sigmoid(np.matmul(self.weights[0], input_values) + self.bias[0])
        # print('hidden results after first layer\n', hidden_results[:, :1])
        # calculate results for subsequent hidden layers if any
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                # print('weights index split for ' + str(idx+1), '\n', self.weights[idx+1])
                # print('results as input: \n', hidden_results[:self.hidden[idx], idx:idx+1])
                hidden_results[:self.hidden[idx+1], idx+1:idx+2] = sigmoid(np.matmul(self.weights[idx+1], hidden_results[:self.hidden[idx], idx:idx+1]) + self.bias[idx])
                # print('results after ' + str(idx+2) + ' hidden layers\n', hidden_results)
        # calculate results for output
        # print('weights index split for output\n', self.output_weights)
        # print('hidden results for output sliced\n', hidden_results[:self.hidden[-1], -1:])
        output_results = sigmoid(np.matmul(self.output_weights, hidden_results[:self.hidden[-1], -1:]) + self.bias[-1])
        # print('output results: \n', output_results)
        return output_results


