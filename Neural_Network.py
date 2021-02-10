import numpy as np

class NeuralNetwork:

    def __init__(self, inputs, hidden, outputs):
        print(inputs, hidden, outputs)
        self.inputs = inputs
        self.hidden = np.array([hidden])
        if self.hidden.ndim == 0:
            # create weights matrix based on an integer (single hidden layer)
            self.weights = np.random.randint(1, 10, size=(self.hidden[0], self.inputs))
        else:
            # create weights matrix based on a list input (multiple hidden layer)
            self.weights = \
                np.random.randint(1,
                                  10,
                                  size=(self.hidden.shape[1] + 1,
                                        np.max(self.hidden) if np.max(self.hidden) > self.inputs else self.inputs)
                                  )
        self.outputs = outputs
        self.bias = np.random.randint(1, 10, size=(self.hidden.shape[1]+1))
        print('bias', self.bias)

    def forward_propagation (self, input_values):
        # create results matrix
        hidden_results = np.zeros((np.max(self.hidden), self.hidden.shape[1]))
        print('hidden results matrix\n', hidden_results)
        print('hidden results matrix split\n', hidden_results[:, :1])
        # calculate results for the first hidden layer because it depends on the inputs
        print('weights \n', self.weights)
        input_values = np.array(input_values)[np.newaxis]
        input_values = input_values.T
        print('inputs: \n', input_values)

        print('weights index split: \n', self.weights[:(self.hidden[0][0]+1), :input_values.shape[0]])

        hidden_results[:, :1] = np.matmul(self.weights[:(self.hidden[0][0]+1), :input_values.shape[0]], input_values) + self.bias[0]
        print('hidden results after first layer\n', hidden_results[:, :1])
        # calculate results for subsequent hidden layers if any
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                print('weights index split for ' + str(idx), self.weights[idx][:self.hidden[idx-1]])
                hidden_results[idx] = np.matmul(self.weights[idx][:self.hidden[idx-1], hidden_results[idx-1]]) + self.bias[idx]
                print('hidden results after ' + str(idx+1) + ' layers', hidden_results)
        # calculate results for output
        print('weights index split for output', self.weights[:, :self.hidden[-1][-1]])
        print('hidden results for output', hidden_results)
        print('hidden results for output sliced', hidden_results)
        output_results = np.matmul(self.weights[-1][:self.hidden[-1]], hidden_results) + self.bias[-1]
        print('output results: ', output_results)
        return output_results


