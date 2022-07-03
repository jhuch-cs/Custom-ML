from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from math import exp as e_to_the
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error as mse

class MLP(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=True,hidden_layer_widths=None):
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr # Learning rate
        self.momentum = momentum
        self.shuffle = shuffle

    def fit(self, X, y, initial_weights=None, default_weight=None, numEpochs=None, num_outputs=None, validation_size=0.2):
        self.num_inputs = X.shape[1]
        self.num_outputs = y.shape[1] if num_outputs is None else num_outputs
        self.weights = self.initialize_weights(default_val=default_weight) if not initial_weights else initial_weights
        self.weights_change = self.initialize_weights(default_val=0.0)
        
        self.training_mse = []
        self.validation_mse = []
        self.validation_accuracy = []
        self.training_accuracy = []

        if self.shuffle:
            X, y = self._shuffle_data(X, y)

        if numEpochs is not None:
            for epoch in range(numEpochs):
                for input_data, target_data in zip(X, y):
                    final_output, output = self._forward(input_data)
                    self._backward(output, target_data) # Back-propagate error
        else:
            X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=validation_size)
            while (self._score_still_changing(X_validate, y_validate)):
                self._store_training_mse(X_train, y_train)
                self.validation_accuracy.append(self.score(X_validate, y_validate))
                self.training_accuracy.append(self.score(X_train, y_train))
                for input_data, target_data in zip(X_train, y_train):
                    final_output, output = self._forward(input_data)
                    self._backward(output, target_data)

        return self
    
    def _store_training_mse(self, X_train, y_train):
        train_mse = self._calc_mse(X_train, y_train)
        self.training_mse.append(train_mse)
    
    def _score_still_changing(self, X, y):
        new_mse = self._calc_mse(X, y)
        self.validation_mse.append(new_mse)
        if len(self.validation_mse) == 1:
            return True
        difference = abs(self.validation_mse[-2] - new_mse)
        # When the change is error is very low or when the error itself is very low
        return difference > 0.0001 or new_mse < 0.01 

    def _calc_mse(self, X, y):
        predictions = self.predict(X)
        return mse(y, predictions)

    def _forward(self, input_data):
        output = [input_data]
        for layer in self.weights:
            input_data = np.append(input_data, 1) # bias input
            new_input_data = np.zeros(layer.shape[1]) # nodes in this layer

            # calculate layer output as function of layer input
            for column in range(layer.shape[1]):
                net = np.dot(input_data, layer[:,column])
                thresholded = self._activation(net)
                new_input_data[column] = thresholded
            
            input_data = new_input_data
            output.append(input_data)
        return output[-1], output

    def _backward(self, all_output, targets):
        layers_error = self._calc_errors(all_output, targets)
        self._change_weights(layers_error, all_output)

    def predict(self, X):
        predictions = [self._forward(instance_data)[0] for instance_data in X]
        return predictions

    def initialize_weights(self, default_val=None):
        weights = []
        layer_widths = self.hidden_layer_widths.copy() if self.hidden_layer_widths is not None else [2 * self.num_inputs]
        layer_widths.insert(0, self.num_inputs)
        layer_widths.append(self.num_outputs)
        num_layers = len(layer_widths) - 1
        for layer in range(num_layers):
            num_nodes = layer_widths[layer] + 1
            num_weights = layer_widths[layer + 1]
            if default_val is not None:
                weights.append(np.full((num_nodes, num_weights), default_val))
            else:
                weights.append(np.random.uniform(-1, 1, (num_nodes, num_weights)))
        return weights # weights[layer][node][weights to next layer], with bias as last node in layer

    def score(self, X, y):
        predictions = self.predict(X)
        if self.num_outputs > 1:
            num_correct = len([1 for prediction, target in zip(predictions, y) if np.argmax(prediction) == np.argmax(target)])
        else:
            num_correct = len([1 for prediction, target in zip(predictions, y) if prediction == int(target)])
        return round(num_correct / len(y), 5)

    def _shuffle_data(self, X, y):
        return shuffle(X, y)

    def _activation(self, x):
        try:
            return 1.0 / (1.0 + e_to_the(-1.0 * x))
        except OverflowError:
            return 0.0


    def _activation_derivative(self, x):
        return x * (1.0 - x)

    def _vec_error(self, output, targets):
        nodes = []
        for i in range(len(output)):
            target = targets[i] if type(targets) in [list,tuple,np.ndarray] else float(targets)
            nodes.append((target - output[i]) * self._activation_derivative(output[i]))
        return nodes

    def _calc_errors(self, all_output, targets):
        output_layer_error = self._vec_error(all_output[-1], targets)
        layers_error = [output_layer_error]
        for layer in range(len(all_output) - 1, 1, -1):
            layers_error.insert(0, [])

            # Calculate the error at each node in the layer
            for node in range(len(all_output[layer - 1])):
                node_error = np.dot(self.weights[layer - 1][node], layers_error[1])
                node_error = node_error * self._activation_derivative(all_output[layer - 1][node])
                layers_error[0].append(node_error)
                
        return layers_error
    
    def _change_in_weight(self, error, output, previous_change):
        return self.lr * np.array(error) * output + self.momentum * previous_change
    
    def _change_weights(self, layers_error, layers_output):
        for layer_weights, i, layer_error, layer_output in zip(self.weights, range(len(self.weights_change)), layers_error, layers_output):
            layer_output = np.append(layer_output, 1) # re-introduce bias
            for node_weights, j, node_output in zip(layer_weights, range(len(self.weights_change[i])), layer_output):
                change = self._change_in_weight(layer_error, node_output, self.weights_change[i][j]) # weights_change might need to be a np.ndarray
                node_weights += change
                self.weights_change[i][j] = change
        
    def get_weights(self):
        return self.weights

    def get_validation_mse(self):
        return self.validation_mse

    def get_training_mse(self):
        return self.training_mse

    def get_validation_accuracy(self):
        return self.validation_accuracy

    def get_training_accuracy(self):
        return self.training_accuracy