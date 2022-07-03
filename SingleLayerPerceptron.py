from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron
import numpy as np
import collections

class PerceptronClassifier(BaseEstimator,ClassifierMixin):
    SIGMA = 0.1 # Significance theshold
    ROUNDS_TO_REMEMBER = 4

    def __init__(self, lr=.1, shuffle=True):
        self.lr = lr
        self.shuffle = shuffle
        self.recent_changes = collections.deque(maxlen=self.ROUNDS_TO_REMEMBER)
        self.num_epochs = 0
        self.misclass = []

    def fit(self, X, y, initial_weights=None, numRounds=0):
        self.num_epochs = 0
        self.misclass = []
        self.num_weights = X.shape[1]
        self.weights = self.initialize_weights() if not initial_weights else initial_weights
        self.recent_changes.append(np.array(self.weights))

        if (numRounds == 0):
            while(True):
                self.num_epochs += 1
                weights_before_epoch = self.weights
                misclassification_this_epoch = 1 - self.score(X, y)
                self.misclass.append(misclassification_this_epoch)

                weights_after_epoch = self._run_epoch(X, y)
                weights_change = weights_after_epoch - weights_before_epoch
                self.recent_changes.append(weights_change)

                if (np.linalg.norm(weights_change) < self.SIGMA / 10 or \
                    self._magnitude_of_recent_changes() < self.SIGMA):
                    break
        else:
            self.num_epochs = numRounds
            for _ in range(numRounds):
                misclassification_this_epoch = 1 - self.score(X, y)
                self.misclass.append(misclassification_this_epoch)
                
                weights_after_epoch = self._run_epoch(X, y)

        return self
    
    def _run_epoch(self, X, y):
        shuffled_X = X
        shuffled_y = y
        if self.shuffle:
            shuffled_tuple = self._shuffle_data(X, y)
            shuffled_X = shuffled_tuple[0]
            shuffled_y = shuffled_tuple[1]

        # Run a single epoch, mutating the weights
        for input, target in zip(shuffled_X, shuffled_y):
            output = self._get_output(input)
            change_in_weights = self.lr * (int(target) - output) * input
            self.weights = np.add(self.weights, change_in_weights)        
        return self.weights
    
    def _get_output(self, instance_data):
        net = np.dot(instance_data, self.weights)
        output = int(net > 0)
        return output
    
    def _magnitude_of_recent_changes(self):
        return np.linalg.norm(sum(self.recent_changes)) / (self.num_weights / 4)

    def predict(self, X):
        return [self._get_output(instance_data) for instance_data in X]

    def initialize_weights(self):
        return [0] * self.num_weights

    def score(self, X, y):
        predictions = self.predict(X)
        num_correct = len([1 for prediction, target in zip(predictions, y) if prediction == int(target)])
        return round(num_correct / len(y), 2)

    def _shuffle_data(self, X, y):
        concatenated_data = np.hstack((X, y.reshape(X.shape[0],1)))
        np.random.shuffle(concatenated_data)
        shuffled_y = concatenated_data[:,-1] # get last column
        shuffled_X = np.delete(concatenated_data, -1, axis=1)
        return (shuffled_X, shuffled_y)

    def get_weights(self):
        return self.weights

    def get_epochs(self):
        return self.num_epochs
    
    def get_misclassification(self):
        return self.misclass