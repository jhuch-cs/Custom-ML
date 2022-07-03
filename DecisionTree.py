from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import math

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,counts,encoded_values=None):
        self.counts = counts
        self.encoded_values = encoded_values

    def fit(self, X, y):
        self.decision_tree = DecisionTreeNode(X, y, self.counts, encoded_values=self.encoded_values)
        self.decision_tree.split() # splits recursively
        return self

    def predict(self, X):
        predictions = [self.decision_tree.predict_instance(instance) for instance in X]
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        num_correct = len([1 for prediction, target in zip(predictions, y) if prediction == target])
        return round(num_correct / len(y), 4)
    
    def get_info_gains(self):
        return self.decision_tree.get_info_gains()

class DecisionTreeNode():
    def __init__(self, instance_data, target_data, counts, parent_node = None, 
                        attributes_already_split = None, encoded_values = None):
        if attributes_already_split is None:
            attributes_already_split = []

        self.attributes_already_split = attributes_already_split

        self.counts = counts

        self.instance_data = instance_data
        self.target_data = target_data
        self.encoded_values = encoded_values
        self.info_gains = []

        self.parent_node = parent_node
        self.children = []

        self.prediction = None

    # Print the tree recursively
    def __str__(self, level=0, child_num=None):
        tab = '\t'
        if self.prediction is None: # non-leaf node
            value_name = str(self.encoded_values[self.attributes_already_split[-2]][1][child_num], 'utf-8')
            build_str = f"{level * tab}{value_name}: " if child_num is not None else f"{level * tab}"
            build_str += f"Split on attr {self.encoded_values[self.attributes_already_split[-1]][0]}\n"

            for i, child in enumerate(self.children):
                build_str += child.__str__(level + 1, i)

            return build_str
        else: # leaf node
            value_name = str(self.encoded_values[self.attributes_already_split[-1]][1][child_num], 'utf-8')
            build_str = f"{level * tab}{value_name}: " if child_num is not None else f"{level * tab}"
            build_str += f"Predicted: {str(self.encoded_values[-1][1][self.prediction], 'utf-8')}\n"
            return build_str


    def predict_instance(self, instance):
        if self.prediction is not None: # leaf node
            return self.prediction
        
        attribute_to_split = self.attributes_already_split[-1]
        value_of_atttribute = instance[attribute_to_split]
        child_node = self.children[value_of_atttribute]

        return child_node.predict_instance(instance)

    
    def split(self):
        if self._is_leaf():
            self._save_prediction()
            return

        attribute_to_split, split_instance_data, split_target_data = self._create_split()
        self._create_children(attribute_to_split, split_instance_data, split_target_data)

        for child in self.children:
            child.split()


    def _create_children(self, attribute_to_split, split_instance_data, split_target_data): 
        self.attributes_already_split.append(attribute_to_split)

        for new_instance_data, new_target_data in zip(split_instance_data, split_target_data):
            self.children.append(DecisionTreeNode(new_instance_data, new_target_data, self.counts, self, self.attributes_already_split.copy(), encoded_values=self.encoded_values))


    def _create_split(self):
        unsplit_attributes = [attribute for attribute in range(self.instance_data.shape[1]) if attribute not in self.attributes_already_split]

        current_entropy = self._calc_entropy(self.target_data)

        potential_gains = [(self._calc_info_gain(attribute, current_entropy), attribute) for attribute in unsplit_attributes]
        best_gain = max(potential_gains)
        self.info_gains = [best_gain[0]]
        attribute_to_split = best_gain[1]

        split_instance_data, split_target_data = self._split_on_attribute(attribute_to_split)

        return attribute_to_split, split_instance_data, split_target_data

    def _split_on_attribute(self, attribute_to_split):
        concatenated_data = np.hstack((self.instance_data, self.target_data.reshape(self.instance_data.shape[0],1)))
        concatenated_data = concatenated_data[concatenated_data[:, attribute_to_split].argsort()]

        split_data = np.split(concatenated_data, np.unique(concatenated_data[:, attribute_to_split], return_index=True)[1][1:])

        split_target_data = [child_data[:,-1] for child_data in split_data]
        split_instance_data = [np.delete(child_data, -1, axis=1) for child_data in split_data]

        if len(split_instance_data) != self.counts[attribute_to_split]:
            which_missing = set(range(self.counts[attribute_to_split])).difference(np.unique(concatenated_data[:, attribute_to_split]))
            for i in which_missing:
                split_instance_data.insert(i, np.array([]))
                split_target_data.insert(i, np.array([]))

        return split_instance_data, split_target_data

    def _is_leaf(self):
        # if the target data is homogenous, or we're out of attributes
        return len(self.target_data) == 0 or np.all(self.target_data == self.target_data[0]) or len(self.attributes_already_split) == self.instance_data.shape[1]


    def _save_prediction(self):
        if len(self.target_data) == 0:
            if self.parent_node is not None:
                self.prediction = np.unique(self.parent_node.target_data)[0] # parent plurality class
            else:
                self.prediction = 0
        elif len(self.attributes_already_split) == self.instance_data.shape[1]:
            self.prediction = np.unique(self.target_data)[0] # if out of attributes, use plurality class
        else:
            self.prediction = self.target_data[0] # if homogenous target data, use that class

    def _calc_info_gain(self, attribute_index, current_entropy):
        split_instances, split_targets = self._split_on_attribute(attribute_index)
        unique_values_of_attribute = len(split_instances)

        s = self.instance_data.shape[0]

        split_entropy = sum([(len(split_instances[j]) / s) * self._calc_entropy(split_targets[j]) for j in range(unique_values_of_attribute)])

        return current_entropy - split_entropy

    def _calc_entropy(self, targets):
        proportions = np.unique(targets, return_counts=True)[1] / len(targets)
        return sum([-1 * p * math.log(p, 2) for p in proportions])

    def get_info_gains(self):
        children_gains = []
        for child in self.children:
            children_gains += child.get_info_gains()
        
        return self.info_gains + children_gains