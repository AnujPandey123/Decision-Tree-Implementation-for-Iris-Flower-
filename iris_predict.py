import csv
import random
import math

# Conditions to stop splitting the tree further
MIN_SAMPLE_SIZE = 4
MAX_DEPTH = 3

class TreeNode:
    def __init__(self, data, features, feature_values, depth):
        self.is_leaf = False
        self.data = data
        self.split_feature = None
        self.split_value = None
        self.features = features
        self.feature_values = feature_values
        self.left = None
        self.right = None
        self.prediction = None
        self.depth = depth

    def build_tree(self):
        data = self.data

        # Continue tree building unless a stopping criterion is met
        if self.depth < MAX_DEPTH and len(data) >= MIN_SAMPLE_SIZE and len(set([item["species"] for item in data])) > 1:
            # Find the best attribute and split point based on information gain
            best_gain, best_feature, best_split = find_best_split(self.features, self.feature_values, data)

            # Proceed if information gain is positive
            if best_gain > 0:
                self.split_value = best_split
                self.split_feature = best_feature

                # Split the data based on the best split
                left_data = [item for item in data if item[best_feature] < best_split]
                right_data = [item for item in data if item[best_feature] >= best_split]
                self.left = TreeNode(left_data, self.features, self.feature_values, self.depth + 1)
                self.right = TreeNode(right_data, self.features, self.feature_values, self.depth + 1)
                self.left.build_tree()
                self.right.build_tree()
            else:
                self.is_leaf = True
        else:
            self.is_leaf = True

        if self.is_leaf:
            # Leaf node prediction is the most frequent class in the subset
            species_counts = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}
            for item in data:
                species_counts[item["species"]] += 1
            self.prediction = max(species_counts, key=species_counts.get)

    def predict(self, sample):
        if self.is_leaf:
            return self.prediction
        else:
            if sample[self.split_feature] < self.split_value:
                return self.left.predict(sample)
            else:
                return self.right.predict(sample)

    def merge_identical_leaves(self):
        if not self.is_leaf:
            self.left.merge_identical_leaves()
            self.right.merge_identical_leaves()
            if self.left.is_leaf and self.right.is_leaf and self.left.prediction == self.right.prediction:
                self.is_leaf = True
                self.prediction = self.left.prediction

    def display_tree(self, prefix=""):
        if self.is_leaf:
            print("\t" * self.depth + prefix + self.prediction)
        else:
            print("\t" * self.depth + prefix + f"{self.split_feature} < {self.split_value}?")
            self.left.display_tree("[True] ")
            self.right.display_tree("[False] ")

class ID3Tree:
    def __init__(self):
        self.root = None

    def build(self, data, features, feature_values):
        self.root = TreeNode(data, features, feature_values, 0)
        self.root.build_tree()

    def merge_identical_leaves(self):
        self.root.merge_identical_leaves()

    def predict(self, sample):
        return self.root.predict(sample)

    def display_tree(self):
        print("----------------")
        print("DECISION TREE")
        self.root.display_tree()
        print("----------------")

def calculate_entropy(data):
    if len(data) == 0:
        return 0

    target = "species"
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    entropy = 0
    for sp in species:
        p = len([item for item in data if item[target] == sp]) / len(data)
        if p > 0:
            entropy -= p * math.log(p, 2)

    return entropy

def calculate_info_gain(feature, split_value, data):
    left_split = [item for item in data if item[feature] < split_value]
    right_split = [item for item in data if item[feature] >= split_value]

    p_left = len(left_split) / len(data)
    p_right = len(right_split) / len(data)

    info_gain = calculate_entropy(data)
    info_gain -= p_left * calculate_entropy(left_split)
    info_gain -= p_right * calculate_entropy(right_split)

    return info_gain

def find_best_split(features, feature_values, data):
    best_gain = 0
    for feature in features:
        for split_value in feature_values[feature]:
            gain = calculate_info_gain(feature, split_value, data)
            if gain >= best_gain:
                best_gain = gain
                best_feature = feature
                best_split = split_value
    return best_gain, best_feature, best_split

def load_iris_data():
    data = []
    with open('IRIS.csv', newline='') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # Skip header row
        for row in reader:
            instance = {
                "sepal_length": float(row[0]),
                "sepal_width": float(row[1]),
                "petal_length": float(row[2]),
                "petal_width": float(row[3]),
                "species": row[4]
            }
            data.append(instance)
    return data

if __name__ == '__main__':
    data = load_iris_data()

    if not data:
        print('Dataset is empty!')
        exit(1)

    test_data = random.sample(data, int(0.25 * len(data)))
    training_data = [item for item in data if item not in test_data]

    print('Total dataset size:', len(data))
    print('Training set size:', len(training_data))
    print('Test set size:', len(test_data))

    feature_list = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    feature_domains = {}
    for feature in feature_list:
        feature_domains[feature] = list(set(item[feature] for item in data))

    decision_tree = ID3Tree()
    decision_tree.build(training_data, feature_list, feature_domains)
    decision_tree.merge_identical_leaves()

    accuracy = sum(1 for sample in test_data if sample["species"] == decision_tree.predict(sample)) / len(test_data)

    decision_tree.display_tree()

    print("Accuracy on test set: {:.2f}%".format(accuracy * 100))
