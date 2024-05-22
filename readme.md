# Decision Tree Implementation for Iris Dataset
This repository contains an implementation of a decision tree classifier using the ID3 algorithm. The classifier is specifically designed to work with the Iris dataset, a classic dataset in machine learning.

## Overview
The decision tree is built using the ID3 algorithm, which splits the dataset based on the attribute that provides the maximum information gain. The tree construction stops when a predefined maximum depth is reached or when the number of samples in a node falls below a minimum threshold. The resulting decision tree can then be used to classify new samples.

## Features
Builds a decision tree using the ID3 algorithm
Supports binary splits for continuous attributes
Includes a simple leaf merging mechanism to simplify the tree
Calculates the accuracy of the decision tree on a test set
Prints the structure of the decision tree
### Requirements
Python 3.x
Installation
No special installation is required. Simply clone this repository and ensure you have Python installed.

## Usage
### Dataset
Ensure you have the IRIS.csv file in the same directory as the script. The CSV file should have the following structure:


sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
...
### Running the Script
Clone the repository and navigate to the directory:


git clone <repository_url>
cd <repository_directory>
Place the IRIS.csv file in the same directory.

### Run the script:

python decision_tree.py
### Script Output
The script performs the following steps:

Loads the Iris dataset from the CSV file.
Splits the dataset into a training set (75%) and a test set (25%).
Builds the decision tree using the training set.
Merges leaves with identical predictions.
Evaluates the decision tree on the test set and prints the accuracy.
Prints the structure of the decision tree.
### Example output:

Total dataset size: 150
Training set size: 112
Test set size: 38
----------------
DECISION TREE
petal_length < 2.45?
[True] Iris-setosa
[False] petal_width < 1.75?
    [True] Iris-versicolor
    [False] Iris-virginica
----------------
Accuracy on test set: 97.37%
Code Explanation
Class TreeNode
Represents a node in the decision tree.
Contains methods to build the tree, predict the class for a sample, and merge identical leaf nodes.
Class ID3Tree
Manages the overall decision tree.
Contains methods to build the tree, merge identical leaves, predict the class for a sample, and print the tree structure.
Functions
calculate_entropy(data): Computes the entropy of the dataset.
calculate_info_gain(feature, split_value, data): Computes the information gain for a given feature and split value.
find_best_split(features, feature_values, data): Identifies the best feature and split value based on information gain.
load_iris_data(): Loads the Iris dataset from the CSV file.
Main Execution
Loads the dataset.
Splits the dataset into training and test sets.
Builds the decision tree.
Evaluates the decision tree on the test set.
Prints the accuracy and structure of the decision tree.
License
This project is licensed under the MIT License.

Contact
For any questions or issues, please open an issue on the repository or contact the author.
