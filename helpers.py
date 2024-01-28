import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from knn import KNNClassifier


def loading_random_forest():
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)


def loading_xgboost():
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)


def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def knn_examples(X_train, Y_train, X_test, Y_test, k=5, distance_metric='l2'):
    """
    Notice the similarity to the decision tree demo above.
    This is the sklearn standard format for models.
    """

    # Initialize the KNNClassifier with k=5 and L2 distance metric
    knn_classifier = KNNClassifier(k, distance_metric)

    # Train the classifier
    knn_classifier.fit(X_train, Y_train)

    # Predict the labels for the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)
    return accuracy


def knn_que2(X_train, Y_train, X_test, Y_test, k=5, distance_metric='l2'):
    """
    Notice the similarity to the decision tree demo above.
    This is the sklearn standard format for models.
    """

    # Initialize the KNNClassifier with k=5 and L2 distance metric
    knn_classifier = KNNClassifier(k, distance_metric)

    # Train the classifier
    knn_classifier.fit(X_train, Y_train)

    # Predict the labels for the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)

    plot_decision_boundaries(knn_classifier, X_test, Y_test, f"Distance metric = {distance_metric}, k= {k}")

    return accuracy


def knn_anomaly(AD_col_names, X_train, Y_train, AD_test, k=5, distance_metric='l2'):
    """
    Notice the similarity to the decision tree demo above.
    This is the sklearn standard format for models.
    """

    # Initialize the KNNClassifier with k=5 and L2 distance metric
    knn_classifier = KNNClassifier(k, distance_metric)

    # Train the classifier
    knn_classifier.fit(X_train, Y_train)

    # The distances to the knn
    distances, indices = knn_classifier.knn_distance(AD_test)

    distances_sums = np.sum(distances, axis=1, keepdims=True)

    # 50 test examples with the highest anomaly scores
    sorted_indices = np.argsort(distances_sums, axis=0)[::-1]
    anomalous_points = AD_test[sorted_indices[:50].flatten()]

    # normal points
    normal_points = AD_test[sorted_indices[50:].flatten()]

    # Plotting the points
    plt.scatter(anomalous_points[:, 0], anomalous_points[:, 1], c='red', label='Anomalous Points')
    plt.scatter(normal_points[:, 0], normal_points[:, 1], c='blue', label='Normal Points')
    plt.scatter(X_train[:, 0], X_train[:, 1], c='black', alpha=0.01, label='Training Points')

    plt.xlabel(AD_col_names[0])
    plt.ylabel(AD_col_names[1])
    plt.legend()
    plt.show()


def read_data_demo(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    data_numpy = df.values

    return data_numpy, col_names


def read_data(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = (list)(df.columns)

    # the data in numpy array format
    data_numpy = df.values

    # Extract the first two columns as features
    features = data_numpy[:, :2]

    if data_numpy.shape[1] > 2:
        # Extract the third column as labels
        labels = data_numpy[:, 2]
        return col_names, features, labels

    return col_names, features


def create_table(title, col_name, row_name, col_values, row_values, data):
    col_labels = [f"{col_name}= {c}" for c in col_values]
    row_labels = [f"{row_name}= {r}" for r in row_values]

    fig, ax = plt.subplots()
    ax.set_title(title)
    # Create the table
    table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=col_labels, rowLabels=row_labels)

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    ax.axis('off')
    plt.show()


def que_1():
    k_values = [1, 10, 100, 1000, 3000]
    l_values = ['l1', 'l2']

    accuracy_table = np.zeros((5, 2))

    for k in range(len(k_values)):
        for l in range(len(l_values)):
            accuracy_table[k][l] = knn_examples(X_train, Y_train, X_test, Y_test, k_values[k], l_values[l])

    create_table("Classification with k-Nearest Neighbours", "Distance Metric", "K", l_values, k_values, accuracy_table)


def que_2():
    k_max = 1
    k_min = 3000

    knn_que2(X_train, Y_train, X_test, Y_test, k_max, 'l2')
    knn_que2(X_train, Y_train, X_test, Y_test, k_min, 'l2')
    knn_que2(X_train, Y_train, X_test, Y_test, k_max, 'l1')


def que_3():
    AD_col_names, AD_test = read_data("AD_test.csv")
    knn_anomaly(AD_col_names, X_train, Y_train, AD_test)


def decision_tree_demo():
    # Create random data
    np.random.seed(42)
    X = np.random.rand(100, 2)  # Feature matrix with 100 samples and 2 features
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels based on a simple condition

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree classifier
    tree_classifier = DecisionTreeClassifier(random_state=42)

    # Train the Decision Tree on the training data
    tree_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = tree_classifier.predict(X_test)

    # Compute the accuracy of the predictions
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")


def decision_tree(max_depth, max_leaf_nodes):
    # Initialize Decision Tree classifier with current hyperparameters
    tree_classifier = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

    # Train the Decision Tree on the training data
    tree_classifier.fit(X_train, Y_train)

    # Make predictions
    Y_pred_train = tree_classifier.predict(X_train)
    Y_pred_test = tree_classifier.predict(X_test)
    Y_pred_val = tree_classifier.predict(X_val)

    # Compute the accuracies of the predictions
    accuracy_train = np.mean(Y_pred_train == Y_train)
    accuracy_test = np.mean(Y_pred_test == Y_test)
    accuracy_val = np.mean(Y_pred_val == Y_val)

    accuracy = np.array([accuracy_train, accuracy_test, accuracy_val])
    return accuracy


def que_4():
    max_depth_values = [1, 2, 4, 6, 10, 20, 50, 100]
    max_leaf_nodes_values = [50, 100, 1000]

    # List to store the trained models and their information
    tree_models = np.zeros((8, 3, 3))

    # Loop through hyperparameter combinations
    for i in range(len(max_depth_values)):
        for j in range(len(max_leaf_nodes_values)):
            tree_models[i, j] = decision_tree(max_depth_values[i], max_leaf_nodes_values[j])

    # create_table("Train accuracy", "Max leaf_nodes", "Max depth", max_leaf_nodes_values, max_depth_values,
    #              tree_models[:, :, 0])
    # create_table("Test accuracy", "Max leaf nodes", "Max depth", max_leaf_nodes_values, max_depth_values,
    #              tree_models[:, :, 1])
    # create_table("Validation accuracy", "Max leaf nodes", "Max depth", max_leaf_nodes_values, max_depth_values,
    #              tree_models[:, :, 2])

    best_validation_accuracy = np.max(tree_models[:, :, 2])
    best_validation_index = np.argmax(tree_models[:, :, 2])
    best_max_depth_index = best_validation_index // tree_models.shape[1]
    best_max_leaf_nodes_index = best_validation_index % tree_models.shape[1]
    train_accuracy = tree_models[best_max_depth_index, best_max_leaf_nodes_index, 0]
    test_accuracy = tree_models[best_max_depth_index, best_max_leaf_nodes_index, 1]
    print("The best validation accuracy = ", best_validation_accuracy,
          "the tree with the best validation accuracy has a training accuracy= ", train_accuracy, "and test accuracy= ",
          test_accuracy)


if __name__ == '__main__':
    np.random.seed(0)
    col_names, X_train, Y_train = read_data()
    col_names, X_test, Y_test = read_data("test.csv")
    col_names, X_val, Y_val = read_data("validation.csv")

    # que_1()
    # que_2()
    # que_3()
    que_4()
