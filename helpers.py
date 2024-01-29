import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from knn import KNNClassifier


def loading_random_forest():
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)
    return model


def loading_xgboost():
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)
    return model


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

    # plotting
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

    # extract the first two columns as features
    features = data_numpy[:, :2]

    if data_numpy.shape[1] > 2:
        # extract the third column as labels
        labels = data_numpy[:, 2]
        return col_names, features, labels

    return col_names, features


def create_table(title, col_name, row_name, col_values, row_values, data):
    """
    creates and shows a table.
    """
    col_labels = [f"{col_name}= {c}" for c in col_values]
    row_labels = [f"{row_name}= {r}" for r in row_values]

    fig, ax = plt.subplots()
    ax.set_title(title)

    # create the table
    table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=col_labels, rowLabels=row_labels)

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    ax.axis('off')
    plt.show()


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


def init_decision_tree(X_train, Y_train, max_depth, max_leaf_nodes):
    """
    creates decision tree

    """
    # Initialize Decision Tree classifier with current hyperparameters
    tree_classifier = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)

    # Train the Decision Tree on the training data
    tree_classifier.fit(X_train, Y_train)

    return tree_classifier


def create_tree_accuracies_table(max_leaf_nodes_values, max_depth_values, tree_models_accuracies):
    """
    creates 3 tables in size 8X3, each containing accuracies on a different data set.

    """
    create_table("Train accuracy", "Max leaf_nodes", "Max depth", max_leaf_nodes_values, max_depth_values,
                 tree_models_accuracies[:, :, 0])
    create_table("Test accuracy", "Max leaf nodes", "Max depth", max_leaf_nodes_values, max_depth_values,
                 tree_models_accuracies[:, :, 1])
    create_table("Validation accuracy", "Max leaf nodes", "Max depth", max_leaf_nodes_values, max_depth_values,
                 tree_models_accuracies[:, :, 2])
