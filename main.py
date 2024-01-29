from helpers import *


def que_5_2_1():
    """
    creates a table in size 5X2 of k values and a metric distance.
    """
    # define the values
    k_values = [1, 10, 100, 1000, 3000]
    l_values = ['l1', 'l2']

    accuracy_table = np.zeros((5, 2))

    # create the knn models
    for k in range(len(k_values)):
        for l in range(len(l_values)):
            accuracy_table[k][l] = knn_examples(X_train, Y_train, X_test, Y_test, k_values[k], l_values[l])

    create_table("Classification with k-Nearest Neighbours", "Distance Metric", "K", l_values, k_values, accuracy_table)


def que_5_2_2():
    """
    creates 3 knn models.
    """
    k_max = 1
    k_min = 3000

    knn_que2(X_train, Y_train, X_test, Y_test, k_max, 'l2')
    knn_que2(X_train, Y_train, X_test, Y_test, k_min, 'l2')
    knn_que2(X_train, Y_train, X_test, Y_test, k_max, 'l1')


def que_5_3():
    """
    creates an anomaly detection using knn.
    """
    AD_col_names, AD_test = read_data("AD_test.csv")
    knn_anomaly(AD_col_names, X_train, Y_train, AD_test)


def decision_tree_accuracies(tree_classifier):
    """
    computes the accuracies of the decision tree on the train, test and validation.

    - tree_classifier: decision tree
    """
    # make predictions
    Y_pred_train = tree_classifier.predict(X_train)
    Y_pred_test = tree_classifier.predict(X_test)
    Y_pred_val = tree_classifier.predict(X_val)

    # compute the accuracies of the predictions
    accuracy_train = np.mean(Y_pred_train == Y_train)
    accuracy_test = np.mean(Y_pred_test == Y_test)
    accuracy_val = np.mean(Y_pred_val == Y_val)

    accuracies = np.array([accuracy_train, accuracy_test, accuracy_val])
    return accuracies


def find_best_validation(row, col, tree_models_accuracies):
    """
    finds the index of the tree with the best validation accuracy.

    Parameters:
    - tree_models_accuracies: a numpy array in size 8X3 of vector in size 3 of accuracies (8X3X3)
    - row: the maximum row of the tree models accuracies we would like to search in
    - col: the maximum column of the tree models accuracies we would like to search in
    """

    best_validation_index = np.argmax(tree_models_accuracies[:row, :col, 2])
    best_depth_index = best_validation_index // tree_models_accuracies[:row, :col, 2].shape[1]
    best_leaf_nodes_index = best_validation_index % tree_models_accuracies[:row, :col, 2].shape[1]
    return best_depth_index, best_leaf_nodes_index


def print_accuracies(tree_models_accuracies, depth_index, leaf_nodes_index):
    """
    prints the accuracies of the decision tree in the coordinates (depth_index, lead_nodes_index) in the
    tree_models_accuracies array.

    -tree_models_accuracies: a numpy array in size 8X3 of vector in size 3 of accuracies (8X3X3)
    -depth_index: the row index in the array
    -leaf_nodes_index: the column index in the array

    """
    # compute all accuracies
    validation_accuracy = tree_models_accuracies[depth_index, leaf_nodes_index, 2]
    train_accuracy = tree_models_accuracies[depth_index, leaf_nodes_index, 0]
    test_accuracy = tree_models_accuracies[depth_index, leaf_nodes_index, 1]

    print("The best validation accuracy = ", validation_accuracy,
          "the tree with the best validation accuracy has a training accuracy= ", train_accuracy, "and test accuracy= ",
          test_accuracy)


def que_6():
    """
    creates two numpy array of size 8X3
    one hold a vector of size 3 of the accuracies of a tree and the other hold the tree models.

    """
    max_depth_values = [1, 2, 4, 6, 10, 20, 50, 100]
    max_leaf_nodes_values = [50, 100, 1000]

    tree_models = np.empty((8, 3), dtype=object)
    tree_models_accuracies = np.zeros((8, 3, 3))

    # loop through models combinations
    for i in range(len(max_depth_values)):
        for j in range(len(max_leaf_nodes_values)):
            tree_models[i, j] = init_decision_tree(X_train, Y_train, max_depth_values[i], max_leaf_nodes_values[j])
            tree_models_accuracies[i, j] = decision_tree_accuracies(tree_models[i][j])

    create_tree_accuracies_table(max_leaf_nodes_values, max_depth_values, tree_models_accuracies)

    # find the index of the tree with the best validation accuracy
    best_depth_index, best_leaf_nodes_index = find_best_validation(8, 3, tree_models_accuracies)
    plot_decision_boundaries(tree_models[best_depth_index, best_leaf_nodes_index], X_val, Y_val,
                             "Tree with the best validation accuracy")
    print_accuracies(tree_models_accuracies, best_depth_index, best_leaf_nodes_index)

    # find the index of the tree with the best validation accuracy that has 50 leaf nodes
    best_depth_index_50, best_leaf_nodes_index_50 = find_best_validation(8, 1, tree_models_accuracies)
    plot_decision_boundaries(tree_models[best_depth_index_50, 0], X_val, Y_val,
                             "Tree with the best validation accuracy that has only 50 leaf nodes")
    print_accuracies(tree_models_accuracies, best_depth_index_50, best_leaf_nodes_index_50)

    # find the index of the tree with the best validation accuracy that has max depth of 6
    best_depth_index_6, best_leaf_nodes_index_6 = find_best_validation(4, 3, tree_models_accuracies)
    plot_decision_boundaries(tree_models[best_depth_index_6, best_leaf_nodes_index_6], X_val, Y_val,
                             "Tree with the best validation accuracy that has maximum depth of 6")
    print_accuracies(tree_models_accuracies, best_depth_index_6, best_leaf_nodes_index_6)


def que_7():
    """
    creates a random forest model.

    """
    random_forest_model = loading_random_forest()
    random_forest_model.fit(X_train, Y_train)
    plot_decision_boundaries(random_forest_model, X_test, Y_test, "Random forest model")

    # predict the labels for the test set
    y_pred = random_forest_model.predict(X_test)

    # calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)
    print("The test accuracy of the random forest model= ", accuracy)


def que_8():
    """
    creates a XGBoost model.

    """
    xgboost_model = loading_xgboost()
    xgboost_model.fit(X_train, Y_train)
    plot_decision_boundaries(xgboost_model, X_test, Y_test, "XGBoost model")

    # predict the labels for the test set
    y_pred = xgboost_model.predict(X_test)

    # calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)
    print("The test accuracy of the XGBoost model= ", accuracy)


if __name__ == '__main__':
    np.random.seed(0)
    col_names, X_train, Y_train = read_data()
    col_names, X_test, Y_test = read_data("test.csv")
    col_names, X_val, Y_val = read_data("validation.csv")

    que_5_2_1()
    que_5_2_2()
    que_5_3()
    que_6()
    que_7()
    que_8()
