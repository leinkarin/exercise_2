from helpers import *


def que_5_2_1():
    k_values = [1, 10, 100, 1000, 3000]
    l_values = ['l1', 'l2']

    accuracy_table = np.zeros((5, 2))

    for k in range(len(k_values)):
        for l in range(len(l_values)):
            accuracy_table[k][l] = knn_examples(X_train, Y_train, X_test, Y_test, k_values[k], l_values[l])

    create_table("Classification with k-Nearest Neighbours", "Distance Metric", "K", l_values, k_values, accuracy_table)


def que_5_2_2():
    k_max = 1
    k_min = 3000

    knn_que2(X_train, Y_train, X_test, Y_test, k_max, 'l2')
    knn_que2(X_train, Y_train, X_test, Y_test, k_min, 'l2')
    knn_que2(X_train, Y_train, X_test, Y_test, k_max, 'l1')


def que_5_3():
    AD_col_names, AD_test = read_data("AD_test.csv")
    knn_anomaly(AD_col_names, X_train, Y_train, AD_test)


def decision_tree_accuracies(tree_classifier):
    # Make predictions
    Y_pred_train = tree_classifier.predict(X_train)
    Y_pred_test = tree_classifier.predict(X_test)
    Y_pred_val = tree_classifier.predict(X_val)

    # Compute the accuracies of the predictions
    accuracy_train = np.mean(Y_pred_train == Y_train)
    accuracy_test = np.mean(Y_pred_test == Y_test)
    accuracy_val = np.mean(Y_pred_val == Y_val)

    accuracies = np.array([accuracy_train, accuracy_test, accuracy_val])
    return accuracies


def find_best_validation(tree_models_accuracies):
    best_validation_accuracy = np.max(tree_models_accuracies[:, :, 2])
    best_validation_index = np.argmax(tree_models_accuracies[:, :, 2])
    best_max_depth_index = best_validation_index // tree_models_accuracies.shape[1]
    best_max_leaf_nodes_index = best_validation_index % tree_models_accuracies.shape[1]
    train_accuracy = tree_models_accuracies[best_max_depth_index, best_max_leaf_nodes_index, 0]
    test_accuracy = tree_models_accuracies[best_max_depth_index, best_max_leaf_nodes_index, 1]
    print("The best validation accuracy = ", best_validation_accuracy,
          "the tree with the best validation accuracy has a training accuracy= ", train_accuracy, "and test accuracy= ",
          test_accuracy)

    return best_max_depth_index, best_max_leaf_nodes_index


def que_6():
    max_depth_values = [1, 2, 4, 6, 10, 20, 50, 100]
    max_leaf_nodes_values = [50, 100, 1000]

    # List to store the trained models and their information
    tree_models = np.empty((8, 3), dtype=object)
    tree_models_accuracies = np.zeros((8, 3, 3))

    # Loop through hyperparameter combinations
    for i in range(len(max_depth_values)):
        for j in range(len(max_leaf_nodes_values)):
            tree_models[i, j] = init_decision_tree(X_train, Y_train, max_depth_values[i], max_leaf_nodes_values[j])
            tree_models_accuracies[i, j] = decision_tree_accuracies(tree_models[i][j])

    create_tree_accuracies_table(max_leaf_nodes_values, max_depth_values, tree_models_accuracies)

    # find the index of the tree with the best validation accuracy
    best_max_depth_index, best_max_leaf_nodes_index = find_best_validation(tree_models_accuracies)
    # plot_decision_boundaries(tree_models[best_max_depth_index, best_max_leaf_nodes_index], )


if __name__ == '__main__':
    np.random.seed(0)
    col_names, X_train, Y_train = read_data()
    col_names, X_test, Y_test = read_data("test.csv")
    col_names, X_val, Y_val = read_data("validation.csv")

    # que_5_2_1()
    que_5_2_2()
    # que_5_3()
    # que_6()
