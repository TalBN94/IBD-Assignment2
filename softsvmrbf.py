import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator
from softsvm import softsvm, get_error_percentage

gram_matrices_memo = {}


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmbf(l: float, sigma: float, trainX: np.array, trainy: np.array):
    """
    RBF Soft SVM algorithm (uses gaussian kernel for finding a separator)
    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    G = get_gram_matrix(sigma, trainX)
    G_mat = matrix(np.array(2*l*G))
    m = trainX.shape[0]
    zeros_mm = spmatrix([], [], [], (m, m))
    H = sparse([[G_mat, zeros_mm], [zeros_mm, zeros_mm]], tc='d')
    H = H + sparse(matrix(np.eye(2*m) * 1e-6))

    v1 = matrix(np.array(np.ones((m, 1))))
    v2 = matrix(np.array(np.zeros((m, 1))))
    v = matrix([v1, v2], tc='d')

    u1 = matrix(np.array(np.zeros((m, 1))))
    u2 = matrix(np.array((1 / m) * np.ones((m, 1))))
    u = matrix([u1, u2], tc='d')

    y_diag = np.diag(trainy.reshape(-1))
    A1 = y_diag @ G
    A1 = matrix(np.array(A1))
    Im = np.eye(m)
    Im = matrix(np.array(Im))
    A = sparse([[A1, zeros_mm], [Im, Im]], tc='d')

    sol = solvers.qp(H, u, -A, -v)
    z = sol["x"]
    return np.array(z)[:m, :]


def get_gram_matrix(sigma, X):
    """
    Calculates the Gram matrix for a given train data X and the gaussian kernel function
    :param sigma: sigma parameter value for the gaussian kernel function
    :param X: train data
    :return: Gram matrix (gaussian kernel) for the train data
    """
    m = X.shape[0]
    if sigma in gram_matrices_memo:
        return gram_matrices_memo.get(sigma)
    G = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            K_ij = gaussian_kernal(sigma, X[i], X[j])
            G[i][j] = K_ij
            G[j][i] = K_ij
    gram_matrices_memo[sigma] = G
    return G


def gaussian_kernal(sigma, x_i, x_j):
    """
    Calculates the gaussian kernel function for 2 data points
    :param sigma: sigma parameter value
    :param x_i: first data point
    :param x_j: second data point
    :return: K(x_i, x_j) (gaussian kernel)
    """
    diff = np.linalg.norm(x_i - x_j) ** 2
    return np.exp(-diff / (2 * sigma))


def plot_data_by_label(X, y):
    """
    Plots the given data points colores by label (for section a)
    :param X: the data
    :param y: the labels
    """
    plt.title('Data points colored by label')
    for label in np.unique(y):
        i = np.where(y == label)
        plt.scatter(X[i, 0], X[i, 1], label=label)
    plt.legend()
    plt.show()


def get_folds(X, y, k):
    """
    Divides the data and labels to k folds
    :param X: the data
    :param y: the labels
    :param k: amount of folds
    :return: batches and labels, where batches[i] is the i'th data fold and labels[i] is the i'th label fold which
    matches batches[i]
    """
    m = X.shape[0]
    batch_size = int(m / 5)
    shuffler = np.random.permutation(m)
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]
    batches = []
    labels = []
    for i in range(k):
        batches.append(X_shuffled[i * batch_size:(i + 1) * batch_size, :])
        labels.append(y_shuffled[i * batch_size:(i + 1) * batch_size, :])
    return batches, labels


def cross_validation_rbf(X, y, k, ls, sigmas):
    """
    Performs cross validation for the RBF Soft SVM algorithm
    :param X: the data
    :param y: the labels
    :param k: amount of folds
    :param ls: list of values to check for lambda parameter
    :param sigmas: list of values to check for sigma parameter
    :return: the best combination of lambda and sigma which minimizes the average error on the data folds
    """
    batches, labels = get_folds(X, y, k)
    min_error = float('inf')
    l_min = 0
    sigma_min = 0
    for l in ls:
        for sigma in sigmas:
            print(f'Checking combination: l={l}, sigma={sigma}...\n')
            combination_error = 0
            gram_matrices_memo.clear()
            for i in range(k):
                print(f'k={i}\n')
                validation_set_X = batches[i]
                validation_set_y = labels[i]
                batches_copy = batches.copy()
                batches_copy.pop(i)
                labels_copy = labels.copy()
                labels_copy.pop(i)
                train_set_X = np.concatenate(batches_copy, axis=0)
                train_set_y = np.concatenate(labels_copy, axis=0)
                alpha_i = softsvmbf(l, sigma, train_set_X, train_set_y)
                combination_error += calc_validation_error(train_set_X, validation_set_X, validation_set_y, sigma,
                                                           alpha_i)
            avg_error = combination_error / k
            print(f'average err: {avg_error}\n\n')
            if avg_error < min_error:
                min_error = avg_error
                l_min = l
                sigma_min = sigma
    return l_min, sigma_min


def cross_validation_linear(X, y, k, ls):
    """
    Performs cross validation for the Soft SVM algorithm
    :param X: the data
    :param y: the labels
    :param k: amount of folds
    :param ls: list of values to check for lambda parameter
    :return: the best lambda which minimizes the average error on the data folds
    """
    batches, labels = get_folds(X, y, k)
    min_error = float('inf')
    l_min = 0
    for l in ls:
        print(f'Checking for lamda: {l}...\n')
        combination_error = 0
        for i in range(k):
            validation_set_X = batches[i]
            validation_set_y = labels[i]
            batches_copy = batches.copy()
            batches_copy.pop(i)
            labels_copy = labels.copy()
            labels_copy.pop(i)
            train_set_X = np.concatenate(batches_copy, axis=0)
            train_set_y = np.concatenate(labels_copy, axis=0)
            w_i = softsvm(l, train_set_X, train_set_y)
            combination_error += get_error_percentage(np.sign(validation_set_X @ w_i), validation_set_y)
        avg_error = combination_error / k
        print(f'Average error: {avg_error}\n\n')
        if avg_error < min_error:
            min_error = avg_error
            l_min = l
    return l_min


def get_predictions(validation_X, train_X, sigma, alpha):
    """
    Calculates the predictions for a given validation set
    :param validation_X: the validation set data
    :param train_X: the training data
    :param sigma: sigma parameter value (for gaussian kernel function)
    :param alpha: the seperator alpha received from RBF Soft SVM algorithm
    :return: predictions vector (label for each validation data point)
    """
    G_val = get_validation_kernels_matrix(validation_X, train_X, sigma)
    preds = np.sign(G_val @ alpha)
    return preds


def calc_validation_error(train_X, validation_X, validation_y, sigma, alpha):
    """
    Calculates the validation error rate, which is the ratio between misclassified validation data points to the
    entire validation data
    :param train_X: train data
    :param validation_X: validation set data
    :param validation_y: validation set labels
    :param sigma: sigma parameter value (for gaussian kernel function)
    :param alpha: the seperator alpha received from RBF Soft SVM algorithm
    :return: validation error rate
    """
    validation_size = validation_X.shape[0]
    preds = get_predictions(validation_X, train_X, sigma, alpha)
    error_count = 0
    for i in range(validation_size):
        if preds[i] != validation_y[i][0]:
            error_count += 1
    return error_count / validation_size


def get_validation_kernels_matrix(validation_X, train_X, sigma):
    """
    Calculates the kernels for the entire validation data and returns it as a matrix, where the i'th row of the returned
    matrix holds the kernel calculations for the i'th validation data point
    :param validation_X: validation set data
    :param train_X: train data
    :param sigma: sigma parameter value (for gaussian kernel function)
    :return: Kernels matrix for the validation data
    """
    m = train_X.shape[0]
    n = validation_X.shape[0]
    G_val = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K_ij = gaussian_kernal(sigma, train_X[j], validation_X[i])
            G_val[i][j] = K_ij
    return G_val


def run_grid_experiment(trainX, trainy):
    """
    Performs classification of a grid consists of 10,000 point in the range of the train data points, and plots the grid
    colored by label (for section d)
    :param trainX: train data
    :param trainy: train labels
    """
    sigmas = [0.01, 0.5, 1]
    l = 100
    grid_size = 100
    grid_min = np.min(trainX)
    grid_max = np.max(trainX)
    grid_points = np.linspace(grid_min, grid_max, num=grid_size)
    xs, ys = np.meshgrid(grid_points, grid_points)
    grid = np.array([xs.flatten(), ys.flatten()]).T
    for sigma in sigmas:
        print(f'Running experiment with sigma={sigma}...\n')
        alpha_i = softsvmbf(l, sigma, trainX, trainy)
        preds_i = get_predictions(grid, trainX, sigma, alpha_i)
        grid_labeled = preds_i.reshape(grid_size, grid_size)
        grid_colors = ListedColormap(['blue', 'red'])
        patches = [Patch(color='blue', label='-1'), Patch(color='red', label='1')]
        fig, ax = plt.subplots()
        loc_x = AutoMinorLocator(n=10)
        loc_y = AutoMinorLocator(n=10)
        ax.xaxis.set_minor_locator(loc_x)
        ax.yaxis.set_minor_locator(loc_y)
        ax.grid(which='both', axis='both')
        ax.imshow(grid_labeled, cmap=grid_colors, extent=[grid_min, grid_max, grid_min, grid_max])
        plt.legend(handles=patches)
        plt.title(f'Grid colored by label (sigma={sigma})')
        plt.show()


def simple_test():
    # load question 4 data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvmbf(10, 0.1, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
    # load data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    # section a - plot data points colored by label
    plot_data_by_label(trainX, trainy)

    # section b - running RBF soft SVM experiment
    print('==================\nSection B\n==================\n')
    print('\n==================\nRBF Soft SVM\n==================\n')
    ls = [1, 10, 100]
    sigmas = [0.01, 0.5, 1]
    best_l, best_sigma = cross_validation_rbf(trainX, trainy, 5, ls, sigmas)
    print(f'Best sigma: {best_sigma}\nBest l: {best_l}')

    alpha = softsvmbf(best_l, best_sigma, trainX, trainy)
    test_error = calc_validation_error(trainX, testX, testy, best_sigma, alpha)
    print(f"The test error classifying with optimal alpha: {test_error}\n")

    print('\n==================\nLinear Soft SVM\n==================\n')
    best_l_linear = cross_validation_linear(trainX, trainy, 5, ls)
    print(f'Best l: {best_l_linear}')

    w = softsvm(best_l_linear, trainX, trainy)
    linear_test_error = get_error_percentage(np.sign(testX @ w), testy)
    print(f"The test error classifying with optimal w: {linear_test_error}\n")

    print('==================\nSection D\n==================\n')
    run_grid_experiment(trainX, trainy)
