import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def softsvm(l, trainX: np.array, trainy: np.array):
    """
    Soft SVM algorithm
    :param l: the parameter lambda of the soft SVM algorithm
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: linear predictor w, a numpy array of size (d, 1)
    """
    d = trainX.shape[1]
    m = trainX.shape[0]

    # build H
    Id = spmatrix(2.0*l, range(d), range(d))
    zeros_dm = spmatrix([], [], [], (d, m))
    zeros_md = spmatrix([], [], [], (m, d))
    zeros_mm = spmatrix([], [], [], (m, m))
    H = sparse([[Id, zeros_md], [zeros_dm, zeros_mm]], tc='d')

    # build v
    v1 = matrix(np.array(np.ones((m, 1))))
    v2 = matrix(np.array(np.zeros((m, 1))))
    v = matrix([v1, v2], tc='d')

    # build u
    u1 = matrix(np.array(np.zeros((d, 1))))
    u2 = matrix(np.array((1 / m) * np.ones((m, 1))))
    u = matrix([u1, u2], tc='d')

    # build A
    y_diag = np.diag(trainy.reshape(-1))
    A1 = y_diag @ trainX
    A1 = matrix(np.array(A1))
    Im = np.eye(m)
    Im = matrix(np.array(Im))
    A = sparse([[A1, zeros_md], [Im, Im]], tc='d')

    # Run solver and extract w from the solution z
    sol = solvers.qp(H, u, -A, -v)
    z = sol["x"]
    return np.array(z)[:d, :]


def load_n_data(n):
    """
    Load n data points (randomly) for the train data and the entire test data for Q2
    :param n: number of train data points to load randomly
    :return: train data (n data points), test data for Q2
    """
    # load question 2 data
    data = np.load('ex2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    # Get a random n training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:n]]
    _trainy = trainy[indices[:n]]

    return _trainX, _trainy, testX, testy


def get_error_percentage(pred_labels, real_labels):
    """
    Calculates the error rate of the prediction
    :param pred_labels: predicted labels
    :param real_labels: actual labels
    :return: ratio between misclassified data and the entire data
    """
    pred_list = pred_labels.flatten()
    real_list = real_labels.flatten()
    error_count = 0
    for i in range(len(real_list)):
        if pred_list[i] != real_list[i]:
            error_count += 1
    return error_count / len(real_list)


def plot_experiment_result(results_train, results_test, lamda_exps, results_train_large=None, results_test_large=None,
                           lambda_exps_large=None, is_small=True):
    """
    Plots the experiment results according to the requirements
    :param results_train: train error results
    :param results_test: test error results
    :param lamda_exps: the lambda powers used in the experiment
    :param results_train_large: train error result (if it's the large train size experiment)
    :param results_test_large: test error result (if it's the large train size experiment)
    :param lambda_exps_large: the lambda powers used in the experiment (if it's the large train size experiment)
    :param is_small: indicates if it's the small train sample experiment of the large train sample experiment
    """
    error_mins_train = np.asarray([result[2] for result in results_train])
    error_maxes_train = np.asarray([result[1] for result in results_train])
    error_means_train = np.asarray([result[0] for result in results_train])
    plt.errorbar(lamda_exps, error_means_train, [error_means_train - error_mins_train, error_maxes_train - error_means_train],
                 fmt='o', lw=1, ecolor='blue', zorder=2)

    error_mins_test = np.asarray([result[2] for result in results_test])
    error_maxes_test = np.asarray([result[1] for result in results_test])
    error_means_test = np.asarray([result[0] for result in results_test])
    plt.errorbar(lamda_exps, error_means_test, [error_means_test - error_mins_test, error_maxes_test - error_means_test],
                 fmt='o', lw=1, ecolor='orange', zorder=2)

    if is_small:
        plt.legend(['Train result', 'Test results'], loc='upper left')
    else:
        error_means_train_large = np.asarray([result[0] for result in results_train_large])
        error_means_test_large = np.asarray([result[0] for result in results_test_large])
        plt.scatter(lambda_exps_large, error_means_train_large, color='green', zorder=3)
        plt.scatter(lambda_exps_large, error_means_test_large, color='purple', zorder=3)
        plt.legend(['Train result (1000)', 'Test results (1000)', 'Train result (100)', 'Test results (100)'],
                   loc='upper left')

    plt.plot(lamda_exps, error_means_train, 'blue', zorder=0)
    plt.plot(lamda_exps, error_means_test, 'orange', zorder=0)
    plt.xticks(lamda_exps, lamda_exps)
    plt.xlabel(r'$\lambda$ (log scale)')
    plt.ylabel('Error rate')
    plt.title(r'Error rate per $\lambda$')
    plt.show()


def run_experiment(sample_size, ls, iterations):
    """
    Executes the experiment for the given lambdas
    :param sample_size: amount of train data points to load for the experiment
    :param ls: list of lambda values to test
    :param iterations: number of times to run the experiment for each lambda value
    :return: results on the train and test data
    """
    results_train = []
    results_test = []
    for l in ls:
        result_l_train = []
        result_l_test = []
        for i in range(iterations):
            trainX, trainy, testX, testy = load_n_data(sample_size)
            w = softsvm(l, trainX, trainy)
            result_l_train.append(get_error_percentage(np.sign(trainX @ w), trainy))
            result_l_test.append(get_error_percentage(np.sign(testX @ w), testy))
        results_train.append([np.mean(result_l_train), np.max(result_l_train), np.min(result_l_train)])
        results_test.append([np.mean(result_l_test), np.max(result_l_test), np.min(result_l_test)])
    return results_train, results_test


def simple_test():
    # load question 2 data
    data = np.load('ex2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")
    print(f'The real label is {trainy[i]}')


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 2
    exps1 = range(1, 11)
    ls1 = [10**i for i in exps1]
    results_100_samples_train, results_100_samples_test = run_experiment(100, ls1, 10)
    plot_experiment_result(results_100_samples_train, results_100_samples_test, exps1)

    exps2 = [1, 3, 5, 8]
    ls2 = [10**i for i in exps2]
    results_1000_samples_train, results_1000_samples_test = run_experiment(1000, ls2, 1)
    plot_experiment_result(results_100_samples_train, results_100_samples_test, exps1,
                           results_1000_samples_train, results_1000_samples_test, exps2, is_small=False)
