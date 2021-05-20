import numpy as np
import matplotlib.pyplot as plt
import math

"""
a2map.py

This program reads in the data.npy file from Assignment 1, and creates a plot showing 
1) the training data, 
2) the decision boundary separating class 0 and 1, and 
3) the classification regions.

@author: Anushree Das (ad1707)
"""


def class_priors(y):
    """
    Calculate prior probablity, P(class_k), for each class
    :param y: class labels
    :return:  prior probabilities
    """
    # get unique class labels
    labels = np.unique(y)
    n = len(y)
    # create dictionary to store priors for each class
    priors = dict()

    # initialize each prior probability with 0
    for label in labels:
        priors[label] = 0

    # get total number of instances for each class
    for x in y:
        priors[x] += 1

    # divide by total number of samples to get probability
    for label in labels:
        priors[label] /= n

    return priors


def classwise_separate_data(X,y):
    """
    Group data in X based on the y class label
    :param X: sample input features
    :param y: output class labels
    :return:  dictionary with data in X grouped based on y label
    """
    labels = np.unique(y)
    n = len(y)
    classwise_data = dict()

    # initialize each classwise data with empty []
    for label in labels:
        classwise_data[label] = []

    # append row from X to respective [] of y label in dict
    for i in range(n):
        classwise_data[y[i]].append(X[i])

    return classwise_data


def mean_vectors(X,y):
    """
    Calculate mean vector for each class
    :param X: sample input features
    :param y: output class labels
    :return: dictionary with mean vectors for each class
    """
    classwise_means = dict()

    # get data separated based on class label
    classwise_data = classwise_separate_data(X,y)

    # calculate mean vector for each class
    for (label,data) in classwise_data.items():
        classwise_means[label] = [np.sum(x)/len(data) for x in zip(*data)]

    return classwise_means


def covariance_matrices(X,y,means):
    """
    Calculate covariance matrices for each class
    :param X: sample input features
    :param y: output class labels
    :param means: dictionary of class-wise mean vectors
    :return: dictionary with covariance matrices for each class
    """
    labels = np.unique(y)
    classwise_cov_mat = dict()
    no_features = len(X[0])

    # initialize covariance matrix for each class
    for label in labels:
        classwise_cov_mat[label] = [[0 for _ in range(no_features)]for _ in range(no_features)]

    # get data separated based on class label
    classwise_data = classwise_separate_data(X, y)

    # Calculate covariance matrices for each class
    for (label,data) in classwise_data.items():
        n = len(data)
        # calculate cov(i,j)
        for i in range(no_features):
            for j in range(no_features):
                sum = 0
                for k in range(n):
                    sum+=(data[k][i]-means[label][i])*(data[k][j]-means[label][j])
                classwise_cov_mat[label][i][j] = sum/(n-1)

    return classwise_cov_mat


def class_conditional(z,cov_mats,means):
    """
    Calculate p(z|class_k) for each class
    :param z:           input test vector
    :param cov_mats:    dict of class-wise covariance matrices
    :param means:       dict of class-wise mean vectors
    :return:            dict of class-wise conditional probabilities
    """
    n = len(z)
    labels = cov_mats.keys()
    class_prob = dict()
    pi = 3.14
    e = 2.72

    # calculate p(z|class_k) using gaussian distribution
    for label in labels:
        C = 1/math.sqrt(((2*pi)**n)*np.linalg.det(cov_mats[label]))
        diff =  (np.array(z-means[label])[np.newaxis]).T
        N = -(diff.T @ np.linalg.inv(cov_mats[label]) @ (diff))[0][0]
        class_prob[label] = (C * (e**(N/2)))

    return class_prob


def predict(priors,Z,cov_mats,means):
    """
    Predict class labels for list of input test samples using MAP
    :param priors:      dict of class-wise prior probabilities
    :param Z:           list of input test vectors
    :param cov_mats:    dict of class-wise covariance matrices
    :param means:       dict of class-wise mean vectors
    :return:            predicted class for input vectors
    """
    y_pred = []

    for z in Z:
        # get class conditional probabilities for vector z
        class_probs = class_conditional(z, cov_mats, means)

        MAP = dict()

        for label in priors.keys():
            # p(z|class_k) * P(class_k)
            MAP[label] = class_probs[label]*priors[label]

        # index of max p(z|class_k) * P(class_k)
        y_pred.append(max(MAP, key=MAP.get))

    return y_pred


def plot_training_samples(X, y):
    """
    Plots the training samples
    :param X: Input Features
    :param y: Output Class
    :return:  None
    """
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    # assign color for each input according to its output class
    colors = get_colors(y)
    # plot features
    plt.scatter(X[:, 0], X[:, 1],marker='o', s=20, facecolors='none',edgecolors=colors)


def get_colors(classlabels):
    """
    Returns array of colors based on output labels
    :param classlabels: Output labels
    :return: array of colors
    """
    # assign color for each input according to its output class
    colors = []
    for c in classlabels:
        if c == 0:
            colors.append('skyblue')
        else:
            if c == 1:
                colors.append('orange')
    return colors


def plot_classification(X,priors,cov_mats,means):
    """
    Plot decision boundary and decision region
    :param X: sample input features
    :param priors: dict of class-wise prior probabilities
    :param cov_mats: dict of class-wise covariance matrices
    :param means:    dict of class-wise mean vectors
    :return:
    """
    # find min and max values of both features
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # k-nearest neighbors classifier
    X_all = np.c_[x_vals.ravel(), y_vals.ravel()]
    # Predict class for all combinations of values of both features
    predictions = np.array(predict(priors,X_all,cov_mats,means))
    z = predictions.reshape(x_vals.shape)
    # draw decision boundary
    plt.contour(x_vals, y_vals, z, linewidths=0.5, levels=[0.9, 1], colors=['black'])

    # assign color for each input according to its output class
    colors = get_colors(predictions)
    # plot classification region
    plt.scatter(x_vals, y_vals, s=0.1, color=colors)

    plt.savefig('MAP_decision_boundary.png')
    plt.show()


def confusion_matrix(X,y,priors,cov_mats,means):
    """
    Print confusion matrix and classification rate
    :param X: sample input features
    :param y: output class labels
    :param priors: dict of class-wise prior probabilities
    :param cov_mats: dict of class-wise covariance matrices
    :param means: dict of class-wise mean vectors
    :return:
    """
    # get predicted value for training data samples
    y_pred = predict(priors, X, cov_mats, means)

    # create confusion matrix
    conf_mat = dict()
    # initialize confusion matrix
    for label1 in priors.keys():
        for label2 in priors.keys():
            conf_mat[(label1, label2)] = 0
    # calculate confusion matrix
    for i in range(len(y_pred)):
        conf_mat[(y_pred[i], y[i])] += 1

    # print confusion matrix
    print("{:<15} {:^20}".format(' ', 'Ground Truth'))
    print("{:>15}".format('Predictions'), end="")
    for label in priors.keys():
        print("{:^10}".format(label), end="")
    print('\n', '_' * (15+len(priors.keys())*10))
    for label1 in priors.keys():
        print("{:^15}".format(label1), end="")
        for label2 in priors.keys():
            print("{:^10}".format(conf_mat[(label1, label2)]), end="")
        print()
    print('_' * (15+len(priors.keys())*10))

    # calculate and print classification rate
    sum_correct = 0
    for label in priors.keys():
        sum_correct += conf_mat[(label, label)]
    rate = (sum_correct * 100) / len(y_pred)
    print('Classification rate:', round(rate,2), '%')


def main():
    # load data
    data = np.load("data.npy")
    # array of features
    X = data[:,:-1]
    # array of output class for corresponding feature set
    y = data[:,-1].astype(int)

    # calculate priors
    priors = class_priors(y)

    # calculate mean vectors for each class
    means = mean_vectors(X, y)

    # calculate covariance matrix for each class
    cov_mats = covariance_matrices(X, y, means)

    # plot data and decision boundary
    plot_training_samples(X, y)
    plot_classification(X, priors, cov_mats, means)

    # print confusion matrix and classification rate
    confusion_matrix(X, y, priors, cov_mats, means)


if __name__ == '__main__':
    main()
