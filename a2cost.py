import numpy as np
import matplotlib.pyplot as plt
import math

"""
a2cost.py

This program reads in the nuts_bolts.npy file, and creates 3 plots 
using the modified Bayesian classifier defined in a2map.py,
first, with cost function with manually defined cost matrix,
second, with uniform cost function,
lastly, with prior probability of the scrap class(4) increased to 0.5

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


def cost_function_one(assigned_class,true_class):
    cost_mat = [[-0.20,0.07,0.07,0.07],
                [0.07,-0.15,0.07,0.07],
                [0.07,0.07,-0.05,0.07],
                [0.03,0.03,0.03,0.03]]
    return cost_mat[int(assigned_class)-1][int(true_class)-1]

def cost_function_two(assigned_class,true_class):
    cost_mat = [[-0.20,0.07,0.07,0.07],
                [0.07,-0.15,0.07,0.07],
                [0.07,0.07,-0.05,0.07],
                [1.80,2.35,1.90,-0.03]]
    return cost_mat[int(assigned_class)-1][int(true_class)-1]


def uniform_cost_function(assigned_class,true_class):
    if assigned_class == true_class:
        return 0
    else:
        return 1

def predict(priors,Z,cov_mats,means,cost_function):
    """
    Predict class labels for list of input test samples using modified MAP
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

        for label1 in priors.keys():
            sum = 0
            for label2 in priors.keys():
                # Cost(class_i,class_k) * p(z|class_k) * P(class_k)
                sum += cost_function(label1,label2)*class_probs[label2]*priors[label2]
            MAP[label1] = sum

        # index of min Cost(class_i,class_k) * p(z|class_k) * P(class_k)
        y_pred.append(min(MAP, key=MAP.get))

    return y_pred


def plot_training_samples(X, y,plotname):
    """
    Plots the training samples
    :param X: Input Features
    :param y: Output Class
    :return:
    """
    plt.xlabel('measure of eccentricity')
    plt.ylabel('measure of six-fold rotational symmetry')
    plt.title(plotname)

    # markers for different classes
    markers = dict()
    markers[1] =  '+'
    markers[2] = '*'
    markers[3] = 'o'
    markers[4] = 'x'
    # facecolors for different classes
    facecolors = dict()
    facecolors[1] = 'black'
    facecolors[2] = 'black'
    facecolors[3] = 'none'
    facecolors[4] = 'black'

    # plot all classes with different marker
    classwise_data = classwise_separate_data(X, y)

    for (label, data) in classwise_data.items():
        x_vals = [d[0] for d in data]
        y_vals = [d[1] for d in data]
        plt.scatter(x_vals,y_vals,s=17,marker=markers[label],facecolors=facecolors[label],edgecolors='black')

    plt.savefig(plotname + '.png')
    plt.show()

def get_colors(classlabels):
    """
    Returns array of colors based on output labels
    :param classlabels: Output labels
    :return: array of colors
    """
    # assign color for each input according to its output class
    colors = []
    for c in classlabels:
        # print(c)
        if c == 1:
            colors.append('grey')
        else:
            if c == 2:
                colors.append('red')
            else:
                if c == 3:
                    colors.append('blue')
                else:
                    if c == 4:
                        colors.append('yellow')
    return colors


def plot_classification(X,priors,cov_mats,means,cost_function):
    """
    Plot decision boundary and decision region
    :param X: sample input features
    :param priors: dict of class-wise prior probabilities
    :param cov_mats: dict of class-wise covariance matrices
    :param means:    dict of class-wise mean vectors
    :param cost_function: cost function for classifier
    :return: None
    """
    # find min and max values of both features
    x_min, x_max = X[:, 0].min()-0.2, X[:, 0].max()+0.2
    y_min, y_max = X[:, 1].min()-0.2, X[:, 1].max()+0.2

    x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # k-nearest neighbors classifier
    X_all = np.c_[x_vals.ravel(), y_vals.ravel()]
    # Predict class for all combinations of values of both features
    predictions = np.array(predict(priors,X_all,cov_mats,means,cost_function))
    z = predictions.reshape(x_vals.shape)
    # draw decision boundary
    # plt.contour(x_vals, y_vals, z, linewidths=[0.5,0.6,0.7,0.8], levels=[1,2,3,4], colors=['black'])

    # assign color for each input according to its output class
    colors = get_colors(predictions)
    # plot classification region
    plt.scatter(x_vals, y_vals, s=0.1, color=colors)




def confusion_matrix(X,y,priors,cov_mats,means,cost_function_one):
    """
    Print confusion matrix and classification rate
    :param X: sample input features
    :param y: output class labels
    :param priors: dict of class-wise prior probabilities
    :param cov_mats: dict of class-wise covariance matrices
    :param means: dict of class-wise mean vectors
    :param cost_function: cost function for classifier
    :return: None
    """
    # get predicted value for training data samples
    y_pred = predict(priors,X,cov_mats,means,cost_function_one)

    # create confusion matrix
    mat = dict()
    # initialize confusion matrix
    for label1 in priors.keys():
        for label2 in priors.keys():
            mat[(label1,label2)] = 0
    # calculate confusion matrix
    for i in range(len(y_pred)):
        mat[(y_pred[i], y[i])] += 1

    # print confusion matrix
    print("{:<15} {:^20}".format(' ', 'Ground Truth'))
    print("{:>15}".format('Predictions'),end="")
    for label in priors.keys():
        print("{:^10}".format(label),end="")
    print('\n','_'*(15+len(priors.keys())*10))
    for label1 in priors.keys():
        print("{:^15}".format(label1), end="")
        for label2 in priors.keys():
            print("{:^10}".format(mat[(label1,label2)]), end="")
        print()
    print('_' * (15+len(priors.keys())*10))

    # calculate and print classification rate
    sum_correct = 0
    for label in priors.keys():
        sum_correct += mat[(label,label)]
    rate = (sum_correct*100)/len(y_pred)
    print('Classification rate:',round(rate,2),'%')

def plot_decision_and_confusion_mat(X,y,priors,cov_mats,means,cost_function,plotname):
    """
    :param X: sample input features
    :param y: output class labels
    :param priors: dict of class-wise prior probabilities
    :param cov_mats: dict of class-wise covariance matrices
    :param means: dict of class-wise mean vectors
    :param cost_function: cost function for classifier
    :return: None
    """

    plot_classification(X, priors, cov_mats, means, cost_function)
    plot_training_samples(X, y,plotname)
    confusion_matrix(X, y, priors, cov_mats, means, cost_function)

def main():
    # load data
    data = np.genfromtxt("nuts_bolts.csv",delimiter=',')

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

    print(priors)

    print('\033[1m'+"Cost function with manually defined cost matrix"+'\033[0m')
    # plot for cost function with manually defined cost matrix
    plot_decision_and_confusion_mat(X, y, priors, cov_mats, means, cost_function_one,"Manually defined cost matrix")

    print('\033[1m'+"Uniform cost function"+'\033[0m')
    # plot for uniform cost function
    plot_decision_and_confusion_mat(X, y, priors, cov_mats, means, uniform_cost_function,"Uniform cost function")

    print('\033[1m'+"Prior probability of the scrap class(4) increased to 0.5"+'\033[0m')
    # increase prior probability of the scrap class(4) to 0.5
    # and decrease prior probabilities of other classes evenly
    k = 0.5
    n = 0
    for label in priors.keys():
        if label != 4:
            n += priors[label]
    for label in priors.keys():
        if label == 4:
            priors[label] = k
        else:
            priors[label] = (priors[label] * (1-k))/n
    # plot for when prior probability of scrap is incresed to 0.5
    plot_decision_and_confusion_mat(X, y, priors, cov_mats, means, cost_function_one,"Scrap prior prob 0.5")

    print('\033[1m'+"Amount of scrap class(4) doubled"+'\033[0m')
    # double the amount of scrap class(4) and calculate new priors
    for label in priors.keys():
        if label == 4:
            priors[label] = (2*priors[label]*100)/(100+priors[label]*100)
        else:
            priors[label] = (priors[label]*100)/(100+priors[label]*100)

    print(priors)

    # plot for when amount of scrap is doubled
    plot_decision_and_confusion_mat(X, y, priors, cov_mats, means, cost_function_two,"Scrap amount doubled")


if __name__ == '__main__':
    main()
