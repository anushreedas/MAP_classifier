# MAP_classifier
Maximum a posteriori (MAP) classifier
a2map.py
Implements a maximum a posteriori (MAP) classifier, using Gaussian distributions to estimate class-conditional densities.
The program reads in the data.npy file, and creates a plot showing 1) the training data, 2) the decision boundary separating class 0 and 1, and 3) the classification regions.

a2cost.py
This program uses (approximated)Bayesian classifier from a2map, but with different cost functions:
first, with cost function with manually defined cost matrix,
second, with uniform cost function,
lastly, with prior probability of the scrap class(4) increased to 0.5

It uses the nuts_bolts.csv file from Duin et al.’s “Classification, Parameter Es-timation and State Estimation,”

