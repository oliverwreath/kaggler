"""
============================
Nearest Neighbors regression
============================

Demonstrate the resolution of a regression problem
using a k-Nearest Neighbor and the interpolation of the
target using both barycenter and constant weights.

"""
print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#
# License: BSD 3 clause (C) INRIA


###############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, linear_model, svm 
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor


np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

###############################################################################
# Fit regression model
n_neighbors = 5

names = [
	'KNeighborsRegressor uniform', 'KNeighborsRegressor distance', "1.1.2. Ridge Regression",
	'1.1.6. Least Angle Regression', "1.1.9.1. Bayesian Ridge Regression", "1.4. Support Vector Machines",
	"1.5. Stochastic Gradient Descent", "1.6. Nearest Neighbors", "Radius KNN"
]
classifiers = [
	KNeighborsRegressor(n_neighbors, weights='uniform'),
	KNeighborsRegressor(n_neighbors, weights='distance'), 
	linear_model.Ridge (alpha = .5), 
	
	linear_model.Lars(), 
	linear_model.BayesianRidge(), 
	svm.SVR(), 

	linear_model.SGDRegressor(),
	KNeighborsRegressor(),
	RadiusNeighborsRegressor()
]

i = 0 
plt.plot(3,3)
for name, classifier in zip(names, classifiers): 
    y_ = classifier.fit(X, y).predict(T)

    
    plt.subplot( len(names), 1, i + 1 )
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('auto')
    plt.legend()
    plt.title(name)
    i += 1

plt.show()
