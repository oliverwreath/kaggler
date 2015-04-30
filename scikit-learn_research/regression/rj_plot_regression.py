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
from sklearn import neighbors, linear_model, svm, tree 
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor

np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
T = np.linspace(0, 5, 500)[:, np.newaxis]
y_test = np.sin(T).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

###############################################################################
# Fit regression model
n_neighbors = 5

names = [
	"1.1.2. Ridge Regression", '1.1.6. Least Angle Regression', "1.1.9.1. Bayesian Ridge Regression", 
	"1.4. Support Vector Machines", "1.5. Stochastic Gradient Descent", 
	"1.6. Nearest Neighbors uniform", "1.6. Nearest Neighbors distance", "Radius KNN",
	"1.10. Decision Trees", "1.11.1. Bagging", 
	"Random Forest", "Extremely Randomized Trees", 
	"AdaBoost", "Gradient Tree Boosting" 
]
classifiers = [
	linear_model.Ridge (alpha = .5), 
	linear_model.Lars(), 
	linear_model.BayesianRidge(), 

	svm.SVR(), 
	linear_model.SGDRegressor(),

	KNeighborsRegressor(n_neighbors, weights='uniform'),
	KNeighborsRegressor(n_neighbors, weights='distance'), 
	RadiusNeighborsRegressor(), 

	tree.DecisionTreeRegressor(), 
    BaggingRegressor(),

    RandomForestRegressor(max_depth=5, n_estimators=10, max_features=1),
    ExtraTreesRegressor(n_estimators=10, max_depth=None,
            min_samples_split=1, random_state=0), 

    AdaBoostRegressor(),
    GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0) 
]

i = 1 
x_num = 7#len(names)
y_num = 2
X_size = 1.7
Y_size = 7.0
plt.subplots(x_num, y_num, sharey = True, figsize=(y_num * Y_size, x_num * X_size) )

for name, classifier in zip(names, classifiers): 
    y_ = classifier.fit(X, y).predict(T)
    score = classifier.score(T, y_test)
    
    plt.subplot( x_num, y_num, i )
    plt.scatter(X, y, c='k', label='data')
    plt.plot(T, y_, c='g', label='prediction')
    plt.axis('auto')
    plt.legend()
    plt.title(name+" "+str(score))
    i += 1

plt.subplots_adjust(hspace=0.4)
plt.show()
