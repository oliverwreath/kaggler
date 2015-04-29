#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
Classifier comparison
=====================

A comparison of a several classifiers in scikit-learn on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
of different classifiers.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

Particularly in high-dimensional spaces, data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization than is achieved by other classifiers.

The plots show training points in solid colors and testing points
semi-transparent. The lower right shows the classification accuracy on the test
set.
http://scikit-learn.org/dev/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets, svm, linear_model, neighbors, tree, ensemble 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier


h = .02  # step size in the mesh

names = ["Generalized Linear Models", "LDA", "QDA", "Linear SVM", "RBF SVM", 
            "Stochastic Gradient Descent", "Nearest Neighbors", 
            "Naive Bayes", "Decision Tree",
            "Random Forest", "Extremely Randomized Trees", 
            "AdaBoost", "Gradient Tree Boosting"
            ]
classifiers = [
    linear_model.LogisticRegression(C=1e5), 
    LDA(),
    QDA(),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    
    linear_model.SGDClassifier(loss="log"), 
    KNeighborsClassifier(3), 
    
    GaussianNB(), 
    DecisionTreeClassifier(max_depth=5), 

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    ExtraTreesClassifier(n_estimators=10, max_depth=None,
            min_samples_split=1, random_state=0), 

    AdaBoostClassifier(),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0).fit(X_train, y_train) 
    ]
# #1.1. Generalized Linear Models
# classifier = linear_model.LogisticRegression(C=1e5)
# #1.2. Linear and quadratic discriminant analysis
# classifier = LDA()
# classifier = QDA()
# #1.4. Support Vector Machines
# classifier = svm.SVC(kernel='linear', C=0.01)
# classifier = svm.SVC(gamma=2, C=1)
# #1.5. Stochastic Gradient Descent
# classifier = linear_model.SGDClassifier(loss="log")
# #1.6. Nearest Neighbors
# classifier = neighbors.KNeighborsClassifier()
# #1.9. Naive Bayes
# classifier = GaussianNB()
# #1.10. Decision Trees
# classifier = tree.DecisionTreeClassifier()
# #1.11. Ensemble methods
# #>1.11.1. Bagging meta-estimator
# classifier = ensemble.BaggingClassifier(KNeighborsClassifier(),
#     max_samples=0.5, max_features=0.5)
# #>1.11.2. Forests of randomized trees
# #>>1.11.2.1. Random Forests
# classifier = RandomForestClassifier(n_estimators=10)
# #>>1.11.2.2. Extremely Randomized Trees
# classifier = ExtraTreesClassifier(n_estimators=10, max_depth=None,
#             min_samples_split=1, random_state=0)
# #>1.11.3. AdaBoost
# classifier = AdaBoostClassifier(n_estimators=100)
# #>1.11.4. Gradient Tree Boosting
# classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
#         max_depth=1, random_state=0).fit(X_train, y_train)
# #1.15. Isotonic regression
# classifier = IsotonicRegression()

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()
