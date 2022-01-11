from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import numpy as np

## part 1 parameters
to_plot=True
## part 2 parameters
mix_data=True # mix the data for different testing sets if True
test_size_ratio=0.2 # %of data for testing
## part 2.1 parameters
n_neighbors=3 ## num of neighbors for knn check
## part 2.2 parameters
max_iterations=1000 ## num of max iterations for Logistic regression


iris = load_iris()

## part 1

if to_plot:
    data_T=iris.data.T

    s_l=data_T[0]
    s_w=data_T[1]
    p_l=data_T[2]
    p_w=data_T[3]

    s_l_label=iris.feature_names[0]
    s_w_label=iris.feature_names[1]
    p_l_label=iris.feature_names[2]
    p_w_label=iris.feature_names[3]

    fig = plt.figure()

    ax = fig.add_subplot(2, 2, 1,projection='3d')

    ax.scatter(s_l, s_w, p_l, c=iris.target)
    ax.set_xlabel(s_l_label)
    ax.set_ylabel(s_w_label)
    ax.set_zlabel(p_l_label)

    ax = fig.add_subplot(2, 2, 2,projection='3d')

    ax.scatter(s_l, s_w, p_w, c=iris.target)
    ax.set_xlabel(s_l_label)
    ax.set_ylabel(s_w_label)
    ax.set_zlabel(p_w_label)
    ax = fig.add_subplot(2, 2, 3, projection='3d')

    ax.scatter(s_l, p_l, p_w, c=iris.target)
    ax.set_xlabel(s_l_label)
    ax.set_ylabel(p_l_label)
    ax.set_zlabel(p_w_label)
    ax = fig.add_subplot(2, 2, 4, projection='3d')

    ax.scatter(s_w, p_l, p_w, c=iris.target)
    ax.set_xlabel(s_w_label)
    ax.set_ylabel(p_l_label)
    ax.set_zlabel(p_w_label)

    plt.show(block=False)


## part 2

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],test_size=test_size_ratio, shuffle=mix_data)


## part 2.1

knn=KNeighborsClassifier(n_neighbors)
knn.fit(X_train,y_train)

print('KNN algorithm with ' + str(n_neighbors) + ' neighbors check, done with: ' + str(round(knn.score(X_test, y_test),4)*100) + '% score')


## part 2.2

lr=LogisticRegression( max_iter=max_iterations)
lr.fit(X_train,y_train)

print('LR algorithm done with: ' + str(round(lr.score(X_test, y_test),4)*100) + '% score')

if to_plot:
    plt.show()
