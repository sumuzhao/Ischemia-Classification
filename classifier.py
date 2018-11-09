import numpy as np
import pandas as pd
from glob import glob
from os import path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import andrews_curves, parallel_coordinates, radviz
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# x = glob('./trainingsets/*.npy')
# x.sort(key=lambda x: path.splitext(path.basename(x))[0][1:5])
# print(x)
# c_n = 0
# c_st = 0
# for i in x:
#     if path.splitext(path.basename(i))[0][6:] in ['ST0-', 'ST1-', 'ST0+', 'ST1+']:
#         c_st += np.load(i).shape[0]
#     else:
#         c_n += np.load(i).shape[0]
# print("ST {} and normal {}.".format(c_st, c_n))
# d = np.load(x[0])
# for i in x[1:]:
#     d = np.concatenate((d, np.load(i)))
# print(d.shape)
#
# # np.random.shuffle(d)
# X_train = d[:int(d.shape[0] * 0.9), :-1]
# X_test = d[int(d.shape[0] * 0.9):, :-1]
# y_train = d[:int(d.shape[0] * 0.9), -1] * 2. - 1.
# y_test = d[int(d.shape[0] * 0.9):, -1] * 2. - 1.
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = np.load('./trainingsets/edb_2018_11_08/X_train.npy')
X_test = np.load('./trainingsets/edb_2018_11_08/X_test.npy')
y_train = np.load('./trainingsets/edb_2018_11_08/y_train.npy')
y_test = np.load('./trainingsets/edb_2018_11_08/y_test.npy')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# np.save('./trainingsets/edb_2018_11_08/X_train.npy', X_train)
# np.save('./trainingsets/edb_2018_11_08/X_test.npy', X_test)
# np.save('./trainingsets/edb_2018_11_08/y_train.npy', y_train)
# np.save('./trainingsets/edb_2018_11_08/y_test.npy', y_test)

X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test)) / 2 + 1 / 2
print(X.shape, y.shape)

skb = SelectKBest(f_classif, k=15)
X = skb.fit_transform(X, y)
print(skb.get_support(), skb.get_params(deep=True))

# Min Max scaling
mms = MinMaxScaler()
X = mms.fit_transform(X)

# ddd = np.concatenate((X, y.reshape([y.shape[0], 1])), axis=1)
# print(ddd.shape)
# normal = ddd[ddd[:, -1] == -1]
# print(normal.shape)

X_train = X[:int(X.shape[0] * 0.8)]
X_test = X[int(X.shape[0] * 0.8):int(X.shape[0] * 0.9)]
X_valid = X[int(X.shape[0] * 0.9):]
y_train = y[:int(y.shape[0] * 0.8)]
y_test = y[int(X.shape[0] * 0.8):int(X.shape[0] * 0.9)]
y_valid = y[int(X.shape[0] * 0.9):]
print(X_train.shape, X_test.shape, X_valid.shape, y_train.shape, y_test.shape, y_valid.shape)

# Xy_test = np.concatenate((X_test, y_test.reshape([y_test.shape[0], 1])), axis=1)
# Xy_test_plot = Xy_test[np.random.choice(Xy_test.shape[0], 200, False)]
#
# dt = pd.DataFrame(Xy_test_plot, columns=["f1", "f2", "f3", "f4", "f5", "label"])
# print(dt)
# plt.figure()
# radviz(dt, 'label')
# plt.show()

# SVM
# clf = SVC()
# clf.fit(X_train, y_train)
# print("SVM:", clf.score(X_valid, y_valid))

# KNN
# clf = KNeighborsClassifier(n_neighbors=10)
# clf.fit(X_train, y_train)
# print("KNN({})".format(clf.n_neighbors), clf.score(X_valid, y_valid))
# print(clf.score(Xy_test_plot[:, :X_test.shape[1]], Xy_test_plot[:, X_test.shape[1]]))

# RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=10)
clf.fit(X_train, y_train)
print("RF({}):".format(clf.n_estimators), clf.score(X_valid, y_valid))
# print("RF({}) oob:".format(clf.n_estimators), clf.oob_score_)
print(clf.feature_importances_)

# param_test1 = {'n_estimators':range(100, 500, 50)}
# gsearch1 = GridSearchCV(estimator=clf, param_grid=param_test1)
# gsearch1.fit(X_train, y_train)
# print(gsearch1.best_score_, gsearch1.best_estimator_)

# joblib.dump(clf, './clf_model/rf_clf_2018_11_08.model')
# clf = joblib.load('./clf_model/rf_clf_2018_11_08.model')
# # print(clf)
# print(clf.score(X_test, y_test))
# target_names = ['ST', 'Normal']
# print(classification_report(y_test, clf.predict(X_test), target_names=target_names))
#
# print(clf.score(X, y))
# print(y)
# print(clf.predict(X))