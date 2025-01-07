import math
from matplotlib import pyplot as plt
from itertools import combinations
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree

class DecisionTree:
    def __init__(self, X_train, y_train):
        self.DT_model=DecisionTreeClassifier() 
        self.DT_model.fit(X_train, y_train)
        print("[Info]: Decision Tree model are trained!!!")

    def predict(self, X):
        NB_ans=self.DT_model.predict(X)
        ans = []
        for index, value in enumerate(NB_ans):
            ans.append([index, int(value+0.5)])
        return ans

    def score(self, X_test, y_test, num_score=5000):
        if num_score>X_test.shape[0]:
            num_score=X_test.shape[0]
        right_count = 0
        now_count = 0
        test_data=random.sample(list(zip(X_test, y_test)), num_score)
        for X, y in test_data:
            now_count+=1
            # print(X.reshape(1,-1), y)
            label = self.predict(X.reshape(1,-1))
            #print (label[0][0])
            if label[0][1] == int(y+0.5):
                right_count += 1
            if now_count>=num_score:
                break
        print('[Info]: Score the Decision Tree model though ', num_score, ' sample point in train set.')
        return right_count/now_count