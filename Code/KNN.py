import math
from itertools import combinations
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import random
from collections import Counter

# 注意knn实际上在高维空间中表现并不好，因为高维情况下，近邻也会离得很远。

def Lp_norm(x, y, p=2):
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        if p==0: #等于不相等的坐标个数
            for i in range(len(x)):
                sum +=(1 if abs(x[i] - y[i])>0 else 0)
        elif p==-1: #等于最大的单坐标差
            for i in range(len(x)):
                sum =max(sum, abs(x[i] - y[i]))
        else:
            for i in range(len(x)):
                sum += math.pow(abs(x[i] - y[i]), p)
            sum=math.pow(sum, 1 / p)
        return sum
    else:
        return 0
    
class my_KNN:
    def __init__(self, X_train, y_train, num_neighbors=8, p=2):
        self.n = num_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])

        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        return max_count

    def score(self, X_test, y_test, num_score=5000):
        if num_score>X_test.shape[0]:
            num_score=X_test.shape[0]
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

class KNN:
    def __init__(self, X_train, y_train, num_neighbors=5, p=2):
        self.knn_model=KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=-1)
        self.knn_model.fit(X_train, y_train)
        print("[Info]: K-nearest Neighbor model are trained!!!")

    def predict(self, X):
        knn_ans=self.knn_model.predict(X)
        #print("[Info]: K-nearest Neighbor has made predictions!!!")
        # print(knn_ans)
        ans = []
        for index, value in enumerate(knn_ans):
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
            # print (label[0][1], y)
            if label[0][1] == int(y+0.5):
                right_count += 1
            if now_count>=num_score:
                break
        print('[Info]: Score the K-nearest Neighbor model though ', num_score, ' sample point in train set.')
        return right_count/now_count
        return right_count / len(X_test)