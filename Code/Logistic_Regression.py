from math import exp
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

class my_Logistic_Regression:
    def __init__(self, X, y, max_iter=10, learning_rate=0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights={}
        for i in range(100):
            y_tmp=y-i
            y_tmp[np.nonzero(y_tmp)]=1
            self.fit_bin(X, y_tmp, i)
            if i%10 ==0:
                print('[Info]: Already trained', i, 'model for binary classification.')
        #print(self.weights)

    def sigmoid(self, x):
        return np.divide(1, (1+np.exp(-x)))

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat

    def matrixing_y(self, y): #unused
        total_label=len(list(set(y)))
        matr_y=np.zeros((len(y), total_label), dtype=np.float32)
        # print(y)
        for index, value in enumerate(y):
            matr_y[index][int(value+0.5)]=1
        # print(matr_y)
        return matr_y

    def fit_bin(self, X, y, ind):
        data_mat = self.data_matrix(X)  
        self.weights[ind] = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights[ind]))
                error = y[i] - result
                self.weights[ind] += self.learning_rate * error * np.transpose(
                    [data_mat[i]])

    def fit(self, X, y): # Error:Unfinished
        data_mat = self.data_matrix(X)  
        total_label=len(list(set(y)))
        matr_y=self.matrixing_y(y)
        self.weights = np.zeros((len(data_mat[0]), total_label), dtype=np.float32)
        #print(self.weights.shape)
        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = matr_y[i] - result
                # print(error)
                error_value=np.sum(np.abs(error))
                # print(error_value)
                self.weights += self.learning_rate * error_value * np.transpose(
                    [data_mat[i]])
                # unfinished！！

    def predict_bin(self, X_test, ind):
        result = np.dot(X_test, self.weights[ind])
        return result
    
    def predict(self, X_test):
        #print(X_test, X_test.shape)
        ans_ind=0
        ans_val=1000000000
        for i in range(100):
            if(ans_val>self.predict_bin(X_test, i)):
                ans_ind=i
                ans_val=self.predict_bin(X_test, i)
        return ans_ind

    def predict_all(self, X_test):
        X_test = self.data_matrix(X_test)
        LR_ans=[]
        ans=[]
        for x in X_test:
            LR_ans.append(self.predict(np.array(x)))
        for index, value in enumerate(LR_ans):
            ans.append([index, int(value+0.5)])
        return ans

    def score(self, X_test, y_test, num_score=5000):
        if num_score>X_test.shape[0]:
            num_score=X_test.shape[0]
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = self.predict(np.array(x))
            if (result == y):
                right += 1
        return right / len(X_test)

class Logistic_Regression:
    def __init__(self, X_train, y_train, max_iter = 150, solver = 'lbfgs'): # ‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’
        self.LR_model=LogisticRegression(max_iter=max_iter, solver=solver) 
        self.LR_model.fit(X_train, y_train)
        print("[Info]: Logistic Regression model are trained!!!")

    def predict(self, X):
        LR_ans=self.LR_model.predict(X)
        ans = []
        for index, value in enumerate(LR_ans):
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
        print('[Info]: Score the Logistic Regression model though ', num_score, ' sample point in train set.')
        return right_count/now_count