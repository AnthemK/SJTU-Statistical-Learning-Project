import math
from itertools import combinations
import numpy as np
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB

class my_NaiveBayes:
    def __init__(self, X, y):
        self.model = None
        self.total_num=len(X)
        self.Multinomial_lambda=0.5
        self.fit(X, y)

    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    def gaussian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) /
                              (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
        
    def multinomial_probability(self, x, m_k=1):
        return (x+self.Multinomial_lambda)/(m_k+self.total_num*self.Multinomial_lambda)

    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)if self.stdev(i)>0 else 0.0000001) for i in zip(*train_data)]
        return summaries

    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(value)
            for label, value in data.items()
        }
        self.dataset=data
        return 1
    
    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            m_label=len(self.dataset[label])
            # print(label, '=', m_label)
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
                #probabilities[label] *= self.multinomial_probability(input_data[i], m_k=m_label)
        return probabilities

    def predict_one(self, X_test):
        label = sorted(
            self.calculate_probabilities(X_test).items(),
            key=lambda x: x[-1])[-1][0]
        return label
    
    def predict(self, X_test):
        NB_ans=[]
        for i in range(0, len(X_test)):
            NB_ans.append(self.predict_one(X_test[i]))
        ans = []
        for index, value in enumerate(NB_ans):
            ans.append([index, int(value+0.5)])
        return ans

    def score(self, X_test, y_test, num_score=5000):
        if num_score>X_test.shape[0]:
            num_score=X_test.shape[0]
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict_one(X)
            if label == y:
                right += 1

        return right / float(len(X_test))

class NaiveBayes:
    def __init__(self, X_train, y_train, type=1):
        match type:
            case 1:
                self.NB_model=GaussianNB() # acc=0.5518
                print("[Info]: Gaussian Naive Bayes model are trained!!!")
            case 2:
                self.NB_model=BernoulliNB() # acc=0.6346
                print("[Info]: Bernoulli Naive Bayes model are trained!!!")
            case 3:
                self.NB_model=MultinomialNB() # acc=0.6840
                print("[Info]: Multinomial Naive Bayes model are trained!!!")
            case 4:
                self.NB_model=CategoricalNB() # 用于离散变量的分类，不适合使用在实数数据上，不过硬要用也可以. 
                X_train=np.multiply(X_train, 1500)
                X_train=np.round(X_train)
                # print(X_train) # 当放大1000倍的时候，acc=0.0092，放大300倍的时候，acc=0.0092，放大1500倍的时候，acc=0.0092。都很低，不建议使用这个方法
                print("[Info]: Categorical Gaussian Naive Bayes model are trained!!!")
            case 5:
                self.NB_model=ComplementNB() # acc=0.5626
                print("[Info]: Complement Gaussian Naive Bayes model are trained!!!")
        self.NB_model.fit(X_train, y_train)

    def predict(self, X):
        NB_ans=self.NB_model.predict(X)
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
        print('[Info]: Score the Naive Bayes model though ', num_score, ' sample point in train set.')
        return right_count/now_count