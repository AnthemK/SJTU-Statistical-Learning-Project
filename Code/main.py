import numpy as np

import SVD_Dimen_Reduc
import PCA_Dimen_Reduc

import IO_csv
import KNN
import NaiveBayes
import DecisionTree
import Logistic_Regression
import AdaBoost

from sklearn.model_selection import KFold
# https://liaoxuefeng.com/books/python/built-in-modules/venv/
# 需要测试降维到多少维度可以
# PCA
# 测试集是数据集加了一些噪声。
# 如果数据比较粗糙，标记的时候可能会有问题
# 用预训练的模型，看训练在嵌入空间中的分布
# 模型越简单正确性越高
# 测试集的分类很乱
# 数据到验证的整个思路走一遍，对于性能要求不高
# dropout非常高，高bias的方案更好用

def do_classify(X_trans, y_trans, X_vali, y_vali, classify_type=2):
    match 2:    
        case 1: # knn model
            print('[Info]: Using the K-nearest Neighbor model.')
            KNN_model=KNN.KNN(X_trans, y_trans.ravel(), num_neighbors=12) # num_neighbors=12, acc=0.031
            knn_score = KNN_model.score(X_vali, y_vali.ravel(), num_score=5000)
            print('[Result]: the acc of K-nearest Neighbor model is', knn_score)
            ans = KNN_model.predict(test_data)

        case 2:
            print("[Info]: Using the Naive Bayes model.")
            NB_model=NaiveBayes.my_NaiveBayes(X_trans, y_trans.ravel())
            NB_score = NB_model.score(X_vali, y_vali.ravel(), num_score=10000)
            print('[Result]: the acc of Naive Bayes model is', NB_score)
            ans = NB_model.predict(test_data)

        case 3: # score=0.5252
            print("[Info]: Using the Decision Tree model.")
            DT_model=DecisionTree.DecisionTree(X_trans, y_trans.ravel())
            DT_score = DT_model.score(X_vali, y_vali.ravel(), num_score=10000)
            print('[Result]: the acc of Decision Tree model is', DT_score)
            ans = DT_model.predict(test_data)

        case 4: #solver=‘lbfgs’, score=0.6764
            print("[Info]: Using the Logistic Regression model.")
            # LR_model=Logistic_Regression.Logistic_Regression(X_trans, y_trans.ravel())
            # LR_score = LR_model.score(X_vali, y_vali.ravel(), num_score=10000)
            # print('[Result]: the acc of Logistic Regression model is', LR_score)
            # ans = LR_model.predict(test_data)


            LR_model=Logistic_Regression.my_Logistic_Regression(X_trans, y_trans.ravel())
            LR_score = LR_model.score(X_vali, y_vali.ravel(), num_score=10000)
            print('[Result]: the acc of Logistic Regression model is', LR_score)
            ans = LR_model.predict_all(test_data)

        case 5: 
            print("[Info]: Using the AdaBoost model.")
            AB_model=AdaBoost.AdaBoost(X_trans, y_trans.ravel(), n_estimators=120, learning_rate=0.999)
            AB_score = AB_model.score(X_vali, y_vali.ravel(), num_score=10000)
            print('[Result]: the acc of AdaBoost model is', AB_score)
            ans = AB_model.predict(test_data)
    return ans

(X_trans, y_trans) = IO_csv.Input_train_data(shuffle_row=1)
test_data = IO_csv.Input_test_data()
print("[Info]: Input train and test data finished!!")

# SVD_DR=SVD_Dimen_Reduc.SVD_Dimention_Reduction(remain_components=400)
# X_new_trans=SVD_DR.SVD_dimen_reduct(X_trans)
# X_trans=X_new_trans
# print(X_new_trans.shape[0], X_new_trans.shape[1])

dimensionality_reduction_type = 1

match dimensionality_reduction_type:    
    case 1: #PCA by sklearn
        PCA_model=PCA_Dimen_Reduc.PCA_Dimension_Reduction(remain_components=400)
        X_trans=PCA_model.PCA_dimen_reduct(X_trans)
        #print(X_trans.shape)
        #print(PCA_model.do_Regulation(X_trans))
        #print(X_new_trans)
        #print(PCA_model.do_recov(X_new_trans))
        test_data=np.asarray(PCA_model.do_project(test_data))
    case 2: #PCA by myself
        PCA_model=PCA_Dimen_Reduc.PCA_Dimension_Reduction(remain_components=400)
        PCA_model.PCA_myself(X_trans)
        X_trans=PCA_model.project_data(X_trans)
        test_data=PCA_model.project_data(test_data)
        
    case 3:
        SVD_model=SVD_Dimen_Reduc.SVD_Dimension_Reduction(remain_components=400)
        X_y=np.append(X_trans,test_data,axis=0)
        X_y=SVD_model.SVD_dimen_reduct(X_y)
        X_trans=X_y[:19573, :]
        test_data=X_y[19573:, :]

model_selection_type=2
match model_selection_type:   
    case 0:
        X_vali=X_trans
        y_vali=y_trans
        ans=do_classify(X_trans, y_trans, X_vali, y_vali, classify_type=4)
    case 1: 
        size_of_train_set = 19000
        X_vali=X_trans[size_of_train_set:, :]
        y_vali=y_trans[size_of_train_set:, :]
        X_trans=X_trans[:size_of_train_set, :]
        y_trans=y_trans[:size_of_train_set, :]
        ans=do_classify(X_trans, y_trans, X_vali, y_vali, classify_type=4)
    case 2: 
        kf = KFold(n_splits = 5)
        for trainind,validind in kf.split(X_trans):
            print("KFold(CV): train:%s,valid:%s"%(trainind,validind))
            ans=do_classify(X_trans[trainind], y_trans[trainind], X_trans[validind], y_trans[validind], classify_type=1)

IO_csv.Output_ans(ans)
