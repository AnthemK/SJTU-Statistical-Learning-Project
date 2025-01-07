import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCA_Dimension_Reduction:
    def __init__(self, remain_components=512):
        self.n_components=remain_components
        self.PCA_model=PCA(n_components=remain_components)
    
    def do_Regulation(self, X):
        scaler = StandardScaler()
        scaler.fit(X)
        x_regu = scaler.transform(X)
        return x_regu

    def do_PCA(self, X):
        self.PCA_model.fit(X)   # 拟合数据，n_components定义要降的维度
        Z = self.PCA_model.transform(X)    # transform就会执行降维操作
        return Z
    
    def do_recov(self, X):
        Ureduce = self.PCA_model.components_     # 得到降维用的Ureduce
        x_rec = np.dot(X,Ureduce)       # 数据恢复
        return x_rec
    
    def do_project(self, X):
        Ureduce = self.PCA_model.components_     # 得到降维用的Ureduce
        Z = X * np.asmatrix(Ureduce).I 
        #Z = np.dot(X, np.linalg.inv(Ureduce))
        return Z

    def PCA_dimen_reduct(self, X):
        print('[Info]: Reducing dimensionality to', self.n_components,'dimensions using the PCA approach')
        return self.do_PCA(self.do_Regulation(X))
    
    def PCA_myself(self, X):
        print('[Info]: Reducing dimensionality to', self.n_components,'dimensions using the PCA approach')
        X = (X - X.mean()) / X.std()
        X = np.matrix(X)
        cov = (X.T * X) / X.shape[0]
        U, S, V = np.linalg.svd(cov)
        self.U_reduced = U[:,:self.n_components]
        return U, S, V

    def project_data(self, X):
        return np.asarray(np.dot(X, self.U_reduced))