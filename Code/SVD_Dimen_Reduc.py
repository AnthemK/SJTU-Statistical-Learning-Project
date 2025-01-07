import numpy as np

class SVD_Dimension_Reduction:
    def __init__(self, remain_components=512):
        self.n_components=remain_components
        self.U=None
        self.sigma_matr=None
        self.V=None

    def do_SVD(self, X):
        self.U, self.sigma_matr, self.V = np.linalg.svd(X)
        #print('U=', self.U, 'Sigma=', self.sigma_matr, 'V=', self.V)

    def transform(self, X):
        self.U = self.U[:, :self.n_components]
        self.sigma_matr = np.diag(self.sigma_matr[:self.n_components])
        self.V = self.V[:self.n_components, :]
        #print(self.U.shape[0], self.U.shape[1])
        #print(self.sigma_matr.shape[0], self.sigma_matr.shape[1])
        #print(self.V.shape[0], self.V.shape[1])
        return np.dot(self.U, np.dot(self.sigma_matr, self.V))

    def SVD_dimen_reduct(self, X):
        print('[Info]: Reducing dimensionality to', self.n_components,'dimensions using the SVD approach')
        self.do_SVD(X)
        return self.transform(X)

#??????