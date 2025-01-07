import numpy as np                  # for basic operations over arrays
from scipy.spatial import distance  # to compute the Gaussian kernel
import cvxopt                       # to solve the dual opt. problem
import copy                         # to copy numpy arrays

class SVM:
     linear = lambda x, xࠤ , c=0: x @ xࠤ.T
     polynomial = lambda x, xࠤ , Q=5: (1 + x @ xࠤ.T)**Q
     rbf = lambda x, xࠤ, γ=10: np.exp(-γ*distance.cdist(x, xࠤ,'sqeuclidean'))
     kernel_funs = {'linear': linear, 'polynomial': polynomial, 'rbf': rbf}
     
     def __init__(self, kernel='rbf', C=1, k=2):
         # set the hyperparameters
         self.kernel_str = kernel
         self.kernel = SVM.kernel_funs[kernel]
         self.C = C                  # regularization parameter
         self.k = k                  # kernel parameter
         
         # training data and support vectors (set later)
         self.X, y = None, None
         self.αs = None
         
         # for multi-class classification (set later)
         self.multiclass = False
         self.clfs = []    
 
        SVMClass = lambda func: setattr(SVM, func.__name__, func) or func    
   
    @SVMClass
    def fit(self, X, y, eval_train=False):
        # if more than two unique labels, call the multiclass version
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y, eval_train)
        
        # if labels given in {0,1} change it to {-1,1}
        if set(np.unique(y)) == {0, 1}: y[y == 0] = -1
    
        # ensure y is a Nx1 column vector (needed by CVXOPT)
        self.y = y.reshape(-1, 1).astype(np.double) # Has to be a column vector
        self.X = X
        N = X.shape[0]  # Number of points
        
        # compute the kernel over all possible pairs of (x, x') in the data
        # by Numpy's vectorization this yields the matrix K
        self.K = self.kernel(X, X, self.k)
        
        ### Set up optimization parameters
        # For 1/2 x^T P x + q^T x
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N, 1)))
        
        # For Ax = b
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))
    
        # For Gx <= h
        G = cvxopt.matrix(np.vstack((-np.identity(N),
                                    np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N,1)),
                                    np.ones((N,1)) * self.C)))
    
        # Solve    
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.αs = np.array(sol["x"])            # our solution
            
        # a Boolean array that flags points which are support vectors
        self.is_sv = ((self.αs-1e-3 > 0)&(self.αs <= self.C)).squeeze()
        # an index of some margin support vector
        self.margin_sv = np.argmax((0 < self.αs-1e-3)&(self.αs < self.C-1e-3))
        
        if eval_train:  
        print(f"Finished training with accuracy{self.evaluate(X, y)}")