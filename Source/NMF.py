import numpy as np

class nonNegMatrixFac():
    def __init__(self, W, H, max_iters=40000):
        self.W = W
        self.H = H
        self.max_iters = max_iters
        self.loss = []

    def fit(self, V):
        '''
        Iterative non negative matrix factorisation function.
    
        Parameters:
        V: nxn matrix holding the positive data
    
        Returns:
        W: The updated positive W matrix for the formula V Approx= WH
        H: The updated positive H matrix for the formula V Approx= WH
        '''
        self.loss = np.zeros((self.max_iters, 1))
        self.loss[0] = np.linalg.norm(V - self.W @ self.H, ord="fro")

        for iter in range(self.max_iters-1):
            # Update W and H using hadamard product and update rules.
            self.H = self.H * ((self.W.T @ V) / (self.W.T @ self.W @ self.H + 1e-9))
            self.W = self.W * ((V @ self.H.T) / (self.W @ (self.H @ self.H.T) + 1e-9))
            # Measure Frobenius Loss.
            self.loss[iter+1] = np.linalg.norm(V - self.W @ self.H, ord="fro")
        return self.W, self.H
