"""Base classes for FedHEONN"""

# Author: Oscar Fontenla-Romero <oscar.fontenla@udc.es>
# License: GPL-3.0-only

import tenseal as ts
import numpy as np
import scipy as sp

# Configuring the TenSEAL context for the CKKS encryption scheme
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=32768,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )
context.generate_galois_keys()
context.global_scale = 2**40

##########################################################################
#
# Auxiliary functions for the activation functions of the neurons,
# their derivatives and inverses
#
##########################################################################

# Logsig activation function
def logsig(x):
    return 1 / (1 + np.exp(-x))

def ilogsig(x):
    return -np.log((1/x)-1)

def dlogsig(x):
    return 1/((1+np.exp(-x))**2)*np.exp(-x)

# ReLu activation function 
def relu(x):
    return np.log(1+np.exp(x))

def irelu(x):  # The x must have values ​​> 0 because it is the output range of the ReLu function
    return np.log(np.exp(x)-1)

def drelu(x):
    return 1 / (1 + np.exp(-x)) # It is the logistic function

# Linear activation function 
def linear(x):
    return x

def ilinear(x):
    return x # the inverse of a linear function is the same function

def dlinear(x):
    return np.ones(len(x))

##########################################################################
#
# FedHEONN: Client
#
##########################################################################

class FedHEONN_client:
    """FedHEONN client class.
    
    Parameters
    ----------        
    f : {'logs','relu','lin'}, default='logs'
        Activation function for the neurons.
        
        - 'logs', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)                     
        
        - 'lin', linear function,
          returns f(x) = x        
        
    encrypted: bool, default=True
        Indicates if homomorphic encryption is used in the client or not.
        
    sparse: bool, default=True       
        Indicates whether internal sparse matrices will be used during the training process.
        Recommended for large data sets.
    
    Attributes
    ----------
        encrypted : bool            
            Specifies whether the client is using homomorphic encryption or not.
        sparse : bool
            Specifies whether the client is using internal sparse matrices or not.
        M : list of m vectors of shape (n_outputs,).
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the m vector associated with the ith output neuron.        
        US : list of U*S matrices of shape (n_outputs,).
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the U*S matrices associated with the ith output neuron.    
        W : list of weights of shape (n_outputs,).
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the weights associated with the ith output neuron.
            The weights includes the bias as first element.  
    """    
    
    def __init__(self,  f='logs', encrypted=True, sparse=True):                
        """Constructor method"""          
        
        if (f == 'logs'):    # Logistic activation functions
            self.f      = 'logsig' 
            self.finv   = 'ilogsig'
            self.fderiv = 'dlogsig'
        elif (f == 'relu' ):  # ReLu sctivation functions
            self.f      = 'relu' 
            self.finv   = 'irelu'  
            self.fderiv = 'drelu'
        elif (f == 'lin' ):  # Linear sctivation functions
            self.f      = 'linear' 
            self.finv   = 'ilinear'  
            self.fderiv = 'dlinear'    

        self.encrypted = encrypted  # Encryption hyperparameter
        self.sparse    = sparse     # Sparse hyperparameter   
        self.M         = []
        self.US        = []   
        self.W         = None
                
    def _fit(self, X, d):       
        """Private method to fit the model to data matrix X and target(s) d.
    
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_features, n_samples)
            The input data.
    
        d : ndarray of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).
    
        Returns
        -------
        m  : auxiliar matrix for federated/incremental learning
        US : auxiliar matrix for federated/incremental learning
        """ 
        
        # Number of data points (n)
        n = np.size(X,1);

        # The bias is included as the first input (first row)
        Xp = np.insert(X, 0, np.ones(n), axis=0);        
        
        # Inverse of the neural function
        f_d = eval(self.finv)(d);
            
        # Derivate of the neural function
        derf = eval(self.fderiv)(f_d);
        
        if (self.sparse):                                    
            # Diagonal sparse matrix
            F_sparse = sp.sparse.spdiags(derf, 0, derf.size, derf.size, format = "csr")
            
            # Matrix on which the Singular Value Decomposition will be calculated later    
            H = Xp @ F_sparse                  

            # Singular Value Decomposition of H
            U, S, _ = sp.linalg.svd(H, full_matrices=False)                        
            
            M = Xp @ (F_sparse @ (F_sparse @ f_d.T))
            M = M.flatten()            
            
        else:                              
            # Diagonal matrix
            F = np.diag(derf);    
        
            # Matrix on which the Singular Value Decomposition will be calculated later    
            H = Xp @ F;        
            
            # Singular Value Decomposition of H
            U, S, _ = sp.linalg.svd(H, full_matrices=False)
            
            M = Xp @ (F @ (F @ f_d))   
             
        # If the encrypted option is selected then the M vector is encrypted
        if (self.encrypted):
            M = ts.ckks_vector(context, M)   
                           
        return M, U @ np.diag(S)               
        
    def _predict(self, X):
        """Predict using FedHEONN model.
    
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_features, n_samples)
            The input data.
    
        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """             
        # Number of variables (m) and data points (n)
        m, n = X.shape;

        # Number of output neurons
        n_outputs = len(self.W)
        
        y = np.empty((0,n), float)
        
        # For each output neuron
        for o in range(0, n_outputs):  
            
            # If the weights are encrypted then they are decrypted to get the performance results
            if (self.encrypted):
                W = np.array((self.W[o]).decrypt())                     
            else:
                W = self.W[o]
    
            # Neural Network Simulation
            y = np.vstack((y,eval(self.f)(W.transpose() @ np.insert(X, 0, np.ones(n), axis=0))))    

        return y

    def get_param(self):
        """Method that provides the values ​​of m and U*S"""         
        return self.M, self.US 

    def set_weights(self, W):
        """Method that set the values ​​of the weights.

        Parameters
        ----------
        W : list of ndarray of shape (n_outputs,)
            Each element of the list is an array with the weights associated to the corresponding output neuron.
        """        
        self.W = W


class FedHEONN_classifier(FedHEONN_client):    
    """FedHEONN client for classification tasks"""            
    
    def fit(self, X, t_onehot): 
        """Fit the model to data matrix X and target(s) t_onehot.
    
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_features, n_samples)
            The input data.
    
        t_onehot : ndarray of shape (n_samples, n_classes)
            The target values (class labels using one-hot encoding).            
        """                                    
        n, nclasses = t_onehot.shape
        
        # Transforms the values (0, 1) ​​to (0.05, 0.95) to avoid the problem 
        # of the inverse of the activation function at the extremes
        t_onehot = t_onehot*0.9 + 0.05
        
        # A model is generated for each class
        for c in range(0,nclasses):
            M, US = self._fit(X, t_onehot[:,c])
            (self.M).append(M)
            (self.US).append(US)           
            
    def predict(self, X):
        """Predict using FedHEONN model for classification.
    
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_features, n_samples)
            The input data.
    
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes: values ​​between 0 and n_classes-1
        """                     
        y_onehot = self._predict(X)                                 
        y = np.argmax(y_onehot,axis=0)
        return y


class FedHEONN_regressor(FedHEONN_client):    
    """FedHEONN client for regresion tasks"""

    def fit(self, X, t): 
        """Fit the model to data matrix X and target(s) t.
    
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_features, n_samples)
            The input data.
    
        t : ndarray of shape (n_samples, n_outputs)
            The target values.            
        """                                    
        n, noutputs = t.shape
        
        # A model is generated for each output
        for o in range(0,noutputs):
            M, US = self._fit(X, t[:,o])
            (self.M).append(M)
            (self.US).append(US)
            
    def predict(self, X):
        """Predict using FedHEONN model for regresion.
    
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_features, n_samples)
            The input data.
    
        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """                     
        y = self._predict(X)            
        return y

                       
##########################################################################
#
# FedHEONN: Coordinator
#
##########################################################################
class FedHEONN_coordinator:
    """FedHEONN coordinator class.
    
    Parameters
    ----------
    f : {'logs','relu','lin'}, default='logs'
        Activation function for the neurons of the clients.
        
        - 'logs', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)                     
        
        - 'lin', linear function,
          returns f(x) = x         
        
    lam: regularization term, default=0
        Strength of the L2 regularization term.
        
    encrypted: bool, default=True
        Indicates if homomorphic encryption is used in the clients or not.
        
    sparse: bool, default=True       
        Indicates whether sparse matrices will be used during the aggregation process.
        Recommended for large data sets.
    
    Attributes
    ----------
        lam : float
            Strength of the L2 regularization term.
        encrypted : bool            
            Specifies whether the clients are using homomorphic encryption or not.
        sparse : bool
            Specifies whether the coordinator is using internal sparse matrices or not.
        W : list of weights of shape (n_outputs,).
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the weights associated with the ith output neuron.
            The weights includes the bias as first element.    
    """        
        
    def __init__(self,  f='logs', lam=0, encrypted=True, sparse=True):
                
        if (f == 'logs'):    # Logistic activation functions
            self.f      = 'logsig' 
            self.finv   = 'ilogsig'
            self.fderiv = 'dlogsig'
        elif (f == 'relu' ):  # ReLu sctivation functions
            self.f      = 'relu' 
            self.finv   = 'irelu'  
            self.fderiv = 'drelu'
        elif (f == 'lin' ):  # Linear sctivation functions
            self.f      = 'linear' 
            self.finv   = 'ilinear'  
            self.fderiv = 'dlinear'    

        self.lam       = lam        # Regularization hyperparameter
        self.encrypted = encrypted  # Encryption hyperparameter
        self.sparse    = sparse     # Sparse hyperparameter  
        self.W         = []
          
    def aggregate(self, M_list, US_list):
        """Method to aggregate the models of the clients in the federared learning.
    
        Parameters
        ----------
        M_list : list of shape (n_clients,)
            The list of m terms computed previously by a a set o clients.
    
        US_list : list of shape (n_clients,)
            The list of U*S terms computed previously by a a set o clients.
        """         
       
        # Number of classes
        nclasses = len(M_list[0])
        
        # For each class the results of each client are aggregated    
        for c in range(0,nclasses):  

            # Initialization using the first element of the list
            M  = M_list[0][c]
            US = US_list[0][c]
                
            M_rest  = [item[c] for item in M_list[1:]]
            US_rest = [item[c] for item in US_list[1:]]
            
            # Aggregation of M and US from the second client to the last
            for M_k, US_k in zip(M_rest, US_rest):
                M = M + M_k    
                U, S, _ = sp.linalg.svd(np.concatenate((US_k, US),axis=1), full_matrices=False)
                US = U @ np.diag(S)          
                
            if (self.sparse):                                                                       
                I_ones = np.ones(np.size(S))
                I_sparse = sp.sparse.spdiags(I_ones, 0, I_ones.size, I_ones.size, format = "csr")
                S_sparse = sp.sparse.spdiags(S, 0, S.size, S.size, format = "csr")
                aux2 = S_sparse * S_sparse + self.lam * I_sparse
                                
                # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed          
                if (self.encrypted):        
                    aux2 = aux2.toarray()
                    w = (M.matmul(U)).matmul((U @ np.linalg.pinv(aux2)).T)                     
                else:
                    aux2 = aux2.toarray()
                    w = U @ (np.linalg.pinv(aux2) @ (U.transpose() @ M));                       
                
            else:                                              
                # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed          
                if (self.encrypted):
                    w = (M.matmul(U)).matmul((U @ (np.diag(1/(S*S+self.lam*(np.ones(np.size(S))))))).T)                     
                else:
                    w = U @ (np.diag(1/(S*S+self.lam*(np.ones(np.size(S))))) @ (U.transpose() @ M))          
            
            self.W.append(w)           
    
    def send_weights(self):
        """ Method to get the weights of the aggregated model

        Returns
        -------
        W : list of ndarray of shape (n_outputs,)
            Each element of the list is a CKKSVector (encrypted case) or ndarray (not encrypted)
            containing the weights associated with the ith output neuron.
            The weights includes the bias as first element.             
        """           
        return self.W 
