"""
Example of using FedHEONN method for a multiclass classification task.

In this example, the clients and coordinator are created on the same machine.
In a real environment, the clients and also the coordinator can be created on
different machines. In that case, some communication mechanism must be
established between the clients and the coordinator to send the computations
performed by the clients.
"""

# Author: Oscar Fontenla-Romero <oscar.fontenla@udc.es>
# License: GPL-3.0-only

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from fedHEONN import FedHEONN_classifier, FedHEONN_coordinator

# Number of clients
n_clients = 20

# IID or non-IID scenario (True or False)
iid = True

# The data set is loaded (Dry Bean Dataset)
# Source: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
# Article: https://www.sciencedirect.com/science/article/pii/S0168169919311573?via%3Dihub
Data = pd.read_excel('../data/Dry_Bean_Dataset.xlsx', sheet_name='Dry_Beans_Dataset')                
Data['Class'] = Data['Class'].map({'BARBUNYA': 0, 'BOMBAY': 1, 'CALI': 2, 'DERMASON': 3, 'HOROZ': 4, 'SEKER': 5, 'SIRA': 6})        
Inputs = Data.iloc[:, :-1].to_numpy()
Labels = Data.iloc[:, -1].to_numpy()        
train_X, test_X, train_t, test_t = train_test_split(Inputs, Labels, test_size=0.3, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

train_X = train_X.T
test_X = test_X.T

# Number of training and test data
n = len(train_t)
ntest = len(test_t)  

# Non-IID option: Sort training data by class
if not iid:
    ind = np.argsort(train_t)
    train_t = train_t[ind]
    train_X = train_X[:, ind]
    print('non-IID scenario')
else:        
    ind_list = list(range(n))
    np.random.shuffle(ind_list) # Data are shuffled in case they come ordered by class
    train_X  = train_X[:,ind_list]
    train_t = train_t[ind_list]
    print('IID scenario')
    
# Number of classes
nclasses = len(np.unique(train_t))     

# One hot encoding for the targets
t_onehot = np.zeros((n, nclasses))
for i, value in enumerate(train_t):
    t_onehot[i,value] = 1

# Create a list of clients
clients = []
for i in range(0,n_clients):
    clients.append(FedHEONN_classifier(f='logs'))

# Fit the clients with their local data    
M  = []
US = []
for i, client in enumerate(clients):
    rang = range(int(i*n/n_clients),int(i*n/n_clients)+int(n/n_clients))
    print('Training client:', i+1, 'of', n_clients, '(', min(rang) , '-', max(rang), ') - Classes:', np.unique(train_t[rang]))
    client.fit(train_X[:,rang],t_onehot[rang,:])
    M_c, US_c = client.get_param()
    M.append(M_c)    
    US.append(US_c)        

# Create the coordinator
coordinator = FedHEONN_coordinator(f='logs', lam=0.01)

# The coordinator aggregates the information provided by the clients
# to obtain the weights of the collaborative model
coordinator.aggregate(M, US)

# Send the weights of the aggregate model, obtained by the coordinator,
# to all the clients
for client in clients:
    client.set_weights(coordinator.send_weights())

# Predictions for the test set using one client    
test_y = clients[0].predict(test_X)

print("Test accuracy: %0.2f%%" %(100*accuracy_score(test_t, test_y)))
