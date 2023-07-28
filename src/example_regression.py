"""
Example of using FedHEONN method for a regression task.

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
from sklearn.metrics import mean_squared_error
from fedHEONN import FedHEONN_regressor, FedHEONN_coordinator

# Number of clients
n_clients = 20

# The data set is loaded (Carbon Nanotubes)
# Source: https://archive.ics.uci.edu/dataset/448/carbon+nanotubes
Data = pd.read_csv('../data/carbon_nanotubes.csv', delimiter=';')                
Inputs = Data.iloc[:, :-3].to_numpy()
Targets = Data.iloc[:, -3:].to_numpy() # 3 outputs to predict   
train_X, test_X, train_t, test_t = train_test_split(Inputs, Targets, test_size=0.3, random_state=42)

# Data normalization (z-score): mean 0 and std 1
scaler = StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

train_X = train_X.T
test_X = test_X.T

# Number of training and test data
n = len(train_t)
ntest = len(test_t)  
    
# Number of outputs
_ , noutputs = train_t.shape

# Create a list of clients
clients = []
for i in range(0,n_clients):
    clients.append(FedHEONN_regressor(f='lin'))

# Fit the clients with their local data    
M  = []
US = []
for i, client in enumerate(clients):
    rang = range(int(i*n/n_clients),int(i*n/n_clients)+int(n/n_clients))
    print('Training client:', i+1, 'of', n_clients, '(', min(rang) , '-', max(rang), ')')
    client.fit(train_X[:,rang],train_t[rang,:])
    M_c, US_c = client.get_param()
    M.append(M_c)    
    US.append(US_c)        

# Create the coordinator
coordinator = FedHEONN_coordinator(f='lin', lam=0.01)

# The coordinator aggregates the information provided by the clients
# to obtain the weights of the collaborative model
coordinator.aggregate(M, US)

# Send the weights of the aggregate model, obtained by the coordinator,
# to all the clients
for client in clients:
    client.set_weights(coordinator.send_weights())

# Predictions for the test set using one client    
test_y = clients[0].predict(test_X)

# Global MSE for the 3 outputs
print("Test MSE: %0.8f" %(100*mean_squared_error(test_t, test_y.T)))
