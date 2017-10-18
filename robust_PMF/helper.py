import numpy as np 
from scipy.sparse import coo_matrix

datafile = "tw_oc/"
train_data = np.genfromtxt(datafile + 'train.csv', delimiter=',')
val_data = np.genfromtxt(datafile + 'val.csv', delimiter=',')
test_data = np.genfromtxt(datafile + 'test.csv', delimiter=',')

print train_data.max(0), val_data.max(0), test_data.max(0)

num_users = 13559
num_items = 11347

I = np.unique(train_data[:, 0]).shape[0]

# For location that is not tha case. We need to check the maximum location across all 3.
L = np.max(train_data[:, 1])
L = np.max([L, np.max(val_data[:, 1])])
L = np.max([L, np.max(test_data[:, 1])])
L += 1  # It's all 0 based

train = coo_matrix((train_data[:, 2], (train_data[:, 0], train_data[:, 1])), shape=(I, L)).tocsr()
val = coo_matrix((val_data[:, 2], (val_data[:, 0], val_data[:, 1])), shape=(I, L)).tocsr()
test = coo_matrix((test_data[:, 2], (test_data[:, 0], test_data[:, 1])), shape=(I, L)).tocsr()

train_val = train + val
train_val = train_val.tocoo()
train_val = np.column_stack((train_val.row,train_val.col, train_val.data)) 

np.savetxt(datafile+"train_val.csv",train_val,delimiter=",")
