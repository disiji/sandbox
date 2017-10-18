import numpy as np
from scipy.stats import poisson

lambda_ = 10000
beta = 0.001
step = 5000

res = 0
for i in range(max(0,lambda_ - step), lambda_ + step):
	res += poisson.pmf(i,lambda_) ** beta
print res

res = 0
for i in range(100000):
	res += poisson.pmf(i,lambda_) ** beta
print res