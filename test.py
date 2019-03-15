import numpy as np

a = np.array([[1,2,3],[4,5,6]])
b = np.array([['q','w','e'],['a','s','d']])

c = np.concatenate([a,b],axis=0)
print(c)
