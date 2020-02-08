import numpy as np
from submit import solver

data = np.loadtxt("train")

(n,d) = data.shape

X = data[:,0:d-1]
y = data[:,d-1]

