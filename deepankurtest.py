import numpy as np
from numpy import linalg as LA
from submit import solver

def s_mu(w,u):
	l=len(w);
	for i in range(l):
		if (w[i]>u):
			w[i]=w[i]-u;
		elif (w[i]<-u):
			w[i]=w[i]+u;
		else:
			w[i]=0;
	return w;

data = np.loadtxt("train")

(n,d) = data.shape
#split
s1 = np.random.choice(range(data.shape[0]), 700, replace=False);
s2 = list(set(range(data.shape[0])) - set(s1));

x1=data[:,0:d-1];
x2=data[s2,0:d-1];
y1=data[:,d-1:d];
y2=data[s2,d-1:d];
(j,k)=solver(x1,y1,0.1,1);
#print(j)
print(k)
'''w=np.zeros((1000, 1));
w[2]=4;
w[1]=3;
print(LA.norm(w,2))
r=1;

x1t=np.transpose(x1)

D=w-r*(2*(x1t @ ((x1 @ w)-y1)));
w1=s_mu(D,r)
z=np.transpose(w1);
print(5-np.dot(w1,w))
print(np.dot(z,w))
a = [[1, 0], [0, 1]]

print(z.shape);'''




