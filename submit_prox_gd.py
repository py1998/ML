import numpy as np
import random as rnd
import time as tm
from numpy import linalg as LA
import matplotlib.pyplot as plt

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE SUBMIT.PY
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL

# DO NOT CHANGE THE NAME OF THE METHOD "solver" BELOW. THIS ACTS AS THE MAIN METHOD AND
# WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THIS NAME WILL CAUSE EVALUATION FAILURES

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

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

################################
# Non Editable Region Starting #
################################
def solver( X, y, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the model vector and will get returned once timeout happens
	w = np.zeros( (d,) )
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

	# You may reinitialize w to your liking here
	# You may also define new variables here e.g. step_length, mini-batch size etc
	x1=X;y1=y;
	w=np.zeros((1000, 1));
	l=1;#hypertuning parameter
	ll=1;
	value=[];
	timee=[];

################################
# Non Editable Region Starting #
################################
	while True :
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				plt.plot(timee,value);
				plt.show();
				return (w, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
		x1t=np.transpose(x1);
		r=1/l;
		#print(r);
		df=2*(x1t @ ((x1 @ w)-y1))
		#print(df)
		D=w-r*(df);
		#print(D)
		w1=s_mu(D,r);
		#print(w1);
		while ((LA.norm((x1 @ w1)-y1,2)**2)>(LA.norm((x1 @ w)-y1,2)**2)+np.dot(np.transpose(w1-w),df)+(l/2)*(LA.norm(w1-w,2)**2)):
			l=l*3;r=1/l;
			w1=s_mu(D,r);
		
		w=w1+(t/(t+1))*(w1-w)
		value.append(LA.norm(w,1)+(LA.norm((x1 @ w)-y1,2)**2));
		timee.append(t);
		
		
		
		

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses which will be strictly penalized
		
		# Please note that once timeout is reached, the code will simply return w
		# Thus, if you wish to return the average model (as is sometimes done for GD),
		# you need to make sure that w stores the average at all times
		# One way to do so is to define a "running" variable w_run
		# Make all GD updates to w_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w
		# w = (w * (t-1) + w_run)/t
		# This way, w will always store the average and can be returned at any time
		# In this scheme, w plays the role of the "cumulative" variable in the course module optLib
		# w_run on the other hand, plays the role of the "theta" variable in the course module optLib
		
	return (w, totTime) # This return statement will never be reached
