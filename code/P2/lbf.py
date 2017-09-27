import numpy as np
import pdb
import matplotlib.pyplot as plt

from loadFittingDataP2 import getData
import sys
sys.path.insert(0, '../../code/')

from P1.gradient_descent import stochastic_gradient_descent, finite_difference, quad_bowl_deriv, neg_gaussian_deriv, quad_bowl, neg_gaussian

def transform_vector(x, M):
	return np.array([x**i for i in range(M+1)])

def phi_transform(X, M):
	phi_X = np.zeros((X.shape[0], M+1))
	for i,x in enumerate(X):
		phi_X[i] = transform_vector(x, M)
	return phi_X

def mle_vector(data, Y, M):
	phi = phi_transform(data, M)
	moore_penrose = np.dot(np.linalg.inv(np.dot(phi.T, phi)),phi.T)
	return np.dot(moore_penrose,Y)

def sse(w, params):
	X, y = params['p1'], params['p2']
	return np.square(np.linalg.norm(np.dot(X, w) - y, axis=0))

def sse_deriv(w, params):
	X, y = params['p1'], params['p2']
	return np.dot(X.T, 2*(np.dot(X, w) - y)) 


def p2():
	X, Y = getData(ifPlotData=False)
	Y = np.expand_dims(Y, 1)
	#X = np.array([.1, .2])
	#Y = np.array([1, 2])
	M = int(sys.argv[1])
	plot = False
	theta = mle_vector(X, Y, M)

	phi_X = phi_transform(X, M)

	params = {'p1': phi_X, 'p2': Y}
	#sse_val = sse(theta, params)

	print(finite_difference(sse, theta, params, .0000001))
	print(sse_deriv(theta, params))


	pdb.set_trace()
	#theta = np.random.random(11)
	params = {'p1':phi_X, 'p2': Y}
	sgd_ans = stochastic_gradient_descent(sse, sse_deriv, .00002, .001, params)
	pdb.set_trace()


	if plot == True: 
		plt.plot(X,Y,'o')
		plt.xlabel('x')
		plt.ylabel('y')

		p = np.poly1d(np.flip(theta,0))
		x = np.arange(0,1.0,.01)
		y = p(x)
		plt.plot(x, y)

		plt.show()

