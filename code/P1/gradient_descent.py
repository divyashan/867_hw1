from loadParametersP1 import getData
import loadFittingDataP1
import numpy as np
import pdb
import math


def quad_bowl(x, params):
	A, b = params['p1'], params['p2']
	return .5*np.dot(np.dot(x.T,A),x) - np.dot(x.T,b)

def quad_bowl_deriv(x, params):
	A, b = params['p1'], params['p2']
	return np.dot(A,x) - b

def neg_gaussian(x, params):
	mu, cov = params['mu'], params['cov']
	n = x.shape[0]
	cov_det = np.linalg.det(cov)
	const = (-10000/np.sqrt(cov_det*((2*math.pi)**n)))
	cov_inv = np.linalg.inv(cov)
	G = np.dot(np.dot((x-mu).T, cov_inv), (x-mu))
	return const*np.exp(-.5*G)

def neg_gaussian_deriv(x, params):
	mu, cov = params['mu'], params['cov']
	cov_inv = np.linalg.inv(cov)
	return -neg_gaussian(x, mu, cov)*np.dot(cov_inv,(x-mu))

def next_batch(data, y, batch_size):
	curr_ind = 0 
	while True: 
		yield data[curr_ind:curr_ind+batch_size], y[curr_ind:curr_ind+batch_size]
		curr_ind = (curr_ind + batch_size) % data.shape[0]


def finite_difference(f, w, params, h):
	w_p = (w + h*np.eye(len(w)))
	w = np.array([w for i in range(len(w))])
	w = np.squeeze(w, axis=2).T

	gradients = (f(w_p, params) - f(w, params))/h
	return gradients


def least_squares(theta, params):
	X, y = params['p1'], params['p2']
	return np.square(np.linalg.norm(np.dot(X, theta) - y))

def least_squares_deriv(theta, params):
	X, y = params['p1'], params['p2']
	return np.dot(X.T, 2*(np.dot(X, theta) - y)) 


def correct_soln(X, y):
	return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

def batch_gradient_descent(f, df, step_size, threshold, params, guess=None, batch_size=1):
	w = np.zeros((p1.shape[1],))
	if np.any(guess): 
		w = guess

	def stopping_criterion(prev, current, gradient):
		return (np.abs(prev-current) < threshold or np.linalg.norm(gradient) < threshold)
	
	prev = 0
	while True:
		w = w - step_size*df(w, params)
		curr = f(w, params)
		gradient = df(w, params)

		print "W: ", w, "\tObjective f_val: ", curr, "\tDerivative f_val: ", gradient

		if stopping_criterion(prev, curr, gradient):
			break

		prev = curr

	return w

def vary_step_size(t0, t, k):
	return (t0+t)**(-k)

def stochastic_gradient_descent(f, df, step_size, threshold, params, guess=None, batch_size=1):
	p1 = params['p1']
	p2 = params['p2']
	w = np.zeros((p1.shape[1],))
	if np.any(guess): 
		w = guess

	def stopping_criterion(prev, current, gradient):
		return (np.abs(prev-current) < threshold or np.linalg.norm(gradient) < threshold)
	
	prev = 0
	done = False
	t = 0

	while done != True:
		for i in range(len(p1)):
			step_size = vary_step_size(1000000, t, 0.9)
			#pdb.set_trace()
			params = {'p1': p1[i], 'p2': p2[i]}
			w = w - step_size*df(w, params)
			curr = f(w, params)
			gradient = df(w, params)

			#print "W: ", w, "\tObjective f_val: ", curr, "\tDerivative f_val: ", df(w, p1[i], p2[i])
			t += 1

			if stopping_criterion(prev, curr, gradient):
				pdb.set_trace()
				done = True
				break

			prev = curr

	return w

def p1():

	mu, cov, quad_A, quad_b = getData()
	#batch_gradient_descent(neg_gaussian, neg_gaussian_deriv, .01, .000001, mu, cov, guess=np.array([-1, 20]))


	#batch_gradient_descent(quad_bowl, quad_bowl_deriv, .01, .000001, quad_A, quad_b, guess=np.array([-1, 20]))
	#mu = np.array([10, 10])
	#cov = np.array([[10, 0],[0,10]])
	x = np.array([10, 20])

	params = {'p1': quad_A, 'p2': quad_b}
	print finite_difference(quad_bowl, np.array([1, 1]), params, .001)
	print quad_bowl_deriv(np.array([1,1]), params)


	X, y = loadFittingDataP1.getData()

	#test_X = np.array([[1,2],[4,6],[-1,5]])
	#test_y = np.array([3,10,4])


	params = {'p1':X, 'p2':y}
	#batch_gradient_descent(least_squares, least_squares_deriv, .00001, .0001, X, y)
	stochastic_gradient_descent(least_squares, least_squares_deriv, .00002, .001, params)


	print correct_soln(X, y)




