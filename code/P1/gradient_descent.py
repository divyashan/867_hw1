from loadParametersP1 import getData
import loadFittingDataP1
import numpy as np
import pdb
import math


def quad_bowl(x, A, b):
	return .5*np.dot(np.dot(x.T,A),x) - np.dot(x.T,b)

def quad_bowl_deriv(x, A, b):
	return np.dot(A,x) - b

def neg_gaussian(x, mu, cov):
	n = x.shape[0]
	cov_det = np.linalg.det(cov)
	const = (-10000/np.sqrt(cov_det*((2*math.pi)**n)))
	cov_inv = np.linalg.inv(cov)
	G = np.dot(np.dot((x-mu).T, cov_inv), (x-mu))
	return const*np.exp(-.5*G)

def neg_gaussian_deriv(x, mu, cov):
	cov_inv = np.linalg.inv(cov)
	return -neg_gaussian(x, mu, cov)*np.dot(cov_inv,(x-mu))

def next_batch(data, y, batch_size):
	curr_ind = 0 
	while True: 
		yield data[curr_ind:curr_ind+batch_size], y[curr_ind:curr_ind+batch_size]
		curr_ind = (curr_ind + batch_size) % data.shape[0]


def finite_difference(f, mu, cov, x, h):
	x1_p = np.array([x[0]+h,x[1]])
	x2_p = np.array([x[0], x[1]+h])
	grad1 = (f(x1_p, mu, cov) - f(x, mu, cov))/h
	grad2 = (f(x2_p, mu, cov) - f(x, mu, cov))/h
	return np.array([grad1, grad2])


def least_squares(theta,X, y):
	return np.square(np.linalg.norm(np.dot(X, theta) - y))

def least_squares_deriv(theta,X, y):
	return np.dot(2*(np.dot(X, theta) - y),X) 

def correct_soln(X, y):
	return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

def batch_gradient_descent(f, df, step_size, threshold, p1, p2, guess=None, batch_size=1):
	w = np.zeros((p1.shape[1],))
	if np.any(guess): 
		w = guess

	def stopping_criterion(prev, current, gradient):
		return (np.abs(prev-current) < threshold or np.linalg.norm(gradient) < threshold)
	
	prev = 0
	while True:
		w = w - step_size*df(w, p1, p2)
		curr = f(w, p1, p2)
		gradient = df(w, p1, p2)

		print "W: ", w, "\tObjective f_val: ", curr, "\tDerivative f_val: ", df(w, p1, p2)

		if stopping_criterion(prev, curr, gradient):
			break

		prev = curr

	return w

mu, cov, quad_A, quad_b = getData()
#batch_gradient_descent(neg_gaussian, neg_gaussian_deriv, .01, .000001, mu, cov, guess=np.array([-1, 20]))


#batch_gradient_descent(quad_bowl, quad_bowl_deriv, .01, .000001, quad_A, quad_b, guess=np.array([-1, 20]))
#mu = np.array([10, 10])
#cov = np.array([[10, 0],[0,10]])
x = np.array([10, 20])


X, y = loadFittingDataP1.getData()

#test_X = np.array([[1,2],[4,6],[-1,5]])
#test_y = np.array([3,10,4])

batch_gradient_descent(least_squares, least_squares_deriv, .00001, .0001, X, y)

print correct_soln(X, y)




