import numpy as np

def piecewise3poly(X,Y):
	n = len(X)
	A = np.zeros((4*(n-1),4*(n-1)))
	b = np.zeros(4*(n-1))

	# handle i == [1 ... n-1]
	for i in range(1,n-1):
		x1 = X[i]
		x2 = x1*x1
		x3 = x2*x1
		block = np.array((
			(x3,x2,x1,1,0,0,0,0),					 # value of f_i = yi
			(0,0,0,0,x3,x2,x1,1),                    # value of f_i+1 = yi
			(3*x2, 2*x1, 1, 0, -3*x2, -2*x1, -1, 0), # f_i'(x) == f_i+1(x) 
			(6*x1, 2, 0, 0, -6*x1, -2, 0, 0) ))        # f_i''(x) == f_i+1''(x) 

		p = i - 1
		#A[(p*4):(p+1)*4, (p*8):(p+1)*8] = block # insert into main matrix, A
		A[(p*4):(p+1)*4, (p*4):(p+1)*4 + 4] = block # insert into main matrix, A
		b[p*4:(p+1)*4] = [Y[i], Y[i], 0, 0] # insert into right hand side vector b, i.e. (Ax = b)

	# handle i == 0
	block_1 = np.array((
		(X[0]**3,X[0]**2,X[0],1),
		(3*X[0]**2, 2*X[0], 1, 0)))
	A[-4:-2,:4] = block_1
	b[-4:-2] = [Y[0], 0]

	# handle i == n
	block_n = np.array((
		(X[n-1]**3,X[n-1]**2,X[n-1],1),
		(3*X[n-1]**2, 2*X[n-1], 1, 0)))
	A[-2:,-4:] = block_n
	b[-2:] = [Y[n-1], 0]

	# solve matrix, generate piecewise degree-3 polynomial
	sol = np.linalg.solve(A,b)
	polys = get_polys(sol, n-1)
	piecewise_poly = piece_poly(X, polys)

	return piecewise_poly

def get_polys(X,n):
    polys = []
    for i in range(n):
        p = list(reversed(list(X[(i*4):(i+1)*4])))
        polys.append(np.polynomial.Polynomial(p))
    return polys

def piece_poly(X, polys):
    T = X[1]-X[0]
    n = len(X)-2
    def poly(t):
        i = max(0, min(n, int(t // T)))
        return polys[i](t)
    return poly
