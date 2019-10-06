import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'p' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array, subpixel location of saddle point in I (x, y coords).
    """
    #--- FILL ME IN ---

    m, n = I.shape
    
    #define the zeros matrix and the vector to fill in later for lstsq
    b_vector = np.zeros(shape = (m * n,1))
    a_matrix = np.zeros(shape = (m*n,6))
    
    #loop through all the points and fill in the vector and matrix
    counter = 0
    for y in range(0,m):
        for x in range(0,n):
            #add the value to the b vector
            b_vector[counter,0] = I[y,x]
            #add the values to the a matrix
            a_matrix[counter,:] = [x*x, x*y, y*y, x, y, 1]
            counter += 1
    
    #solve the lstsq problem and get the coefficients
    coeffs = lstsq(a_matrix, b_vector,rcond=None)[0]
    alpha,beta,gamma,delta,eps,zeta  = coeffs.T[0]
    
    #solve for the pt
    solver_matrix = np.array([[2* alpha, beta],[beta, 2 * gamma]])
    solver_vector = np.array([[delta],[eps]])
    
    pt = (-1 * inv(solver_matrix)).dot(solver_vector)
    
    #------------------

    return pt
