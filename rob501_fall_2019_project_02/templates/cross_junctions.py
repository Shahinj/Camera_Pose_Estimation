import numpy as np
from scipy.linalg import null_space
from scipy.ndimage.filters import *

def cross_junctions(I, bounds, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I       - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bounds  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts    - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of I. These should be floating-point values.
    """
    #--- FILL ME IN ---
    
    m, n = I.shape
    
    ###estimate the border coordinates in  world frame (meters)
    #square size
    sqaure_d = 63.5 / 1000.0
    #border size in x direction, about 1/3 of a square
    border_dx = (1.0/3.0) * sqaure_d #in meters (in world points)
    #border size in y direction, about 1/5 of a square
    border_dy = (1.0/5.0) * sqaure_d #in meters (in world points)
    #how much we need to travel from closest junctions to the borders to get to the border
    dx = border_dx + sqaure_d
    dy = border_dy + sqaure_d
        
    #get the closest junctions to the border
    left_top, right_top, left_down, right_down = Wpts[:,0:1], Wpts[:,7:8], Wpts[:,-8:-7], Wpts[:,-1:]
    #calculate the border coordinates base on the estimates
    left_top, right_top, left_down, right_down = left_top + np.array([[-1*dx], [-1*dy],[0]]), right_top + np.array([[dx], [-1*dy],[0]]), left_down + np.array([[-1*dx], [dy],[0]]), right_down  + np.array([[dx], [dy],[0]])

    ###homography part
    #define the borders of the world frame
    to_convert = np.hstack([left_top,right_top,right_down, left_down])[:2,:]
    #do the homography from the world frame to the image frame
    H,A = dlt_homography(to_convert,bounds)
    
    ###compute the homographied points
    Wpts_h = Wpts.copy()
    Wpts_h[2,:] = 1
    homographied_junctions = (H.dot(Wpts_h) / H.dot(Wpts_h)[2,:])[:2,:]
        
    #loop through each homographied junction, and pass them to saddle point 
    #with a window to get more accurate results
    final = []
    window = 10
    for index, row in enumerate(homographied_junctions.T):
        x,y = row
        x,y = int(x), int(y)
        poi = saddle_point(I[y-window:y+window,x-window:x+window])
        final.append( [x - window + poi[0,0], y - window + poi[1,0]] )
    
    Ipts = np.array(final).T

    #------------------

    return Ipts

    
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
    coeffs = np.linalg.lstsq(a_matrix, b_vector,rcond=None)[0]
    alpha,beta,gamma,delta,eps,zeta  = coeffs.T[0]
    
    #solve for the pt
    solver_matrix = np.array([[2* alpha, beta],[beta, 2 * gamma]])
    solver_vector = np.array([[delta],[eps]])
    
    pt = (-1 * np.linalg.inv(solver_matrix)).dot(solver_vector)
    
    #------------------

    return pt


def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    #initialize the A matrix with zeros
    A = np.zeros([8,9])
    #loop through every point in source and create the A matrix, 2 row by 2 row
    for i,point in enumerate(I1pts.T):
        x = point[0]
        y = point[1]
        #get the corresponce point from the input destinations points
        correspondence = I2pts.T[i]
        u = correspondence[0]
        v = correspondence[1]
        #construct the A matrix use the formula in the paper
        A_i = np.array([ [-1*x, -1*y, -1,   0 ,   0 ,  0, u*x, u*y, u],
                         [0   ,    0,  0, -1*x, -1*y, -1, v*x, v*y, v],
                        ])
        #add these 2 rows to the resultant A matrix (the one we initialize at the beginning)
        A[i*2 : (i+1)*2,:] = A_i
    
    #calculate the H matrix using null space
    H = null_space(A)
    #reshape the H matrix
    H = H.reshape(3,3)
    #get the last entry and normalize
    last_entry = H[2][2]
    #normalize
    H = H * (1.0/last_entry)

    
    #------------------

    return H, A