import numpy as np
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
    
    # filtered = gaussian_laplace(gaussian_filter(I,1),2)
    
    # c = np.array([ [1,1,-1,-1] ,[1,1,-1,-1],  [-1,-1,1,1], [-1,-1,1,1]])
    # junctions = ((convolve(I,c) > 100) & (convolve(I,c) < 155))  * convolve(I,c)
    # junctions = maximum_filter(junctions,size = 7)
    # 
    # #select vicinities
    # grouped = []
    # x_junctions = []
    # for x in range(np.min(bounds.T[:,0]) ,np.max(bounds.T[:,0]) ):
    #     for y in range(np.min(bounds.T[:,1]) ,np.max(bounds.T[:,1]) ):
    #         if((x,y) in grouped):
    #             continue
    #         # print(x,y)
    #         if(junctions[y,x] > 0):
    #             neighbours = find_neighbours(junctions, np.array([[x],[y]]))
    #             grouped= grouped + neighbours
    #             avg = np.array(neighbours).mean(axis = 0)
    #             x_junctions.append(np.array([[avg[0]],[avg[1]]]))
    #             
    # 
    # IR_filtered = I.copy()
    # 
    # mid_junctions = remove_closest_to_bounds(x_junctions,bounds)
    # 
    # Ipts = mid_junctions
    
    m, n = I.shape
    
    dx,dy = np.gradient(I)
    
    Ix = gaussian_filter(dx,2)
    Iy = gaussian_filter(dy,2)
    
    R_score = np.zeros(I.shape)

    
    alpha = -1 * 0.0000005
    
    A = np.zeros(shape = (m,n,2,2))
    A[:,:,0,0] = gaussian_filter(Ix ** 2,2)
    A[:,:,0,1] = gaussian_filter(Ix * Iy,2)
    A[:,:,1,0] = gaussian_filter(Ix * Iy,2)
    A[:,:,1,1] = gaussian_filter(Iy ** 2,2)
        
    evals, evecs = np.linalg.eig(A)
    
    R_score[:,:] = (evals[:,:,0] * evals[:,:,1]) - alpha * ((evals[:,:,0] + evals[:,:,1])** 2) 
    R_thresh = (R_score > np.percentile(R_score,99)) * R_score
    
    # R_thresh = maximum_filter(R_thresh,5)

    junctions = []
    grouped = []
    poligono = [tuple(i) for i in bounds.T.tolist()]
    for y in range(0,m):
        for x in range(0,n):
            if((x,y) in grouped):
                continue
            pt = np.array([[x],[y]])
            if(R_thresh[y][x] > 0 and point_in_poly(pt, poligono)):
                neighbours = find_neighbours(R_thresh, np.array([[x],[y]]))
                grouped= grouped + neighbours
                avg = np.array(neighbours).mean(axis = 0)
                junctions.append(np.array([[avg[0]],[avg[1]]]))
                # junctions.append(np.array([[x],[y]]))
                
    # junctions = np.array(junctions)
    # junctions = junctions.T.reshape(junctions.shape[1],junctions.shape[0])
    # for i in range(0,500):
    #     centroids = initialize_centroids(junctions.T,48)
    #     closest = closest_centroid(junctions.T, centroids)
    #     centroids = move_centroids(junctions.T, closest, centroids)

    #     
    # 
    # A = np.array([[Ix ** 2, Ix * Iy],[Ix * Iy, Iy ** 2]])
    
    detected = I.copy()
    mid_junctions = remove_closest_to_bounds(junctions,bounds)
    Ipts = mid_junctions.T
    # for i,row in enumerate(mid_junctions):
    #     x,y = mid_junctions[i]
    #     x,y = int(x), int(y)
    #     window = 20
    #     poi = saddle_point(I[y-window:y+window,x-window:x+window])
    #     detected[y - window + int(poi[1,0]),x - window + int(poi[0,0])] = 255


    #------------------

    return Ipts



def plot(I):
    import matplotlib.pyplot as plt
    plt.imshow(I, cmap = 'gray')
    plt.show()
    

def find_neighbours(I, pt):
    to_check_q = []
    visited = []
    
    to_check_q.append( (pt[0,0],pt[1,0]))
    
    while(len(to_check_q) != 0):
        to_check = to_check_q.pop()
        if(to_check in visited):
            continue
        visited.append(to_check)
        x = to_check[0]
        y = to_check[1]
        
        if( x+1 >= I.shape[1] or x+1 < 0 or y >= I.shape[0] or y < 0):
            right = 0
        else:
            right = I[ np.clip(y,0,I.shape[0]), np.clip(x+1,0,I.shape[1]) ]
        if( x-1 < 0 or x-1 >= I.shape[1] or y >= I.shape[0] or y < 0):
            left = 0
        else:
            left  = I[ np.clip(y,0,I.shape[0]), np.clip(x-1,0,I.shape[1]) ]
        if( y+1 >= I.shape[0] or y+1 < 0 or x >= I.shape[1] or x < 0):
            up = 0
        else:
            up    = I[ np.clip(y+1,0,I.shape[0]), np.clip(x,0,I.shape[1]) ]
        if( y-1 < 0 or y-1 >= I.shape[0] or x >= I.shape[1] or x < 0):
            down = 0  
        else:
            down  = I[ np.clip(y-1,0,I.shape[0]), np.clip(x,0,I.shape[1]) ]
        
        
        if(left > 0):
            if( (x-1,y) not in to_check_q):
                to_check_q.append( (x-1,y))
        if(right > 0):
            if( (x+1,y) not in to_check_q):
                to_check_q.append( (x+1,y))
        if(up > 0):
            if( (x,y+1) not in to_check_q):
                to_check_q.append( (x,y+1))
        if(down > 0):
            if( (x,y-1) not in to_check_q):
                to_check_q.append( (x,y-1))
                
    return visited
    
def distance_to_line(p0, line_p1, line_p2):
        x_diff = line_p2[0] - line_p1[0]
        y_diff = line_p2[1] - line_p1[1]
        num = abs(y_diff*p0[0] - x_diff*p0[1] + line_p2[0]*line_p1[1] - line_p2[1]*line_p1[0])
        den = np.sqrt(y_diff**2 + x_diff**2)
        return num / den
    
def remove_closest_to_bounds(points, bounds):
    distances = np.zeros( shape = (len(points),6))
    result = np.zeros( shape = (0,2))
    for i,point in enumerate(points):
        top_d   = distance_to_line(point, bounds[:,1],bounds[:,0])

        right_d = distance_to_line(point, bounds[:,2],bounds[:,1])
        
        bot_d   = distance_to_line(point, bounds[:,3],bounds[:,2])
        
        left_d  = distance_to_line(point, bounds[:,0],bounds[:,3])

        distances[i,:]= [point[0], point[1] ,top_d,right_d, bot_d, left_d] 

    
    distances = np.hstack([distances,distances[:,2:].min(axis = 1).reshape(len(distances),1)])
    return distances[distances[:,-1].argsort()][-48:,:2]

    
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
    
    
def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')
    
    #Get the x and y value from the point
    x = pt[0,0]
    y = pt[1,0]
    #define the x_vector
    x_vec = np.array([[1],[x],[y],[x*y]])
    #define the lower and upper bounds of x and y
    x_low = np.ceil(x).astype(int)  - 1
    x_up = np.floor(x).astype(int)  + 1
    y_low = np.ceil(y).astype(int)  - 1
    y_up = np.floor(y).astype(int)  + 1
    

    #x is across columns of the image, y is across rows, also clip to get the nearest pixel (np)
    x_low_np = np.clip( x_low ,0, I.shape[1] - 1)
    x_up_np  = np.clip( x_up  ,0, I.shape[1] - 1) 
    y_low_np = np.clip( y_low ,0, I.shape[0] - 1) 
    y_up_np  = np.clip( y_up  ,0, I.shape[0] - 1) 
    
    #get the values for the 4 nearest points from the Image
    left_down  = I[y_low_np,x_low_np]
    right_down = I[y_low_np,x_up_np ]
    left_up    = I[y_up_np ,x_low_np]
    right_up   = I[y_up_np ,x_up_np ]
    
    #get the B coefficients
    b_coefs = np.linalg.inv(np.array([
                        [1,x_low, y_low, x_low * y_low],
                        [1,x_low, y_up , x_low * y_up ],
                        [1,x_up , y_low, x_up  * y_low],
                        [1,x_up , y_up , x_up  * y_up ]
                    ])).T.dot(x_vec)
    #calculate the value of b and round, make sure it is >= 0
    b =  np.max(np.round(b_coefs.T.dot(np.array([[left_down], [left_up], [right_down], [right_up]]))), 0)[0]
    #------------------

    return b


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
    H = null_space_calc(A)
    #reshape the H matrix
    H = H.reshape(3,3)
    #get the last entry and normalize
    last_entry = H[2][2]
    #normalize
    H = H * (1.0/last_entry)

    
    #------------------

    return H, A
    
def null_space_calc(U):
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
    # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(e_vals)] 


def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
    
def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)
    
def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]
#     
# def is_in_polygon(pt,pbounds, I_shape):
#     line = np.mgrid[pt[0,0]: I_shape[1],pt[1,0]:pt[1,0] + 1]
#     line = line.reshape(line.shape[0:2])
#     line = line.T.tolist()
#     return pt.T.tolist()[0] in line
    
    
    
def point_in_poly(pt,poly):
   x,y = pt.T[0]
   # check if point is a vertex
   if (x,y) in poly: return "IN"

   # check if point is on a boundary
   for i in range(len(poly)):
      p1 = None
      p2 = None
      if i==0:
         p1 = poly[0]
         p2 = poly[1]
      else:
         p1 = poly[i-1]
         p2 = poly[i]
      if p1[1] == p2[1] and p1[1] == y and x > min(p1[0], p2[0]) and x < max(p1[0], p2[0]):
         return "IN"
      
   n = len(poly)
   inside = False

   p1x,p1y = poly[0]
   for i in range(n+1):
      p2x,p2y = poly[i % n]
      if y > min(p1y,p2y):
         if y <= max(p1y,p2y):
            if x <= max(p1x,p2x):
               if p1y != p2y:
                  xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
               if p1x == p2x or x <= xints:
                  inside = not inside
      p1x,p1y = p2x,p2y

   if inside: 
      return True
   else:
      return False
   
def apply_homography(I,bounds,to_convert):
    Ihack = np.zeros((255,255))
    H,A = dlt_homography(to_convert,bounds)

    #loop through the bounding box
    for x in     range(np.min(to_convert.T[:,0]) ,np.max(to_convert.T[:,0]) ):
        for y in range(np.min(to_convert.T[:,1]) ,np.max(to_convert.T[:,1]) ):
            #define the homogenous point of the billboard
            homogenous_point = np.array([[x,y,1.0]]).T
            #get the corresponding point in the soldiers tower
            correspondence = H.dot(homogenous_point)
            #normalize the correspondence point by w
            correspondence = correspondence / correspondence[2,0]
            #bilinearly interpolate the soldiers tower
            interpolated = bilinear_interp(I,correspondence[:-1,0:])
            #set the intensity of the yd picture to st
            Ihack[y, x] = interpolated
            
    return Ihack
    
import os
os.chdir(r'C:\Users\Shahin\Documents\School\Skule\Year 4\Fall\ROB501\Camera_Pose_Estimation\rob501_fall_2019_project_02\templates')

if __name__ == "__main__":



    import numpy as np
    from imageio import imread
    from mat4py import loadmat
    from cross_junctions import cross_junctions
    
    # Load the boundary.
    bpoly = np.array(loadmat("bounds.mat")["bpolyh1"])
    
    # Load the world points.
    Wpts = np.array(loadmat("world_pts.mat")["world_pts"])
    
    # Load the example target image.
    I = imread("example_target.png")
    
    # Ipts = cross_junctions(I, bpoly, Wpts)
    # mid_junctions = remove_closest_to_bounds(centroids,bpoly)
    m, n = I.shape
    
    
    start_x,start_y = 12,7
    junctions = []
    for down in range(0,6):
        for right in range(0,8):
            poi = saddle_point(Ihack[start_y + (34*down) : start_y + ((down+2)*34),start_x + (25*right) : start_x + ((right+2)*25)])
            junctions.append( [ start_x + (34*right) + poi[0,0] , start_y + (34*down) + poi[1,0]])
    
    #homography part
    # Compute the perspective homography we need...
    to_convert = np.array([[  0, 255, 255,   0],
           [  0,   0, 255, 255]])
           
           
    Ihack = np.zeros((255,255))
    H,A = dlt_homography(to_convert,bpoly)
    
    tc_list = [(x,y) for x in range(0,255) for y in range(0,255)]
    tc_coords = np.array(tc_list).T
    tc_coords_h = np.vstack([tc_coords,[1] * 255 * 255])
    tc_homographied = H.dot(tc_coords_h)
    tc_homographied_norm = tc_homographied[:2,:] / tc_homographied[2,:]
        
    for i in range(0,tc_homographied_norm.shape[1]):
        x,y = tc_coords[:,i]
        Ihack[y, x] =  bilinear_interp(I,tc_homographied_norm[:,i:i+1])
    
    start_x,start_y = 12,7
    junctions = []
    for down in range(0,6):
        for right in range(0,8):
            poi = saddle_point(Ihack[start_y + (34*down) : start_y + ((down+2)*34),start_x + (25*right) : start_x + ((right+2)*25)])
            junctions.append( [ start_x + (34*right) + poi[0,0] , start_y + (34*down) + poi[1,0]])
    
    for item in junctions:
        Ihack[ int(item[1]), int(item[0]) ] = 255
    
    
    #loop through the bounding box
    for x in     range(0 ,255):
        for y in range(0 ,255):
            #define the homogenous point of the billboard
            homogenous_point = np.array([[x,y,1.0]]).T
            #get the corresponding point in the soldiers tower
            correspondence = H.dot(homogenous_point)
            #normalize the correspondence point by w
            correspondence = correspondence / correspondence[2,0]
            #bilinearly interpolate the soldiers tower
            interpolated = bilinear_interp(I,correspondence[:-1,0:])
            #set the intensity of the yd picture to st
            Ihack[y, x] = interpolated
    #        
    # I_homography,H = apply_homography(I,bpoly,to_convert)
     
    
    
    
    
    
    
    dx,dy = np.gradient(Ihack)
    
    Ix = gaussian_filter(dx,1)
    Iy = gaussian_filter(dy,1)
    
    R_score = np.zeros(Ihack.shape)

    
    #alpha = -1 * 0.000000000001
    alpha = 0.03
    
    A = np.zeros(shape = (Ihack.shape[0],Ihack.shape[1],2,2))
    A[:,:,0,0] = gaussian_filter(Ix ** 2,2)
    A[:,:,0,1] = gaussian_filter(Ix * Iy,2)
    A[:,:,1,0] = gaussian_filter(Ix * Iy,2)
    A[:,:,1,1] = gaussian_filter(Iy ** 2,2)
        
    evals, evecs = np.linalg.eig(A)
    
    R_score[:,:] = (evals[:,:,0] * evals[:,:,1]) - alpha * ((evals[:,:,0] + evals[:,:,1])** 2) 
    #(np.linalg.det(A) - alpha * (np.trace(A.T).T) ** 2)
    #(evals[:,:,0] * evals[:,:,1]) - alpha * ((evals[:,:,0] + evals[:,:,1])** 2) 
    R_thresh = (R_score > np.percentile(R_score,90)) * R_score
    
    R_thresh = maximum_filter(R_thresh,10)

    # junctions = []
    # grouped = []
    # poligono = [tuple(i) for i in bpoly.T.tolist()]
    # for y in range(0,m):
    #     for x in range(0,n):
    #         if((x,y) in grouped):
    #             continue
    #         pt = np.array([[x],[y]])
    #         if(R_thresh[y][x] > 0 and point_in_poly(pt, poligono)):
    #             neighbours = find_neighbours(R_thresh, np.array([[x],[y]]))
    #             grouped= grouped + neighbours
    #             avg = plot(R.array(neighbours).mean(axis = 0)
    #             junctions.append(np.array([[avg[0]],[avg[1]]]))
                
                
    junctions = []
    # poligono = [tuple(i) for i in bpoly.T.tolist()]
    for y in range(0,Ihack.shape[0]):
        for x in range(0,Ihack.shape[1]):
            # pt = np.array([[x],[y]])
            if(R_thresh[y][x] > 0): 
            # and point_in_poly(pt, poligono)):
                junctions.append(np.array([[x],[y]]))
                
    junctions = np.array(junctions)
    junctions = junctions.T.reshape(junctions.shape[1],junctions.shape[0])
    centroids = initialize_centroids(junctions.T,48)
    eps = 1
    while(True):
        closest = closest_centroid(junctions.T, centroids)
        new_centroids = move_centroids(junctions.T, closest, centroids)
        if(np.sqrt(np.sum((new_centroids - centroids) ** 2)) < eps):
            break
        else:
            centroids = new_centroids

    #     
    # 
    # A = np.array([[Ix ** 2, Ix * Iy],[Ix * Iy, Iy ** 2]])
    
    detected = Ihack.copy()
    mid_junctions = remove_closest_to_bounds(centroids,bpoly)
    
    for i,row in enumerate(mid_junctions):
        x,y = mid_junctions[i]
        x,y = int(x), int(y)
        # window = 20
        window = 0
        # while(True):
            # poi = np.array([[1],[1]])
        poi = saddle_point(I[y-window:y+window,x-window:x+window])
            # sx,sy = poi.T[0]
            # if(sx > 0 and sy > 0):
                # break
            # else:
                # window += 5
        detected[y - window + int(poi[1,0]),x - window + int(poi[0,0])] = 255
    plot(detected)    
    # You can plot the points to check!
    # print(Ipts)        
          