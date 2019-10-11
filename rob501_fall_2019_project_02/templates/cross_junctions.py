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



# def plot(I):
#     import matplotlib.pyplot as plt
#     plt.imshow(I, cmap = 'gray')
#     plt.show()
    
#     
# def null_space_calc(U):
#     # find the eigenvalues and eigenvector of U(transpose).U
#     e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
#     # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(e_vals)] 
    
    
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
        if point[0,0] == 240:
            print('hi')
        top_d   = distance_to_line(point, bounds[:,1],bounds[:,0])[0]

        right_d = distance_to_line(point, bounds[:,2],bounds[:,1])[0]
        
        bot_d   = distance_to_line(point, bounds[:,3],bounds[:,2])[0]
        
        left_d  = distance_to_line(point, bounds[:,0],bounds[:,3])[0]

   #       
        distances[i,:]= [point[0,0], point[1,0] ,top_d,right_d, bot_d, left_d] 

   #   
    
    distances = np.hstack([distances,distances[:,2:].min(axis = 1).reshape(len(distances),1)])
    # distances = np.hstack([distances,distances[:,2:-1].prod(axis = 1).reshape(len(distances),1)])
    
    return distances[distances[:,-1].argsort()][-48:,:2]
    # distances = np.zeros( shape = (len(points),3))
    # centroid = np.array([[np.mean(bpoly[0,:])] , [np.mean(bpoly[1,:])]])
    # for i,point in enumerate(points):
    #     if point[0,0] == 240:
    #         print('hi')
    #     distances[i,:]= [point[0,0], point[1,0] , np.sqrt(np.sum((point - centroid) ** 2)) ] 
    # return distances[distances[:,-1].argsort()][:48,:2]
    
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
    
    
    


# def move_centroids(points, closest, centroids):
#     """returns the new centroids assigned from the points closest to them"""
#     return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
#     
# def closest_centroid(points, centroids):
#     """returns an array containing the index to the nearest centroid for each point"""
#     distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
#     return np.argmin(distances, axis=0)
#     
# def initialize_centroids(points, k):
#     """returns k centroids from the initial points"""
#     centroids = points.copy()
#     np.random.shuffle(centroids)
#     return centroids[:k]
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
   

    
# import os
# os.chdir(r'C:\Users\Shahin\Documents\School\Skule\Year 4\Fall\ROB501\Camera_Pose_Estimation\rob501_fall_2019_project_02\templates')
# 
# if __name__ == "__main__":
# 
#     import numpy as np
#     from imageio import imread
#     from mat4py import loadmat
#     from cross_junctions import cross_junctions
#     
#     # Load the boundary.
#     bpoly = np.array(loadmat("bounds.mat")["bpolyh1"])
#     
#     # Load the world points.
#     Wpts = np.array(loadmat("world_pts.mat")["world_pts"])
#     
#     # Load the example target image.
#     I = imread("example_target.png")
#     
#     # Ipts = cross_junctions(I, bpoly, Wpts)
#     mid_junctions = remove_closest_to_bounds(junctions,bpoly)
#     m, n = I.shape
#     
#     dx,dy = np.gradient(I)
#     
#     Ix = gaussian_filter(dx,2)
#     Iy = gaussian_filter(dy,2)
#     
#     R_score = np.zeros(I.shape)
# 
#     
#     alpha = -1 * 0.0000005
#     
#     A = np.zeros(shape = (m,n,2,2))
#     A[:,:,0,0] = gaussian_filter(Ix ** 2,2)
#     A[:,:,0,1] = gaussian_filter(Ix * Iy,2)
#     A[:,:,1,0] = gaussian_filter(Ix * Iy,2)
#     A[:,:,1,1] = gaussian_filter(Iy ** 2,2)
#         
#     evals, evecs = np.linalg.eig(A)
#     
#     R_score[:,:] = (evals[:,:,0] * evals[:,:,1]) - alpha * ((evals[:,:,0] + evals[:,:,1])** 2) 
#     R_thresh = (R_score > np.percentile(R_score,99)) * R_score
#     
#     # R_thresh = maximum_filter(R_thresh,5)
# 
#     junctions = []
#     grouped = []
#     poligono = [tuple(i) for i in bpoly.T.tolist()]
#     for y in range(0,m):
#         for x in range(0,n):
#             if((x,y) in grouped):
#                 continue
#             pt = np.array([[x],[y]])
#             if(R_thresh[y][x] > 0 and point_in_poly(pt, poligono)):
#                 neighbours = find_neighbours(R_thresh, np.array([[x],[y]]))
#                 grouped= grouped + neighbours
#                 avg = np.array(neighbours).mean(axis = 0)
#                 junctions.append(np.array([[avg[0]],[avg[1]]]))
#                 # junctions.append(np.array([[x],[y]]))
#                 
#     # junctions = np.array(junctions)
#     # junctions = junctions.T.reshape(junctions.shape[1],junctions.shape[0])
#     # for i in range(0,500):
#     #     centroids = initialize_centroids(junctions.T,48)
#     #     closest = closest_centroid(junctions.T, centroids)
#     #     centroids = move_centroids(junctions.T, closest, centroids)
# 
#     #     
#     # 
#     # A = np.array([[Ix ** 2, Ix * Iy],[Ix * Iy, Iy ** 2]])
#     
#     detected = I.copy()
#     mid_junctions = remove_closest_to_bounds(junctions,bpoly)
#     
#     for i,row in enumerate(mid_junctions):
#         x,y = mid_junctions[i]
#         x,y = int(x), int(y)
#         window = 20
#         while(True):
#             # poi = np.array([[1],[1]])
#             poi = saddle_point(I[y-window:y+window,x-window:x+window])
#             sx,sy = poi.T[0]
#             if(sx > 0 and sy > 0):
#                 break
#             else:
#                 window += 5
#         detected[y - window + int(poi[1,0]),x - window + int(poi[0,0])] = 255
#     plot(detected)    
#     # You can plot the points to check!
#     # print(Ipts)                