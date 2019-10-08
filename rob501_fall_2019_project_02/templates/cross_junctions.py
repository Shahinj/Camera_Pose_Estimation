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
    
    c = np.array([ [1,1,-1,-1] ,[1,1,-1,-1],  [-1,-1,1,1], [-1,-1,1,1]])
    junctions = ((convolve(I,c) > 100) & (convolve(I,c) < 155))  * convolve(I,c)
    junctions = maximum_filter(junctions,size = 7)
    
    #select vicinities
    grouped = []
    x_junctions = []
    for x in range(np.min(bounds.T[:,0]) ,np.max(bounds.T[:,0]) ):
        for y in range(np.min(bounds.T[:,1]) ,np.max(bounds.T[:,1]) ):
            if((x,y) in grouped):
                continue
            # print(x,y)
            if(junctions[y,x] > 0):
                neighbours = find_neighbours(junctions, np.array([[x],[y]]))
                grouped= grouped + neighbours
                avg = np.array(neighbours).mean(axis = 0)
                x_junctions.append(np.array([[avg[0]],[avg[1]]]))
                
    
    IR_filtered = I.copy()
    
    mid_junctions = remove_closest_to_bounds(x_junctions,bounds)
    
    Ipts = mid_junctions
    
    # for item in mid_junctions:
    #     IR_filtered[int(item[1]), int(item[0])] = 255
    # 
    # plot(IR_filtered)


    #------------------

    return Ipts



# def plot(I):
#     import matplotlib.pyplot as plt
#     plt.imshow(I, cmap = 'gray')
#     plt.show()
    
    
def null_space_calc(U):
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
    # extract the eigenvector (column) associated with the minimum eigenvalue
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

        
        distances[i,:]= [point[0,0], point[1,0] ,top_d,right_d, bot_d, left_d] 

    
    
    distances = np.hstack([distances,distances[:,2:].mean(axis = 1).reshape(len(distances),1)])
    distances = np.hstack([distances,distances[:,2:-1].prod(axis = 1).reshape(len(distances),1)])
    
    return distances[distances[:,-1].argsort()][-48:,:2]
    
    # ind = np.argsort(distances, axis=1)
    # np.take_along_axis(distances, ind, axis=0)[-48:]

   ##   for idx,row in enumerate(distances):
    #         
    #     top_d, bot_d, left_d, right_d = row[2:]
    #     
    #     to_add = False
    #     if(np.where(distances[:,2] < top_d)[0].shape[0] != 0):
    #         if(np.where(distances[:,3] < bot_d)[0].shape[0] != 0):
    #             if(np.where(distances[:,4] < left_d)[0].shape[0] != 0):
    #                 if(np.where(distances[:,5] > right_d)[0].shape[0] != 0):
    #                     to_add = True

   ##       if(to_add):
    #         result = np.vstack([result,row[:2]])
    #         
    # return result
    



    # distances = np.zeros( shape = (len(points),3))
    # for i,point in enumerate(points):
    #     top_d   = np.linalg.norm(np.cross(bpoly[:,1]-bpoly[:,0], bpoly[:,0]-point))/np.linalg.norm(bpoly[:,1]-bpoly[:,0])
    #     bot_d   = np.linalg.norm(np.cross(bpoly[:,3]-bpoly[:,2], bpoly[:,2]-point))/np.linalg.norm(bpoly[:,3]-bpoly[:,2])
    #     left_d  = np.linalg.norm(np.cross(bpoly[:,0]-bpoly[:,3], bpoly[:,3]-point))/np.linalg.norm(bpoly[:,0]-bpoly[:,3])
    #     right_d = np.linalg.norm(np.cross(bpoly[:,2]-bpoly[:,1], bpoly[:,1]-point))/np.linalg.norm(bpoly[:,2]-bpoly[:,1])
    #     
    #     distances[i,:]= [point[0,0], point[1,0] ,np.abs(top_d) + np.abs(bot_d) + np.abs(left_d) + np.abs(right_d)] 
    # 
    # counter = 0
    # while( distances.shape[0] != 48):
    #     dist = np.amin(distances, axis = 0)[2:]
    #     distances = np.delete(distances,np.where(distances[:,2] == dist)[0][0],axis = 0)
    #     counter += 1
    
    # return	 distances[:,:].T 

# if __name__ == "__main__":
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
#     to_convert = np.array([[  0, 255, 255,   0],
#        [  0,   0, 255, 255]])
#        
#     # I_homography,H = apply_homography(I,bpoly,to_convert)
#     # 
# 
# 
# 
#     # find_neighbours(junctions,np.array([[517],[430]]))
#     
#     # dx, dy = np.gradient(I,1) 
#     # dxy = dx * dy
#     # 
#     # gx2 =  gaussian_filter(dx  ** 2, 1)
#     # gy2 =  gaussian_filter(dy  ** 2, 1)
#     # gxy =  gaussian_filter(dxy ** 2, 1)
#     # 
#     # M = np.array([[gx2 , gxy ],[gxy,  gy2 ]])
#     # 
#     # new = np.zeros( (480,640,2,2))
#     # 
#     # new[:,:,0,0] = M[0,0,:,:]
#     # new[:,:,0,1] = M[0,1,:,:]
#     # new[:,:,1,0] = M[1,0,:,:]
#     # new[:,:,1,1] = M[1,1,:,:]
#     # 
#     # M = new
#     # tr = np.trace(M,axis1= 2,axis2= 3)
#     # det =  np.linalg.det(M)
#     # 
#     # r_response =  (det - 1*(tr**2))
#         
#     c = np.array([ [1,1,-1,-1] ,[1,1,-1,-1],  [-1,-1,1,1], [-1,-1,1,1]])
#     junctions = ((convolve(I,c) > 100) & (convolve(I,c) < 155))  * convolve(I,c)
#     junctions = maximum_filter(junctions,size = 7)
#     
#     #select vicinities
#     grouped = []
#     x_junctions = []
#     for x in range(np.min(bpoly.T[:,0]) ,np.max(bpoly.T[:,0]) ):
#         for y in range(np.min(bpoly.T[:,1]) ,np.max(bpoly.T[:,1]) ):
#             if((x,y) in grouped):
#                 continue
#             # print(x,y)
#             if(junctions[y,x] > 0):
#                 neighbours = find_neighbours(junctions, np.array([[x],[y]]))
#                 grouped= grouped + neighbours
#                 avg = np.array(neighbours).mean(axis = 0)
#                 x_junctions.append(np.array([[avg[0]],[avg[1]]]))
#                 
#     
#     IR_filtered = I.copy()
#     
#     mid_junctions = remove_closest_to_bounds(x_junctions,bpoly)
#     for item in mid_junctions:
#         IR_filtered[int(item[1]), int(item[0])] = 255
#     #             
#     # detected_points = remove_closest_to_bounds(x_junctions,to_convert) 
#     # for pt in detected_points.T:
#     #     x,y,distance = pt.astype('uint32')
#     #     IR_filtered[y,x] = 255
#     
#     plot(IR_filtered)
#                          
