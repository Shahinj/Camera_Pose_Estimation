import numpy as np

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
    """
    #--- FILL ME IN ---
    R = Twc[0:3,0:3]
    T = Twc[0:3,3]
    tx,ty,tz = T
    wx,wy,wz = Wpt.reshape(3)
    r1,r2,r3,r4,r5,r6,r7,r8,r9 = R.flatten()
    fx,s,cx = K[0,:].flatten()
    fy,cy = K[1,1:].flatten()
    
    #a represents -1*R^(-1)*T
    a1 = -1 * (r1 * tx + r4 * ty + r7 * tz)
    a2 = -1 * (r2 * tx + r5 * ty + r8 * tz)
    a3 = -1 * (r3 * tx + r6 * ty + r9 * tz)
        
        
    q = np.arctan2(r4,r1)                           #psi
    p = np.arctan2(-1*r7,np.sqrt(1-r7 ** 2))        #theta
    r = np.arctan2(r8,r9)                           #phi
    
    denom = r3 * wx + r6 * wy + r9 * wz + a3
    ix_num = wx*(fx*r1 + s*r2 + cx*r3) + wy*(fx*r4 + s*r5 + cx*r6) + wz*(fx*r7 + s*r8 + cx*r9) + fx*a1 + s*a2 + cx*a3
    iy_num = wx*(fy*r2 + cy*r3) + wy*(fy*r5 + cy*r6) + wz*(fy*r8 + cy*r9) + fy*a2 + cy*a3


    #tx, happens in a1,a2,a3
    d_denom = -r3
    #x
    d_num = (fx * -1 * r1) + (s * -1 * r2) + (cx * -1 * r3)
    dx_dtx = (denom * d_num - d_denom * ix_num) / (denom ** 2)
    #y
    d_num = (fy * -1 * r2) + (cy * -1 * r3)
    dy_dtx = (denom * d_num - d_denom * iy_num) / (denom ** 2)
    
    #ty, happens in a1,a2,a3
    d_denom = -r6
    #x
    d_num = (fx * -1 * r4) + (s * -1 * r5) + (cx * -1 * r6)
    dx_dty = (denom * d_num - d_denom * ix_num) / (denom ** 2)
    #y
    d_num = (fy * -1 * r5) + (cy * -1 * r6)
    dy_dty = (denom * d_num - d_denom * iy_num) / (denom ** 2)
    
    
    #tz, happens in a1,a2,a3
    d_denom = -r9
    #x
    d_num = (fx * -1 * r7) + (s * -1 * r8) + (cx * -1 * r9)
    dx_dtz = (denom * d_num - d_denom * ix_num) / (denom ** 2)
    #y
    d_num = (fy * -1 * r8) + (cy * -1 * r9)
    dy_dtz = (denom * d_num - d_denom * iy_num) / (denom ** 2)
    
    cos = np.cos
    sin = np.sin
    phi,theta,psi = r,p,q
    
    
    cq = np.array([[cos(psi), -1*sin(psi), 0],
                   [sin(psi), cos(psi),    0],
                   [0,         0,          1]])
                   
    cp = np.array([[cos(theta)       , 0       ,        sin(theta) ],
                   [0                , 1       ,        0          ],
                   [-1*sin(theta)    , 0       ,       cos(theta)  ]])
                   
    cr = np.array([[1       , 0       , 0             ],
                   [0       , cos(phi),    -1*sin(phi)],
                   [0       , sin(phi),       cos(phi)]])
    
    
    
    #q (psi)
    dcq = np.array([[-1*sin(psi), -1*cos(psi),    0],
                    [cos(psi)   , -1*sin(psi),    0],
                    [0          ,           0,    0]])
                    
    dr = dcq.dot(cp.dot(cr))
    dr1,dr2,dr3,dr4,dr5,dr6,dr7,dr8,dr9 = dr.flatten()
    da1 = -1*(tx*dr1 +ty*dr4 +tz*dr7) 
    da2 = -1*(tx*dr2 +ty*dr5 +tz*dr8) 
    da3 = -1*(tx*dr3 +ty*dr6 +tz*dr9) 
    d_denom = dr3 * wx + dr6 * wy + dr9 * wz + da3
    
    #x
    d_num = wx*(fx  * dr1 + s * dr2 + cx * dr3) + wy*(fx*dr4 + s*dr5 + cx*dr6) + wz*(fx*dr7 + s*dr8 + cx*dr9) + fx * da1 + s * da2 + cx * da3
    dx_dtq = (denom * d_num - d_denom * ix_num) / (denom ** 2)
    #y
    d_num = wx*(fy * dr2 + cy * dr3) + wy*(fy*dr5 + cy*dr6) + wz*(fy*dr8 + cy*dr9) + fy * da2 + cy * da3
    dy_dtq = (denom * d_num - d_denom * iy_num) / (denom ** 2)
    
    #p (theta)
    dcp = np.array([[-1*sin(theta)       , 0       ,        cos(theta) ],
                   [0                    , 0       ,        0          ],
                   [-1*cos(theta)        , 0       ,    -1*sin(theta)  ]])
                    
    dr = cq.dot(dcp.dot(cr))
    dr1,dr2,dr3,dr4,dr5,dr6,dr7,dr8,dr9 = dr.flatten()
    da1 = -1*(tx*dr1 +ty*dr4 +tz*dr7) 
    da2 = -1*(tx*dr2 +ty*dr5 +tz*dr8) 
    da3 = -1*(tx*dr3 +ty*dr6 +tz*dr9) 
    d_denom = dr3 * wx + dr6 * wy + dr9 * wz + da3
    
    #x
    d_num = wx*(fx  * dr1 + s * dr2 + cx * dr3) + wy*(fx*dr4 + s*dr5 + cx*dr6) + wz*(fx*dr7 + s*dr8 + cx*dr9) + fx * da1 + s * da2 + cx * da3
    dx_dtp = (denom * d_num - d_denom * ix_num) / (denom ** 2)
    #y
    d_num = wx*(fy * dr2 + cy * dr3) + wy*(fy*dr5 + cy*dr6) + wz*(fy*dr8 + cy*dr9) + fy * da2 + cy * da3
    dy_dtp = (denom * d_num - d_denom * iy_num) / (denom ** 2)
    
    #r (phi)
    dcr = np.array([[0       , 0        , 0             ],
                   [0       ,-1*sin(phi),    -1*cos(phi)],
                   [0       ,   cos(phi),    -1*sin(phi)]])
                    
    dr = cq.dot(cp.dot(dcr))
    dr1,dr2,dr3,dr4,dr5,dr6,dr7,dr8,dr9 = dr.flatten()
    da1 = -1*(tx*dr1 +ty*dr4 +tz*dr7) 
    da2 = -1*(tx*dr2 +ty*dr5 +tz*dr8) 
    da3 = -1*(tx*dr3 +ty*dr6 +tz*dr9) 
    d_denom = dr3 * wx + dr6 * wy + dr9 * wz + da3
    
    #x
    d_num = wx*(fx  * dr1 + s * dr2 + cx * dr3) + wy*(fx*dr4 + s*dr5 + cx*dr6) + wz*(fx*dr7 + s*dr8 + cx*dr9) + fx * da1 + s * da2 + cx * da3
    dx_dtr = (denom * d_num - d_denom * ix_num) / (denom ** 2)
    #y
    d_num = wx*(fy * dr2 + cy * dr3) + wy*(fy*dr5 + cy*dr6) + wz*(fy*dr8 + cy*dr9) + fy * da2 + cy * da3
    dy_dtr = (denom * d_num - d_denom * iy_num) / (denom ** 2)
    

    
    
    J = np.array([ [dx_dtx, dx_dty, dx_dtz, dx_dtr, dx_dtp, dx_dtq],
                   [dy_dtx, dy_dty, dy_dtz, dy_dtr, dy_dtp, dy_dtq]])

    #------------------

    return J
