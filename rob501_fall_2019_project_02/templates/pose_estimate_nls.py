import numpy as np
from numpy.linalg import inv, norm
from find_jacobian import find_jacobian
from dcm_from_rpy import dcm_from_rpy
from rpy_from_dcm import rpy_from_dcm

def pose_estimate_nls(K, Twcg, Ipts, Wpts):
    """
    Estimate camera pose from 2D-3D correspondences via NLS.

    The function performs a nonlinear least squares optimization procedure 
    to determine the best estimate of the camera pose in the calibration
    target frame, given 2D-3D point correspondences.

    Parameters:
    -----------
    Twcg  - 4x4 homogenous pose matrix, initial guess for camera pose.
    Ipts  - 2xn array of cross-junction points (with subpixel accuracy).
    Wpts  - 3xn array of world points (one-to-one correspondence with Ipts).

    Returns:
    --------
    Twc  - 4x4 np.array, homogenous pose matrix, camera pose in target frame.
    """
    maxIters = 250                          # Set maximum iterations.

    tp = Ipts.shape[1]                      # Num points.
    ps = np.reshape(Ipts, (2*tp, 1), 'F')   # Stacked vector of observations.

    dY = np.zeros((2*tp, 1))                # Residuals.
    J  = np.zeros((2*tp, 6))                # Jacobian.

    #--- FILL ME IN ---
    eps = 0.0000001
    #dx.T = tx, ty, tz, r(phi), p(theta), q(psi)
    X = epose_from_hpose(Twcg) 
    Twc = hpose_from_epose(X)
    for iter in range(0,maxIters):
        J  = np.zeros((2*tp, 6))
        dY = np.zeros((2*tp, 1)) 
        res_sum = 0
        #calculate jacobian for all the points
        for index,row in enumerate(Wpts.T):
            J [index*2:(index+1)*2,:] = find_jacobian(K,Twc,row.reshape(3,1))
            dY[index*2:(index+1)*2,:] = (Ipts[:,index]-Wpts_to_Ipts(K,Twc,Wpts)[:,index]).reshape(2,1)
            res_sum += np.sum(dY[index*2:(index+1)*2,:]) 
        try:
            dx = np.dot(np.dot( inv(np.dot(J.T,J)),J.T),dY)
        except:
            return Twc
        X = X + dx
        Twc = hpose_from_epose(X)
        if( np.abs(res_sum) < eps):
            break
    #------------------
    # Twc = np.eye((4,4))
    
    return Twc

#----- Functions Go Below -----

def Wpts_to_Ipts(K,Twc, Wpts):
    tp = Wpts.shape[1]
    R = Twc[:3,:3]
    T = Twc[:3,3]
    Tcw = np.hstack([R.T, -1*R.T.dot(T).reshape(3,1)])
    Tcw = np.vstack([Tcw, [0,0,0,1]])
    Wpts_h = np.vstack([Wpts, np.ones(shape = (1,tp)) ])
    K_h = np.hstack([K, np.zeros(shape=(3,1)) ])
    K_h = np.vstack([K_h, [0,0,0,1]])
    pts_img = K_h.dot(Tcw.dot(Wpts_h))
    pts_img = pts_img / pts_img[2,:]
    return pts_img[:2,:]

def epose_from_hpose(T):
    """Euler pose vector from homogeneous pose matrix."""
    E = np.zeros((6, 1))
    E[0:3] = np.reshape(T[0:3, 3], (3, 1))
    E[3:6] = rpy_from_dcm(T[0:3, 0:3])
  
    return E

def hpose_from_epose(E):
    """Homogeneous pose matrix from Euler pose vector."""
    T = np.zeros((4, 4))
    T[0:3, 0:3] = dcm_from_rpy(E[3:6])
    T[0:3, 3] = np.reshape(E[0:3], (3,))
    T[3, 3] = 1
  
    return T
