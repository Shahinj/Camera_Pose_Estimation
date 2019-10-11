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
    # k_inv = np.linalg.inv(K)
    # k_inv_stacked = np.vstack([k_inv, [0,0,0]])
    # C = Twc[0:3,0:3]
    # t = Twc[0:3,3]
    # 
    # Tcw = np.hstack([C.T, (-1 * C.T.dot(t)).reshape(3,1) ])
    # Tcw = np.vstack([Tcw, [0,0,0,1]])
    # 
    # wpt_h = np.vstack([Wpt,[1]])
    # 
    # xij_h = K.dot(Tcw.dot(wpt_h)[0:3,:])
    # xij_h_norm = xij_h / xij_h[2,0]
    # xij = xij_h_norm[0:2]
    # 
    # 
    # wpt_calc = C.T.dot(k_inv.dot(xij_h))
    # wpt_calc = np.vstack([wpt_calc,[1]])
    # #translation
    # wpt_calc = np.vstack([np.hstack([np.eye(3),Tcw[0:3,3].reshape(3,1)]),[0,0,0,1]]).dot(wpt_calc)
    # 
    # #using Szeliski, Page 286 notation
    # y_1 = Wpt - Twc[0:3,3].reshape((3,1))
    # y_2 = Twc[0:3,0:3].dot(y_1)
    # y_3 = y_2 / y_2[2,0]
    # xi = K.dot(y_3)
    # 
    # y_2_z = y_2[2,0]
    # dy2_d1 = (y_2 / y_2_z - np.array([[0],[0],[1]]))
    # dy1_dx = np.array([ [0, 0, 0, K[0,0]], [0, 0, 0, 0], [0, 0, 0, 0]])
    # #])
    # 
    # wpt_h = np.vstack([Wpt,[1]])
    # 
    # 
    # y_1 = Wpt - Twc[0:3,3].reshape((3,1))
    # y_2 = np.gradient(Twc[0:3,0:3])[0].dot(y_1) 
    # y_3 = y_2 / y_2[2,0]
    # xi = K.dot(y_3)
    # 
    # J = np.zeros((2,6))

    #------------------

    return J
