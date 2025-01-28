import numpy as np

epsilon = np.finfo('float').eps
P = -1  # passive convention

def om2eu(g):
    """
    Compute the Euler angles from the orientation matrix (or matrices).

    Parameters
    ----------
    g : array_like
        Either a single orientation matrix of shape (3, 3) or
        an array of shape (n, 3, 3).

    Returns
    -------
    eulers : np.ndarray
        If input is (3, 3), returns an array of shape (3,).
        If input is (n, 3, 3), returns an array of shape (n, 3).

    Notes
    -----
    The logic follows the paper of Rowenhorst et al. (2015). When g[2,2] ≈ 1
    (within machine precision), the angle :math:`\\Phi` is set to 0 (or :math:`\\pi`),
    and :math:`\\phi_2` is set to 0, with all the rotation mapped into :math:`\\phi_1`.
    """
    g = np.asarray(g, dtype=float)

    # Determine if input is single (3,3) or multiple (n,3,3)
    if g.ndim == 2:
        # Single orientation matrix => reshape to (1,3,3) for vectorization
        single_input = True
        g = g[np.newaxis, ...]
    elif g.ndim == 3:
        single_input = False
    else:
        raise ValueError("Input must be shape (3,3) or (n,3,3).")

    n = g.shape[0]  # Number of matrices

    eps = np.finfo(float).eps
    # Extract g[2,2] for each matrix
    g22 = g[:, 2, 2]

    # Arrays for the Euler angles
    phi1 = np.zeros(n, dtype=float)
    Phi  = np.zeros(n, dtype=float)
    phi2 = np.zeros(n, dtype=float)

    # Identify special-case indices where |g22| = 1 within machine precision
    # i.e. top/bottom of the sphere
    special_mask = (np.abs(g22) >= 1.0 - eps)

    # Submasks for g22 > 0 (near +1) vs g22 < 0 (near -1)
    pos_mask = special_mask & (g22 > 0)
    neg_mask = special_mask & (g22 < 0)

    # == Handle the special cases: g22 ≈ +1 ==
    # phi1 = atan2(g[0,1], g[0,0]), Phi = 0, phi2 = 0
    phi1[pos_mask] = np.arctan2(g[pos_mask, 0, 1], g[pos_mask, 0, 0])
    # (No need to set Phi or phi2; they stay 0)

    # == Handle the special cases: g22 ≈ -1 ==
    # phi1 = -atan2(-g[0,1], g[0,0]), Phi = π, phi2 = 0
    phi1[neg_mask] = -np.arctan2(-g[neg_mask, 0, 1], g[neg_mask, 0, 0])
    Phi[neg_mask] = np.pi
    # (phi2 remains 0)

    # == Handle the "regular" case: |g22| < 1 ==
    reg_mask = ~special_mask

    # Phi = arccos(g22)
    Phi[reg_mask] = np.arccos(g22[reg_mask])

    # zeta = 1 / sqrt(1 - g22^2)
    zeta = np.zeros(n, dtype=float)
    zeta[reg_mask] = 1.0 / np.sqrt(1.0 - g22[reg_mask] ** 2)

    # phi1 = atan2(g[2,0]*zeta, -g[2,1]*zeta)
    phi1[reg_mask] = np.arctan2(g[reg_mask, 2, 0] * zeta[reg_mask],
                                -g[reg_mask, 2, 1] * zeta[reg_mask])

    # phi2 = atan2(g[0,2]*zeta, g[1,2]*zeta)
    phi2[reg_mask] = np.arctan2(g[reg_mask, 0, 2] * zeta[reg_mask],
                                g[reg_mask, 1, 2] * zeta[reg_mask])

    # Ensure angles are in range [0, 2*pi)
    phi1 %= 2.0 * np.pi
    Phi  %= 2.0 * np.pi
    phi2 %= 2.0 * np.pi

    # Stack into a single array of shape (n,3)
    eulers = np.column_stack([phi1, Phi, phi2])

    # If single input => return shape (3,)
    if single_input:
        return eulers[0]
    return eulers

def om2eu_OLD(g):
    """
    Legacy om2eu before vectorization.
    Kept to compare with the new implementation.
    """
    eps = np.finfo('float').eps
    (phi1, Phi, phi2) = (0.0, 0.0, 0.0)
    # treat special case where g[2, 2] = 1
    if np.abs(g[2, 2]) >= 1 - eps:
        if g[2, 2] > 0.0:
            phi1 = np.arctan2(g[0][1], g[0][0])
        else:
            phi1 = -np.arctan2(-g[0][1], g[0][0])
            Phi = np.pi
    else:
        Phi = np.arccos(g[2][2])
        zeta = 1.0 / np.sqrt(1.0 - g[2][2] ** 2)
        phi1 = np.arctan2(g[2][0] * zeta, -g[2][1] * zeta)
        phi2 = np.arctan2(g[0][2] * zeta, g[1][2] * zeta)
    # ensure angles are in the range [0, 2*pi]
    if phi1 < 0.0:
        phi1 += 2 * np.pi
    if Phi < 0.0:
        Phi += 2 * np.pi
    if phi2 < 0.0:
        phi2 += 2 * np.pi
    return np.array([phi1, Phi, phi2])


def om2ax(om):
    diag_delta = -P * np.array([om[1, 2] - om[2, 1],
                                om[2, 0] - om[0, 2],
                                om[0, 1] - om[1, 0]])
    # make sure cos(omega) is within [-1, 1]
    t = np.clip(0.5 * (np.trace(om) - 1), -1.0, 1.0)
    omega = np.arccos(t)
    if omega < 2 * epsilon:
        return np.array([0.0, 0.0, 1.0, 0.0])
    # determine the right eigenvector corresponding to the eigenvalue of +1
    w, v = np.linalg.eig(om)
    axis = np.real(v.T[np.isclose(w, 1.0 + 0.0j)])[0]
    # check signs, including when diag delta terms are zeros
    axis = np.where(np.abs(diag_delta) < 1e-12, axis,
                    np.abs(axis) * np.sign(diag_delta))
    return np.array([*axis, omega])


def om2ro(om):
    return eu2ro(om2eu(om))

def om2qu(om):
    """
    Convert a rotation matrix or an array of rotation matrices to quaternions.
    
    Parameters
    ----------
    om : numpy.ndarray
        A single rotation matrix of shape (3,3) or an array of shape (n,3,3).

    Returns
    -------
    q : numpy.ndarray
        The resulting quaternion(s). Shape (4,) for a single input or (n,4) for multiple inputs.
    
    Notes
    -----
    The logic follows the paper of Rowenhorst et al. (2015) (A.7.)
    """

    # Ensure om is a NumPy array
    om = np.asarray(om, dtype=float)
    original_shape_was_single = False
    
    # Reshape input to (n,3,3) if a single matrix
    if om.ndim == 2:
        if om.shape == (3, 3):
            om = om[np.newaxis, ...]  # shape becomes (1,3,3)
            original_shape_was_single = True
        else:
            raise ValueError("Rotation matrix must be (3,3) or (n,3,3).")
    elif om.ndim != 3 or om.shape[1:] != (3, 3):
        raise ValueError("Rotation matrix must be (3,3) or (n,3,3).")

    # Extract the a_ij
    a11 = om[:, 0, 0]
    a22 = om[:, 1, 1]
    a33 = om[:, 2, 2]

    a23 = om[:, 1, 2]
    a32 = om[:, 2, 1]

    a13 = om[:, 0, 2]
    a31 = om[:, 2, 0]

    a12 = om[:, 0, 1]
    a21 = om[:, 1, 0]

    # Compute each quaternion component
    # Use np.clip to avoid negative values due to small floating errors
    eps = 1e-14  # a tiny value to keep sqrt arguments non-negative

    q0 = 0.5 * np.sqrt(np.clip(1.0 + a11 + a22 + a33, 0.0, None))
    q1 = 0.5 * P * np.sqrt(np.clip(1.0 + a11 - a22 - a33, 0.0, None))
    q2 = 0.5 * P * np.sqrt(np.clip(1.0 - a11 + a22 - a33, 0.0, None))
    q3 = 0.5 * P * np.sqrt(np.clip(1.0 - a11 - a22 + a33, 0.0, None))

    # Apply sign modifications
    # q1 = -q1 if a32 < a23
    sign_mask_1 = a32 < a23
    q1 = np.where(sign_mask_1, -q1, q1)

    # q2 = -q2 if a13 < a31
    sign_mask_2 = a13 < a31
    q2 = np.where(sign_mask_2, -q2, q2)

    # q3 = -q3 if a21 < a12
    sign_mask_3 = a21 < a12
    q3 = np.where(sign_mask_3, -q3, q3)

    q = np.stack((q0, q1, q2, q3), axis=-1)

    # Normalize each quaternion
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q /= norms

    # If the original input was a single 3x3 matrix, return a single (4,) quaternion
    if original_shape_was_single:
        return q[0]
    return q


def eu2ro(euler):
    """Compute the rodrigues vector from the 3 euler angles (in radians).

    :param euler: the 3 Euler angles (in radians).
    :return: the rodrigues vector as a 3 components numpy array.
    """
    a = 0.5 * (euler[0] - euler[2])
    b = 0.5 * (euler[0] + euler[2])
    r1 = np.tan(0.5 * euler[1]) * np.cos(a) / np.cos(b)
    r2 = np.tan(0.5 * euler[1]) * np.sin(a) / np.cos(b)
    r3 = np.tan(b)
    return np.array([r1, r2, r3])


def eu2om(euler):
    """
    Convert Euler angles (phi, theta, psi) to rotation matrices.

    Parameters
    ----------
    euler : array-like
        Either a single set of Euler angles of shape (3,) or
        an array of shape (n, 3).

    Returns
    -------
    g : np.ndarray
        If input is (3,), returns shape (3, 3).
        If input is (n, 3), returns shape (n, 3, 3).
    """
    # Convert to a NumPy array to ensure proper indexing
    euler = np.asarray(euler)

    # Distinguish between single-vector input and multiple angles
    if euler.ndim == 1:
        # shape (3,) => reshape to (1,3) for vectorized math
        euler = euler[np.newaxis, :]
        single_input = True
    elif euler.ndim == 2:
        single_input = False
    else:
        raise ValueError("Input must be shape (3,) or (n,3).")

    # Compute sines and cosines in vectorized form
    c1 = np.cos(euler[:, 0])
    s1 = np.sin(euler[:, 0])
    c  = np.cos(euler[:, 1])
    s  = np.sin(euler[:, 1])
    c2 = np.cos(euler[:, 2])
    s2 = np.sin(euler[:, 2])

    # Each of these will have shape (n,)
    g11 = c1 * c2 - s1 * s2 * c
    g12 = s1 * c2 + c1 * s2 * c
    g13 = s2 * s
    g21 = -c1 * s2 - s1 * c2 * c
    g22 = -s1 * s2 + c1 * c2 * c
    g23 = c2 * s
    g31 = s1 * s
    g32 = -c1 * s
    g33 = c

    # Stack them up into an (n, 3, 3) array
    g = np.stack([
        np.stack([g11, g12, g13], axis=-1),
        np.stack([g21, g22, g23], axis=-1),
        np.stack([g31, g32, g33], axis=-1),
    ], axis=1)

    # If the original input was just one set of angles (shape (3,)),
    # return a single 3x3 matrix instead of a (1, 3, 3).
    if single_input:
        g = g[0]

    return g

def eu2qu(euler):
    """
    Compute the quaternion(s) from the Euler angle(s) (in radians).
    
    If input is (3,), returns (4,).
    If input is (n,3), returns (n,4).

    The scalar part of the quaternion is made positive if it is negative.

    Parameters
    ----------
    euler : array-like
        Either a single set of Euler angles [phi1, Phi, phi2] in radians,
        or an (n,3) array of n such sets.

    Returns
    -------
    q : numpy.ndarray
        A single quaternion (4,) or an array of quaternions (n,4).
    """
    # Convert to NumPy array and handle shape
    euler = np.asarray(euler)
    if euler.ndim == 1:
        # reshape to (1,3) for uniform vectorized math
        euler = euler[np.newaxis, :]
        single_input = True
    elif euler.ndim == 2:
        single_input = False
    else:
        raise ValueError("Input must be shape (3,) or (n,3).")

    phi1 = euler[:, 0]
    Phi  = euler[:, 1]
    phi2 = euler[:, 2]

    # Compute half-angles (all are shape (n,))
    half_phi1_plus_phi2 = 0.5 * (phi1 + phi2)
    half_phi1_minus_phi2 = 0.5 * (phi1 - phi2)
    half_Phi = 0.5 * Phi

    # Compute quaternion components
    q0 = np.cos(half_phi1_plus_phi2) * np.cos(half_Phi)
    q1 = np.cos(half_phi1_minus_phi2) * np.sin(half_Phi)
    q2 = np.sin(half_phi1_minus_phi2) * np.sin(half_Phi)
    q3 = np.sin(half_phi1_plus_phi2)  * np.cos(half_Phi)

    # Stack them into (n,4)
    q = np.stack([q0, q1, q2, q3], axis=1)

    # Ensure scalar part (q0) is positive
    negative_scalar_mask = q[:, 0] < 0
    q[negative_scalar_mask] *= -1

    # If the original input was just a single set of Euler angles, return shape (4,)
    if single_input:
        return q[0]
    return q


def eu2qu_OLD(euler):
    """Compute the quaternion from the 3 euler angles (in radians).

    :param tuple euler: the 3 euler angles in radians.
    :return: a unit quaternion representing the rotation.
    """
    (phi1, Phi, phi2) = euler
    q0 = np.cos(0.5 * (phi1 + phi2)) * np.cos(0.5 * Phi)
    q1 = np.cos(0.5 * (phi1 - phi2)) * np.sin(0.5 * Phi)
    q2 = np.sin(0.5 * (phi1 - phi2)) * np.sin(0.5 * Phi)
    q3 = np.sin(0.5 * (phi1 + phi2)) * np.cos(0.5 * Phi)
    q = np.array([q0, -P * q1, -P * q2, -P * q3])
    # the scalar part must be positive
    if q[0] < 0.:
        q *= -1 
    return q

def eu2ax(euler):
    """Compute the (axis, angle) representation associated to this (passive)
    rotation expressed by the Euler angles.

    :param euler: 3 euler angles (in radians).
    :returns: a tuple containing the axis (a vector) and the angle (in radians).
    """
    t = np.tan(0.5 * euler[1])
    s = 0.5 * (euler[0] + euler[2])
    d = 0.5 * (euler[0] - euler[2])
    tau = np.sqrt(t ** 2 + np.sin(s) ** 2)
    alpha = 2 * np.arctan2(tau, np.cos(s))
    if alpha > np.pi:
        axis = np.array([-t / tau * np.cos(d),
                         -t / tau * np.sin(d),
                         -1 / tau * np.sin(s)])
        angle = 2 * np.pi - alpha
    else:
        axis = np.array([t / tau * np.cos(d),
                         t / tau * np.sin(d),
                         1 / tau * np.sin(s)])
        angle = alpha
    return np.array([*axis, angle])


def ax2qu(ax):
    """
    Compute the quaternion associated with the rotation defined by 
    the given (axis, angle) pair.

    :param ax: a 4 component vector composed by the rotation axis
        and the rotation angle (radians).
    :return: the corresponding Quaternion.
    """
    if ax[3] < 2 * epsilon:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([np.cos(0.5 * ax[3]), *(np.sin(0.5 * ax[3]) * ax[:3])])


def ax2ro(ax):
    """
    Compute the Rodrigues vector associated the rotation defined by 
    the given (axis, angle) pair.

    :param ax: a 4 component vector composed by the rotation axis
        and the rotation angle (radians).
    :return: the corresponding Rodrigues vector.
    """
    if abs(ax[3] - np.pi) < epsilon:
        # handle this case
        pass
    return ax[:3] * np.tan(ax[3] / 2)


def ro2ax(rod):
    """
    Compute the axis/angle representation from the Rodrigues vector.

    :param rod: The Rodrigues vector as a 3 components array.
    :returns: A tuple in the (axis, angle) form.
    """
    r = np.linalg.norm(rod)
    if r < epsilon:
        return np.array([0., 0., 1., 0.])
    else:
        axis = rod / r
        angle = 2 * np.arctan(r)
        return np.array([*axis, angle])


def ro2qu(rod):
    return ax2qu(ro2ax(rod))

def ro2eu(rod):
    return qu2eu(ro2qu(rod))

def ro2om(rod):
    return qu2om(ro2qu(rod))

def qu2eu(q):
    q_03 = q[0] ** 2 + q[3] ** 2
    q_12 = q[1] ** 2 + q[2] ** 2
    chi = np.sqrt(q_03 * q_12)
    if chi < epsilon:
        if q_03 < epsilon:
            euler = np.array([np.arctan2(-2 * P * q[0] * q[3], q[0] ** 2 - q[3] ** 2), 0., 0.])
        else:
            euler = np.array([np.arctan2(2 * q[1] * q[2], q[1] ** 2 - q[2] ** 2), np.pi, 0.])
    else:
        euler = np.array([
            np.arctan2((q[1] * q[3] - P * q[0] * q[2]) / chi, (- P * q[0] * q[1] - q[2] * q[3]) / chi),
            np.arctan2(2 * chi, q_03 - q_12),
            np.arctan2((P * q[0] * q[2] + q[1] * q[3]) / chi, (q[2] * q[3] - P * q[0] * q[1]) / chi)
        ])
    return euler

def qu2ax(q):
    # start by computing the rotation angle
    omega = 2 * np.arccos(q[0])
    if omega < epsilon:
        return np.array([0., 0., 1., 0.])
    elif abs(q[0] < epsilon):
        return np.array([q[1], q[2], q[3], np.pi])
    else:
        s = np.sign(q[0]) / np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        return np.array([s * q[1], s * q[2], s * q[3], omega])

def qu2ro(q):
    return ax2ro(qu2ax(q))

def qu2om(q):
    qbar = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    g = np.array([[qbar + 2 * q[1] ** 2,
                   2 * (q[1] * q[2] - P * q[0] * q[3]),
                   2 * (q[1] * q[3] + P * q[0] * q[2])],
                  [2 * (q[1] * q[2] + P * q[0] * q[3]),
                   qbar + 2 * q[2] ** 2,
                   2 * (q[2] * q[3] - P * q[0] * q[1])],
                  [2 * (q[1] * q[3] - P * q[0] * q[2]),
                   2 * (q[2] * q[3] + P * q[0] * q[1]),
                   qbar + 2 * q[3] ** 2]])
    return g
