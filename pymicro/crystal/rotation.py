import numpy as np

epsilon = np.finfo('float').eps
P = -1  # passive convention


def om2eu(g):
    """
    Compute the Euler angles from the orientation matrix.

    This conversion follows the paper of Rowenhorst et al. :cite:`Rowenhorst2015`.
    In particular when :math:`g_{33} = 1` within the machine precision,
    there is no way to determine the values of :math:`\phi_1` and :math:`\phi_2`
    (only their sum is defined). The convention is to attribute
    the entire angle to :math:`\phi_1` and set :math:`\phi_2` to zero.

    :param g: The 3x3 orientation matrix
    :return: The 3 euler angles in radians.
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
    return ro2qu(om2ro(om))


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
    c1 = np.cos(euler[0])
    s1 = np.sin(euler[0])
    c = np.cos(euler[1])
    s = np.sin(euler[1])
    c2 = np.cos(euler[2])
    s2 = np.sin(euler[2])
    # rotation matrix g
    g11 = c1 * c2 - s1 * s2 * c
    g12 = s1 * c2 + c1 * s2 * c
    g13 = s2 * s
    g21 = -c1 * s2 - s1 * c2 * c
    g22 = -s1 * s2 + c1 * c2 * c
    g23 = c2 * s
    g31 = s1 * s
    g32 = -c1 * s
    g33 = c
    g = np.array([[g11, g12, g13], [g21, g22, g23], [g31, g32, g33]])
    return g


def eu2qu(euler):
    """Compute the quaternion from the 3 euler angles (in radians).

    :param tuple euler: the 3 euler angles in radians.
    :return: a `Quaternion` instance representing the rotation.
    """
    (phi1, Phi, phi2) = euler
    q0 = np.cos(0.5 * (phi1 + phi2)) * np.cos(0.5 * Phi)
    q1 = np.cos(0.5 * (phi1 - phi2)) * np.sin(0.5 * Phi)
    q2 = np.sin(0.5 * (phi1 - phi2)) * np.sin(0.5 * Phi)
    q3 = np.sin(0.5 * (phi1 + phi2)) * np.cos(0.5 * Phi)
    q = Quaternion(np.array([q0, -P * q1, -P * q2, -P * q3]), convention=P)
    if q0 < 0:
        # the scalar part must be positive
        q.quat = q.quat * -1
    # ambiguous rotation
    if q.quat[0] < 3 * epsilon:
        axis = upper_hemishpere_axis(q.quat[1:])
        q.quat = np.array([0., *axis])
    return q.quat


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
    axis = rod / r
    angle = 2 * np.arctan(r)
    return np.array([*axis, angle])


def ro2qu(rod):
    return ax2qu(ro2ax(rod))


def qu2om(q):
    (q0, q1, q2, q3) = q
    qbar = q0 ** 2 - q1 ** 2 - q2 ** 2 - q3 ** 2
    g = np.array([[qbar + 2 * q1 ** 2,
                   2 * (q1 * q2 - P * q0 * q3),
                   2 * (q1 * q3 + P * q0 * q2)],
                  [2 * (q1 * q2 + P * q0 * q3),
                   qbar + 2 * q2 ** 2,
                   2 * (q2 * q3 - P * q0 * q1)],
                  [2 * (q1 * q3 - P * q0 * q2),
                   2 * (q2 * q3 + P * q0 * q1),
                   qbar + 2 * q3 ** 2]])
    return g
