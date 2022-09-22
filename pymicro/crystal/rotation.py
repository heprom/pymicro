import numpy as np

epsilon = np.finfo('float').eps

def ax2qu(ax):
    """
    Compute the quaternion associated the rotation defined by the given
    (axis, angle) pair.

    :param ax: a 4 component vecteur composed by the rotation axis
        and the rotation angle (radians).
    :return: the corresponding Quaternion.
    """
    if ax[3] < 2 * epsilon:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([np.cos(0.5 * ax[3]), *(np.sin(0.5 * ax[3]) * ax[:3])])

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
