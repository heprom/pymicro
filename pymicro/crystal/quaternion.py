import numpy as np


class Quaternion:
    """Class to describe a Quaternion."""

    def __init__(self, array, convention=1):
        assert len(array) == 4
        array = np.array(array)
        self.quat = array / np.sqrt(np.sum(array ** 2))
        self.convention = convention  # active by default

    def __str__(self):
        s = '({:.3f}, <{:.3f}, {:.3f}, {:.3f}>)'.format(
            self.q0, self.q1, self.q2, self.q3)
        return s

    def __repr__(self):
        return 'Quaternion ' + str(self)

    def __add__(self, other):
        """Redefine addition operator for the Quaternion class."""
        return Quaternion([self.quat + other.quat])

    def __sub__(self, other):
        """Redefine substraction operator for the Quaternion class."""
        return Quaternion([self.quat - other.quat])

    @property
    def q0(self):
        return self.quat[0]

    @property
    def q1(self):
        return self.quat[1]

    @property
    def q2(self):
        return self.quat[2]

    @property
    def q3(self):
        return self.quat[3]

    def norm(self):
        """Compute the norm of the quaternion (should be 1)."""
        return np.sqrt(np.sum(self.quat ** 2))
