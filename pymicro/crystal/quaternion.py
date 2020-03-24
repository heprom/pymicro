import numpy as np

class Quaternion:
    """Class to describe a Quaternion."""

    def __init__(self, array, convention=1):
        assert len(array) == 4
        array = np.array(array)
        self.quat = array / np.sqrt(np.sum(array ** 2))
        self.convention = convention  # active by default

    def q0(self):
        return self.quat[0]

    def q1(self):
        return self.quat[1]

    def q2(self):
        return self.quat[2]

    def q3(self):
        return self.quat[3]

    def __repr__(self):
        return str(self.quat)

    def norm(self):
        """Compute the norm of the quaternion (should be 1)."""
        return np.sqrt(np.sum(self.quat ** 2))
