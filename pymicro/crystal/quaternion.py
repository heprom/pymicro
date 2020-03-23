class Quaternion:
    """Class to describe a Quaternion."""

    def __init__(self, array, convention=1):
        assert len(array) == 4
        self.quat = array
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