import numpy as np
from base import LieAlgebra, LieGroup, EPS

"""
so3: 
- use euler angle as element
- if you want the input be in other format, use SO3 class to do transfomation
"""

"""
SO3:
- to_matrix: return DCM
- to_vec: DCM return euler, others don't need to_vec
- exp & log: inputs should be in so3 or SO3 format, both return matrix lie group
"""

class so3algebra(LieAlgebra): # euler angle body 3-2-1
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.param = param

    def add(self, other):
        return so3algebra(self.param + other.param)

    def rmul(self, scalar):
        return so3algebra(scalar * self.param)
    
    def neg(self):
        return so3algebra(-self.param)

    @property
    def wedge(self):
        theta1 = self.param[0]
        theta2 = self.param[1]
        theta3 = self.param[2]
        return np.array([
            [0, -theta3, theta2],
            [theta3, 0, -theta1],
            [-theta2, theta1, 0]
        ])
    
    @property
    def ad_matrix(self):
        raise NotImplementedError("")

    @classmethod
    def vee(cls, w):
        theta1 = w[2,1]
        theta2 = w[0,2]
        theta3 = w[1,0]
        return np.array([theta1,theta2,theta3])
    

class SO3DCM(LieGroup): # a SO3 direct cosine matrix (3x3)
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (3, 3)
        self.param = param

    @staticmethod
    def identity():
        return SO3DCM(np.eye(3)).param

    @property
    def to_matrix(self):
        return self.param

    @property
    def inv(self):
        return SO3DCM(self.param.T).param

    @property
    def product(self, other: "SO3DCM"):
        raise NotImplementedError("")
    
    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X):
        return SO3Euler.from_dcm(X)
    
    @classmethod
    def log(cls, G: "SO3DCM") -> "so3algebra":
        R = G.param
        theta = np.arccos((np.trace(R) - 1) / 2)
        A = np.where(np.abs(theta) < EPS, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
        return (R - R.T) / (A * 2)
    
    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3DCM": # so3 matrix -> SO3 matrix (DCM)
        v = g.param
        w = g.wedge
        theta = np.linalg.norm(v)
        A = np.where(np.abs(theta) < EPS, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
        B = np.where(np.abs(theta)<EPS, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - np.cos(theta)) / theta ** 2)
        return np.eye(3) + A * w + B * w @ w # return DCM

    # funcions of getting DCM from other format of angles
    @classmethod
    def from_quat(cls, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        R = np.zeros((3,3))
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        aa = a * a
        ab = a * b
        ac = a * c
        ad = a * d
        bb = b * b
        bc = b * c
        bd = b * d
        cc = c * c
        cd = c * d
        dd = d * d
        R[0, 0] = aa + bb - cc - dd
        R[0, 1] = 2 * (bc - ad)
        R[0, 2] = 2 * (bd + ac)
        R[1, 0] = 2 * (bc + ad)
        R[1, 1] = aa + cc - bb - dd
        R[1, 2] = 2 * (cd - ab)
        R[2, 0] = 2 * (bd - ac)
        R[2, 1] = 2 * (cd + ab)
        R[2, 2] = aa + dd - bb - cc
        return R

    @classmethod
    def from_mrp(cls, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        a = r[:3]
        X = so3.wedge(a)
        n_sq = np.dot(a, a)
        X_sq = X @ X
        R = np.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return R.T

    @classmethod
    def from_euler(cls, e):
        return cls.from_quat(SO3Quat.from_euler(e))

class SO3Quat(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (4,1)
        self.param = param

    @staticmethod
    def identity():
        return SO3Quat(np.array([1, 0, 0, 0]))
    
    @property
    def to_matrix(self):
        return SO3DCM.from_quat(self.param)

    @property
    def inv(self):
        return SO3Quat(np.vstack((-self.param[:3], self.param[3])))

    @property
    def product(self, other):
        pass
    
    @property
    def Ad_matrix(self):
        return self.to_matrix

    @classmethod
    def to_vec(cls, X):
        pass
    
    @classmethod
    def log(cls, G: "SO3Quat") -> "so3algebra": # Lie group to Lie algebra
        v = np.zeros((3,))
        q = G.param
        theta = 2 * np.arccos(q[0])
        c = np.sin(theta / 2)
        v[0] = theta * q[1] / c
        v[1] = theta * q[2] / c
        v[2] = theta * q[3] / c
        return np.where(np.abs(c) > EPS, v, np.array([0, 0, 0]))

    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3Quat": # exp: so3 element to quat
        theta = np.norm(g.param)
        w = np.cos(theta / 2)
        c = np.sin(theta / 2)
        v = c * g.param / theta
        return SO3Quat(np.vstack((v, w)))

    # funcions of getting Quat from other format of angles
    @classmethod
    def from_mrp(cls, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        a = r[:3]
        q = np.zeros((4,))
        n_sq = np.dot(a, a)
        den = 1 + n_sq
        q[0] = (1 - n_sq) / den
        for i in range(3):
            q[i + 1] = 2 * a[i] / den
        return np.where(r[3], -q, q)

    @classmethod
    def from_dcm(cls, R):
        assert R.shape == (3, 3)
        b1 = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        b2 = 0.5 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        b3 = 0.5 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        b4 = 0.5 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])

        q1 = np.zeros((4,1))
        q1[0] = b1
        q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
        q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
        q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

        q2 = np.zeros((4,1))
        q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
        q2[1] = b2
        q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
        q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

        q3 = np.zeros((4,1))
        q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
        q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
        q3[2] = b3
        q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

        q4 = np.zeros((4,1))
        q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
        q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
        q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
        q4[3] = b4

        q = np.where(
            np.trace(R) > 0,
            q1,
            np.where(
                np.logical_and(R[0, 0] > R[1, 1], R[0, 0] > R[2, 2]),
                q2,
                np.where(R[1, 1] > R[2, 2], q3, q4),
            ),
        )
        return q

    @classmethod
    def from_euler(self, e):
        assert e.shape == (3, 1) or e.shape == (3,)
        q = np.zeros((4,))
        cosPhi_2 = np.cos(e[0] / 2)
        cosTheta_2 = np.cos(e[1] / 2)
        cosPsi_2 = np.cos(e[2] / 2)
        sinPhi_2 = np.sin(e[0] / 2)
        sinTheta_2 = np.sin(e[1] / 2)
        sinPsi_2 = np.sin(e[2] / 2)
        q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + sinPhi_2 * sinTheta_2 * sinPsi_2
        q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - cosPhi_2 * sinTheta_2 * sinPsi_2
        q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + sinPhi_2 * cosTheta_2 * sinPsi_2
        q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - sinPhi_2 * sinTheta_2 * cosPsi_2
        return q

class SO3Euler(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (3,1) or self.param.shape == (3,)
        self.param = param
    
    @staticmethod
    def identity():
        return np.array([0, 0, 0])
    
    @property
    def to_matrix(self):
        return SO3DCM.from_euler(self.param)

    @property
    def inv(self, cls):
        return cls.from_dcm(SO3DCM.inv(SO3DCM.from_euler(self.param)))

    @property
    def product(self, other: "SO3DCM"):
        raise NotImplementedError("")

    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X):
        pass

    @classmethod
    def log(cls, G: "SO3Euler") -> "so3algebra":
        raise NotImplementedError("")
    
    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3Euler":
        raise NotImplementedError("")
    
    # funcions of getting Euler from other format of angles
    @classmethod
    def from_quat(cls, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        e = np.zeros((3,))
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        e[0] = np.arctan2(2 * (a * b + c * d), 1 - 2 * (b**2 + c**2))
        e[1] = np.arcsin(2 * (a * c - d * b))
        e[2] = np.arctan2(2 * (a * d + b * c), 1 - 2 * (c**2 + d**2))
        return e

    @classmethod
    def from_dcm(cls, R):
        assert R.shape == (3, 3)
        return cls.from_quat(SO3Quat.from_dcm(R))

    @classmethod
    def from_mrp(cls, a):
        assert a.shape == (4, 1) or a.shape == (4,)
        return cls.from_quat(SO3Quat.from_mrp(a))
    
class SO3MRP(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (4, 1) or param.shape == (4,)
        self.param = param

    @staticmethod
    def identity(self):
        return np.array([0, 0, 0, 0])
    
    @property
    def to_matrix(self):
        return DCM.from_mrp(self.param)
    
    @property
    def inv(self, r):
        return np.block([-self.param[:3], self.param[3]])

    @property
    def product(self, other):
        a = self.param[:3]
        b = other.param[:3]
        na_sq = np.dot(a, a)
        nb_sq = np.dot(b, b)
        res = np.zeros((4,1))
        den = 1 + na_sq * nb_sq - 2 * np.dot(b, a)
        res[:3] = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * np.cross(b, a)) / den
        res[3] = 0  # shadow state
        return res
    
    @property
    def Ad_matrix(self):
        return self.to_matrix

    def shadow(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        n_sq = np.dot(r[:3], r[:3])
        res = np.zeros((4, 1))
        res[:3] = -r[:3] / n_sq
        res[3] = np.logical_not(r[3])
        return res

    def shadow_if_necessary(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        return np.where(np.norm(r[:3]) > 1, self.shadow(r), r)
    
    @classmethod
    def to_vec(cls, R):
        return SO3MRP.from_dcm(R)
    
    @classmethod
    def log(cls, G: "SO3MRP") -> "so3algebra":
        r = G.param
        n = np.norm(r[:3])
        return np.where(n > EPS, 4 * np.arctan(n) * r[:3] / n, np.array([0, 0, 0]))

    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3MRP":
        v = g.param
        angle = np.norm(v)
        res = np.zeros((4,1))
        res[:3] = np.tan(angle / 4) * v / angle
        res[3] = 0
        return np.where(angle > EPS, res, np.array([0, 0, 0, 0]))

    @classmethod
    def from_quat(cls, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        x = np.zeros((4,1))
        den = 1 + q[0]
        x[0] = q[1] / den
        x[1] = q[2] / den
        x[2] = q[3] / den
        x[3] = 0
        r = cls.shadow_if_necessary(x)
        r[3] = 0
        return r

    @classmethod
    def from_dcm(cls, R):
        return cls.from_quat(SO3Quat.from_dcm(R))

    @classmethod
    def from_euler(cls, e):
        return cls.from_quat(SO3Quat.from_euler(e))
    
DCM = SO3DCM
Euler = SO3Euler
Quat = SO3Quat
MRP = SO3MRP
so3 = so3algebra