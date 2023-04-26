import numpy as np

"""
so3: 
- use euler angle as element
- exp gives SO3 3x3 matrix (DCM)
- if you want the input or the output of exp be in other format, use SO3 class to do transfomation
"""

EPS = 1e-7

class LieAlgebra:
    def __init__(self, param): # param should be np.array
        self.param = param

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.add(other.neg())

    def __rmul__(self, other):
        return self.rmul(other)

    def __neg__(self):
        return self.neg()

    def __eq__(self, other) -> bool:
        return np.linalg.norm(self.param - other.param) < EPS

class so3(LieAlgebra): # euler angle body 3-2-1
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

    @classmethod
    def vee(cls, w):
        theta1 = w[2,1]
        theta2 = w[0,2]
        theta3 = w[1,0]
        return np.array([theta1,theta2,theta3])

    @property
    def inv(self): # return so3 matrix
        return so3(-self.param).wedge


    def add(self, other):
        return so3(self.param + other.param)

    def rmul(self, scalar):
        return so3(scalar * self.param)
    
    @classmethod
    def exp(cls, w): # so3 matrix -> SO3 matrix (DCM)
        v = cls.vee(w)
        theta = np.linalg.norm(v)
        A = np.where(np.abs(theta) < EPS, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
        B = np.where(np.abs(theta)<EPS, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - np.cos(theta)) / theta ** 2)
        return np.eye(3) + A * w + B * w @ w # return DCM

class LieGroup:
    """
    A Lie Group with group operator (*) is:
    (C)losed under operator (*)
    (A)ssociative with operator (*), (G1*G2)*G3 = G1*(G2*G3)
    (I)nverse: has an inverse such that G*G^-1 = e
    (N)uetral: has a neutral element: G*e = G
    Abstract base class, must implement:
    exp, identity, inv, log, product
    """

    def __init__(self, param):
        self.param = param

    def __mul__(self, other):
        """
        The * operator will be used as the Group multiplication operator
        (see product)
        """
        if not isinstance(other, type(self)):
            return TypeError("Lie Group types must match for product")
        assert isinstance(other, LieGroup)
        return self.product(other)

    def __eq__(self, other) -> bool:
        return np.all(self.param == other.param)

class SO3DCM(LieGroup): # a SO3 direct cosine matrix (3x3)
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (3, 3)

    @property
    def identity(self):
        return SO3DCM(np.eye(3))

    @property
    def inv(self):
        return SO3DCM(self.param.T).param

    @property
    def log(self):
        R = self.param
        theta = np.arccos((np.trace(R) - 1) / 2)
        A = np.where(np.abs(theta) < EPS, 1 - theta**2/6 + theta**4/120, np.sin(theta)/theta)
        return (R - R.T) / (A * 2)
    
    @property
    def product(self, other: "SO3DCM"):
        raise NotImplementedError("")

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
    def __init__(self):
        super().__init__(param)
        assert self.param.shape == (4,1)

    @staticmethod
    def identity(self):
        return SO3Quat(np.array([1, 0, 0, 0]))

    def product(self, other):
        pass
    
    @property
    def inv(self):
        return SO3Quat(np.vstack((-self.param[:3], self.param[3])))

    @property
    def log(self): # Lie group to Lie algebra
        v = np.zeros((3,))
        norm_q = np.norm(self.param)
        theta = 2 * np.arccos(q[0])
        c = np.sin(theta / 2)
        v[0] = theta * q[1] / c
        v[1] = theta * q[2] / c
        v[2] = theta * q[3] / c
        return np.where(np.abs(c) > EPS, v, np.array([0, 0, 0]))

    @property
    def to_matrix(self):
        return SO3DCM.from_quat(self.param)

    # @classmethod
    # def so3_exp(cls, g: so3) -> "LieGroupSO3Quat":
    #     theta = np.norm(g.param)
    #     w = np.cos(theta / 2)
    #     c = np.sin(theta / 2)
    #     v = c * g.param / theta
    #     return SO3Quat(np.vstack((v, w)))

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
    def __init__(self):
        super().__init__(param)
        assert self.param.shape == (3,1) or self.param.shape == (3,)
    
    @staticmethod
    def identity(self):
        return np.array([0, 0, 0])

    @property
    def inv(self, cls):
        return cls.from_dcm(SO3DCM.inv(SO3DCM.from_euler(self.param)))

    @property
    def log(self):
        raise NotImplementedError("")

    def __matmul__(self, other: "LieGroupSO3Dcm"):
        raise NotImplementedError("")
    
    @property
    def to_matrix(self):
        return SO3DCM.from_euler(self.param)

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
        assert param.shape == ()

    def product(self, r1, r2):
        assert r1.shape == (4, 1) or r1.shape == (4,)
        assert r2.shape == (4, 1) or r2.shape == (4,)
        a = r1[:3]
        b = r2[:3]
        na_sq = np.dot(a, a)
        nb_sq = np.dot(b, b)
        res = np.zeros((4,1))
        den = 1 + na_sq * nb_sq - 2 * np.dot(b, a)
        res[:3] = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * np.cross(b, a)) / den
        res[3] = 0  # shadow state
        return res

    def inv(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        return np.block([-r[:3], r[3]])

    def exp(self, v):
        assert v.shape == (3, 1) or v.shape == (3,)
        angle = np.norm(v)
        res = np.zeros((4,1))
        res[:3] = np.tan(angle / 4) * v / angle
        res[3] = 0
        return np.where(angle > EPS, res, np.array([0, 0, 0, 0]))

    def log(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        n = np.norm(r[:3])
        return np.where(n > EPS, 4 * np.arctan(n) * r[:3] / n, np.array([0, 0, 0]))

    def shadow(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        n_sq = np.dot(r[:3], r[:3])
        res = np.zeros((4, 1))
        res[:3] = -r[:3] / n_sq
        res[3] = ca.logic_not(r[3])
        return res

    def shadow_if_necessary(self, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        return np.where(np.norm(r[:3]) > 1, self.shadow(r), r)

    # def kinematics(self, r, w):
    #     assert r.shape == (4, 1) or r.shape == (4,)
    #     assert w.shape == (3, 1) or w.shape == (3,)
    #     a = r[:3]
    #     n_sq = np.dot(a, a)
    #     X = self.wedge(a)
    #     B = 0.25 * ((1 - n_sq) * np.eye(3) + 2 * X + 2 * a @ a.T)
    #     return ca.vertcat(B @ w, 0)

    def from_quat(self, q):
        assert q.shape == (4, 1) or q.shape == (4,)
        x = np.zeros((4,1))
        den = 1 + q[0]
        x[0] = q[1] / den
        x[1] = q[2] / den
        x[2] = q[3] / den
        x[3] = 0
        r = self.shadow_if_necessary(x)
        r[3] = 0
        return r

    def from_dcm(self, R):
        return self.from_quat(SO3Quat.from_dcm(R))

    def from_euler(self, e):
        return self.from_quat(SO3Quat.from_euler(e))

    def identity(self):
        return np.array([0, 0, 0, 0])