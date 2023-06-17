import casadi as ca
from .base import LieAlgebra, LieGroup, EPS, wrap

"""
so3: 
- use euler angle as element
- if you want the input be in other format, use SO3 class to do transfomation
"""

"""
SO3:
- to_matrix: return DCM
- to_vec: DCM return euler, others don't need to_vec
- exp & log: inputs should be in so3 or SO3 format, log return vector of algebra element, exp return the format of Lie gorup that you call
(e.g., DCM.exp(A) -> DCM, where A = so3(a))
"""

class so3algebra(LieAlgebra): # euler angle body 3-2-1
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.param = param
        self.param = ca.reshape(wrap(param), (3,1))

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
        x = ca.SX(3,3)
        x[0,1] = -theta3
        x[0,2] = theta2
        x[1,0] = theta3
        x[1,2] = -theta1
        x[2,0] = -theta2
        x[2,1] = theta1
        return x
    
    @property
    def ad_matrix(self):
        raise NotImplementedError("")

    @classmethod
    def vee(cls, w):
        theta1 = w[2,1]
        theta2 = w[0,2]
        theta3 = w[1,0]
        return ca.vertcat(theta1,theta2,theta3)
    

class SO3DCM(LieGroup): # a SO3 direct cosine matrix (3x3)
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3, 3)
        self.param = param

    @staticmethod
    def identity():
        return SO3DCM(ca.DM.eye(3))

    @property
    def to_matrix(self):
        return self.param

    @property
    def inv(self):
        return SO3DCM(self.param.T).param

    def product(self, other: "SO3DCM"):
        return SO3DCM(self.param @ other.param)
    
    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X):
        return SO3Euler.from_dcm(X)
    
    @classmethod
    def log(cls, G: "SO3DCM") -> "so3algebra":
        R = G.param
        theta = ca.acos((ca.trace(R) - 1) / 2)
        print(theta)
        A = ca.if_else(ca.fabs(theta) < EPS, 1 - theta**2/6 + theta**4/120, ca.sin(theta)/theta)
        return so3(so3.vee((R - R.T) / (A * 2)))
    
    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3DCM": # so3 matrix -> SO3 matrix (DCM)
        v = g.param
        w = g.wedge
        theta = ca.norm_2(v)
        A = ca.if_else(ca.fabs(theta) < EPS, 1 - theta**2/6 + theta**4/120, ca.sin(theta)/theta)
        B = ca.if_else(ca.fabs(theta)<EPS, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - ca.cos(theta)) / theta ** 2)
        return SO3DCM(ca.DM.eye(3) + A * w + B * w @ w) # return DCM

    # funcions of getting DCM from other format of angles
    @classmethod
    def from_quat(cls, q):
        q = q.param
        assert q.shape == (4, 1) or q.shape == (4,)
        R = ca.SX(3,3)
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
        return SO3DCM(R)

    @classmethod
    def from_mrp(cls, r):
        r = r.param
        assert r.shape == (4, 1) or r.shape == (4,)
        a = r[:3]
        X = so3(a).wedge
        n_sq = ca.dot(a, a)
        X_sq = X @ X
        R = ca.DM.eye(3) + (8 * X_sq - 4 * (1 - n_sq) * X) / (1 + n_sq) ** 2
        # return transpose, due to convention difference in book
        return SO3DCM(R.T)

    @classmethod
    def from_euler(cls, e):
        return cls.from_quat(SO3Quat.from_euler(e))

class SO3Quat(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (4,1) or param.shape == (4,)
        self.param = param

    @staticmethod
    def identity():
        return SO3Quat(ca.DM([1, 0, 0, 0]))
    
    @property
    def to_matrix(self):
        return SO3DCM.from_quat(self.param)

    @property
    def inv(self):
        return SO3Quat(ca.vertcat((-self.param[:3], self.param[3])))

    def product(self, other):
        a = self.param
        b = other.param
        r1 = a[0]
        v1 = a[1:]
        r2 = b[0]
        v2 = b[1:]
        res = ca.SX(4,1)
        res[0] = r1 * r2 - ca.dot(v1, v2)
        res[1:] = r1 * v2 + r2 * v1 + ca.cross(v1, v2)
        return SO3Quat(res)
    
    @property
    def Ad_matrix(self):
        return self.to_matrix

    @classmethod
    def to_vec(cls, X):
        pass
    
    @classmethod
    def log(cls, G: "SO3Quat") -> "so3algebra": # Lie group to Lie algebra
        v = ca.SX(3,1)
        q = G.param
        theta = 2 * ca.acos(q[0])
        c = ca.sin(theta / 2)
        v[0] = theta * q[1] / c
        v[1] = theta * q[2] / c
        v[2] = theta * q[3] / c
        return so3(ca.if_else(ca.fabs(c) > EPS, v, ca.SX([0, 0, 0])))
    
    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3Quat": # exp: so3 element to quat
        q = ca.SX(4,1)
        v = g.param
        theta = ca.norm_2(v)
        q[0] = ca.cos(theta / 2)
        c = ca.sin(theta / 2)
        n = ca.norm_2(v)
        q[1] = c * v[0] / n
        q[2] = c * v[1] / n
        q[3] = c * v[2] / n
        return cls(ca.if_else(n > 1e-7, q, ca.SX([1, 0, 0, 0])))


    # funcions of getting Quat from other format of angles
    @classmethod
    def from_mrp(cls, r):
        r = r.param
        assert r.shape == (4, 1) or r.shape == (4,)
        a = r[:3]
        q = ca.SX(4,1)
        n_sq = ca.dot(a, a)
        den = 1 + n_sq
        q[0] = (1 - n_sq) / den
        for i in range(3):
            q[i + 1] = 2 * a[i] / den
        return SO3Quat(ca.if_else(r[3], -q, q))

    @classmethod
    def from_dcm(cls, R):
        R = R.param
        assert R.shape == (3, 3)
        b1 = 0.5 * ca.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        b2 = 0.5 * ca.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        b3 = 0.5 * ca.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        b4 = 0.5 * ca.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])

        q1 = ca.SX(4,1)
        q1[0] = b1
        q1[1] = (R[2, 1] - R[1, 2]) / (4 * b1)
        q1[2] = (R[0, 2] - R[2, 0]) / (4 * b1)
        q1[3] = (R[1, 0] - R[0, 1]) / (4 * b1)

        q2 = ca.SX(4,1)
        q2[0] = (R[2, 1] - R[1, 2]) / (4 * b2)
        q2[1] = b2
        q2[2] = (R[0, 1] + R[1, 0]) / (4 * b2)
        q2[3] = (R[0, 2] + R[2, 0]) / (4 * b2)

        q3 = ca.SX(4,1)
        q3[0] = (R[0, 2] - R[2, 0]) / (4 * b3)
        q3[1] = (R[0, 1] + R[1, 0]) / (4 * b3)
        q3[2] = b3
        q3[3] = (R[1, 2] + R[2, 1]) / (4 * b3)

        q4 = ca.SX(4,1)
        q4[0] = (R[1, 0] - R[0, 1]) / (4 * b4)
        q4[1] = (R[0, 2] + R[2, 0]) / (4 * b4)
        q4[2] = (R[1, 2] + R[2, 1]) / (4 * b4)
        q4[3] = b4

        q = ca.if_else(
            ca.trace(R) > 0,
            q1,
            ca.if_else(
                ca.logic_and(R[0, 0] > R[1, 1], R[0, 0] > R[2, 2]),
                q2,
                ca.if_else(R[1, 1] > R[2, 2], q3, q4),
            ),
        )
        return SO3Quat(q)

    @classmethod
    def from_euler(self, e):
        e = e.param
        assert e.shape == (3, 1) or e.shape == (3,)
        q = ca.SX(4,1)
        cosPhi_2 = ca.cos(e[0] / 2)
        cosTheta_2 = ca.cos(e[1] / 2)
        cosPsi_2 = ca.cos(e[2] / 2)
        sinPhi_2 = ca.sin(e[0] / 2)
        sinTheta_2 = ca.sin(e[1] / 2)
        sinPsi_2 = ca.sin(e[2] / 2)
        q[0] = cosPhi_2 * cosTheta_2 * cosPsi_2 + sinPhi_2 * sinTheta_2 * sinPsi_2
        q[1] = sinPhi_2 * cosTheta_2 * cosPsi_2 - cosPhi_2 * sinTheta_2 * sinPsi_2
        q[2] = cosPhi_2 * sinTheta_2 * cosPsi_2 + sinPhi_2 * cosTheta_2 * sinPsi_2
        q[3] = cosPhi_2 * cosTheta_2 * sinPsi_2 - sinPhi_2 * sinTheta_2 * cosPsi_2
        return SO3Quat(q)

class SO3Euler(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.param = param
    
    @staticmethod
    def identity():
        return SO3Euler(ca.DM([0, 0, 0]))
    
    @property
    def to_matrix(self):
        return SO3DCM.from_euler(self.param)

    @property
    def inv(self, cls):
        return cls.from_dcm(SO3DCM.inv(SO3DCM.from_euler(self.param)))


    def product(self, other: "SO3Euler"):
        return SO3Euler.from_dcm(SO3DCM(SO3DCM.from_euler(self) @ SO3DCM.from_euler(other))).param

    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X):
        pass

    @classmethod
    def log(cls, G: "SO3Euler") -> "so3algebra":
        return SO3DCM.log(SO3DCM.from_euler(G))
    
    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3Euler":
        return cls.from_dcm(SO3DCM.exp(g))
    
    # funcions of getting Euler from other format of angles
    @classmethod
    def from_quat(cls, q):
        q = q.param
        assert q.shape == (4, 1) or q.shape == (4,)
        e = ca.SX(3,1)
        a = q[0]
        b = q[1]
        c = q[2]
        d = q[3]
        e[0] = ca.atan2(2 * (a * b + c * d), 1 - 2 * (b**2 + c**2))
        e[1] = ca.asin(2 * (a * c - d * b))
        e[2] = ca.atan2(2 * (a * d + b * c), 1 - 2 * (c**2 + d**2))
        return SO3Euler(e)

    @classmethod
    def from_dcm(cls, R):
        return cls.from_quat(SO3Quat.from_dcm(R))

    @classmethod
    def from_mrp(cls, a):
        return cls.from_quat(SO3Quat.from_mrp(a))
    
class SO3MRP(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (4, 1) or param.shape == (4,)
        self.param = param

    @staticmethod
    def identity():
        return SO3MRP(ca.DM([0, 0, 0, 0]))
    
    @property
    def to_matrix(self):
        return SO3DCM.from_mrp(self.param)
    
    @property
    def inv(self):
        return ca.vertcat(-self.param[:3], self.param[3])

    def product(self, other):
        a = self.param[:3]
        b = other.param[:3]
        na_sq = ca.dot(a, a)
        nb_sq = ca.dot(b, b)
        res = ca.SX(4,1)
        den = 1 + na_sq * nb_sq - 2 * ca.dot(b, a)
        res[:3] = ((1 - na_sq) * b + (1 - nb_sq) * a - 2 * ca.cross(b, a)) / den
        res[3] = 0  # shadow state
        return SO3MRP(res)
    
    @property
    def Ad_matrix(self):
        return self.to_matrix


    def shadow(self):
        r = self
        assert r.shape == (4, 1) or r.shape == (4,)
        n_sq = ca.dot(r[:3], r[:3])
        res = ca.SX(4, 1)
        res[:3] = -r[:3] / n_sq
        res[3] = ca.logic_not(r[3])
        return res

    @classmethod
    def shadow_if_necessary(cls, r):
        assert r.shape == (4, 1) or r.shape == (4,)
        return ca.if_else(ca.norm_2(r[:3]) > 1, cls.shadow(r), r)
    
    @classmethod
    def to_vec(cls, R):
        return cls.from_dcm(R)
    
    @classmethod
    def log(cls, G: "SO3MRP") -> "so3algebra":
        r = G.param
        n = ca.norm_2(r[:3])
        return so3algebra(ca.if_else(n > EPS, 4 * ca.atan(n) * r[:3] / n, ca.SX([0, 0, 0])))

    @classmethod
    def exp(cls, g: "so3algebra") -> "SO3MRP":
        v = g.param
        angle = ca.norm_2(v)
        res = ca.SX(4,1)
        res[:3] = ca.tan(angle / 4) * v / angle
        res[3] = 0
        return cls(ca.if_else(angle > EPS, res, ca.SX([0, 0, 0, 0])))

    @classmethod
    def from_quat(cls, q):
        q = q.param
        assert q.shape == (4, 1) or q.shape == (4,)
        x = ca.SX(4,1)
        den = 1 + q[0]
        x[0] = q[1] / den
        x[1] = q[2] / den
        x[2] = q[3] / den
        x[3] = 0
        r = cls.shadow_if_necessary(x)
        r[3] = 0
        return SO3MRP(r)

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