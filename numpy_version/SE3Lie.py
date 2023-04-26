import numpy as np
import SO3Lie
from SO3Lie import EPS

DCM = SO3Lie.SO3DCM
Euler = SO3Lie.SO3Euler
so3 = SO3Lie.so3

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

    # def __eq__(self, other) -> bool:
    #     return ca.norm_2(self.param - other.param) < EPS

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


class se3(LieAlgebra): # param: [x,y,z,theta1,theta2,theta3]
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (6,1) or param.shape == (6,)

    @property
    def wedge(self):
        v = self.param[0:3].reshape(3,1)
        theta = self.param[3:6]
        tw = so3(theta).wedge
        return np.block([
            [tw, v],
            [0, 0, 0, 0]
        ])

    @property
    def ad_matrix(self):
        """
        takes 6x1 lie algebra
        input vee operator [x,y,z,theta1,theta2,theta3]
        """
        v = self.param
        ad_se3 = np.zeros((6,6))
        ad_se3[0, 1] = -v[5]
        ad_se3[0, 2] = v[3]
        ad_se3[0, 4] = -v[2]
        ad_se3[0, 5] = v[1]
        ad_se3[1, 0] = v[5]
        ad_se3[1, 2] = -v[3]
        ad_se3[1, 3] = v[2]
        ad_se3[1, 5] = -v[0]
        ad_se3[2, 0] = -v[4]
        ad_se3[2, 1] = v[3]
        ad_se3[2, 3] = -v[1]
        ad_se3[2, 4] = v[0]
        ad_se3[3, 4] = -v[5]
        ad_se3[3, 5] = v[4]
        ad_se3[4, 3] = v[5]
        ad_se3[4, 5] = -v[3]
        ad_se3[5, 3] = -v[4]
        ad_se3[5, 4] = v[3]
        return ad_se3
 
    @classmethod
    def vee(cls, w): # w is 4x4 Lie algebra matrix
        assert w.shape == (4,4)
        x = w[0,3]
        y = w[1,3]
        z = w[2,3]
        theta1 = w[2,1]
        theta2 = w[0,2]
        theta3 = w[1,0]
        return np.array([x,y,z,theta1,theta2,theta3])

    @classmethod
    def exp(cls, vw): # Lie algebra to Lie group# vw is v in wedge form (se3 lie algebra)
        v = cls.vee(vw) # v = [x,y,z,theta1,theta2,theta3]
        v_so3 = v[3:6]  # grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        X_so3 = so3(v_so3).wedge  # wedge operator for so3
        theta = np.linalg.norm(v[3:6])  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)

        # translational components u
        u = np.array([v[0],v[1],v[2]])

        R = DCM.so3_exp(so3(v_so3))  #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational
        C1 = np.where(np.abs(theta)<EPS, 1 - theta ** 2 / 6 + theta ** 4 / 120, np.sin(theta)/theta)
        C2 = np.where(np.abs(theta)<EPS, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - np.cos(theta)) / theta ** 2)
        C = np.where(np.abs(theta)<EPS, 1/6 - theta ** 2 /120 + theta ** 4 / 5040, (1 - C1) / theta ** 2)

        V = np.eye(3) + C2 * X_so3 + C * X_so3 @ X_so3

        return np.block([[R, (V@u).reshape(3,1)],[0,0,0,1]])

class SE3(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (6,1) or param.shape == (6,)
        self.R = DCM.from_euler(param[3:6])
        self.p = param[0:3]
        self.M = np.block([
            [self.R, self.p.reshape(3,1)],
            [np.zeros((1,3)), np.array(1)]
        ])

    @property
    def Ad_matrix(self): # Ad matrix of v(6x1) for SE3 Lie Group
        px = so3(self.param[3:6]).wedge # skew-symmetric (equivalent to wedge in so3)
        return np.block([[self.R, px@self.R],
                         [np.zeros((3,3)),self.R]])

    @property
    def inv(self):  # input a matrix of SX form from casadi
        return np.block([[self.R.T, -self.R.T@self.p.reshape(3,1)],
                         [0,0,0,1]])

    @property
    def log(self): # SE3 matrix to se3 matrix
        R = self.R # get the SO3 Lie group matrix
        theta = np.arccos((np.trace(R) - 1) / 2)
        wSkew = so3(DCM(R).log).wedge
        C1 = np.where(np.abs(theta)<EPS, 1 - theta ** 2 / 6 + theta ** 4 / 120, np.sin(theta)/theta)
        C2 = np.where(np.abs(theta)<EPS, 0.5 - theta ** 2 / 24 + theta ** 4 / 720, (1 - np.cos(theta)) / theta ** 2)
        V_inv = (
            np.eye(3)
            - wSkew / 2
            + (1 / theta**2) * (1 - C1 / (2 * C2)) * wSkew @ wSkew
        )

        t = self.p
        uInv = V_inv @ t
        return np.block([[wSkew, uInv.reshape(3,1)],[0,0,0,0]])
    
    @classmethod
    def to_vec(cls, X):
        R = X[0:3, 0:3]
        theta = Euler.from_dcm(R)
        p = X[0:3,3]
        return np.block([p,theta])