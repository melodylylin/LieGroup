import casadi as ca
from base import EPS, LieAlgebra, LieGroup, wrap
from SO3 import DCM, Euler, so3

"""
- exp & log: inputs should be in se3 or SE3 format, both return matrix lie group
"""


class se3algebra(LieAlgebra): # param: [x,y,z,theta1,theta2,theta3]
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (6,1) or param.shape == (6,)
        theta = ca.reshape(wrap(param[3:6]), (3,1))
        self.w = so3(theta).wedge
        self.v = ca.reshape(param[0:3], (3,1))

    def add(self, other):
        return se3algebra(self.param + other.param)

    def rmul(self, scalar):
        return se3algebra(scalar * self.param)
    
    def neg(self):
        return se3algebra(-self.param)

    @property
    def wedge(self):
        horz1 = ca.horzcat(self.w, self.v)
        horz2 = ca.DM([0,0,0,0]).T
        return ca.vertcat(horz1, horz2)

    @property
    def ad_matrix(self):
        """
        takes 6x1 lie algebra
        input vee operator [x,y,z,theta1,theta2,theta3]
        """
        v = self.v
        vx = ca.SX(3,3)
        vx[0,1] = -v[2]
        vx[0,2] = v[1]
        vx[1,0] = v[2]
        vx[1,2] = -v[0]
        vx[2,0] = -v[1]
        vx[2,1] = v[0]
        horz1 = ca.horzcat(self.w, vx)
        horz2 = ca.horzcat(ca.DM.zeros(3,3), self.w)
        return ca.vertcat(horz1, horz2)
 
    @classmethod
    def vee(cls, w): # w is 4x4 Lie algebra matrix
        assert w.shape == (4,4)
        x = w[0,3]
        y = w[1,3]
        z = w[2,3]
        theta1 = w[2,1]
        theta2 = w[0,2]
        theta3 = w[1,0]
        return cls(ca.DM([x,y,z,theta1,theta2,theta3])).param

class SE3group(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (6,1) or param.shape == (6,)
        self.param = param
        R = DCM.from_euler(Euler(param[3:6]))
        self.R = R.param
        self.p = ca.reshape(param[0:3], (3,1))

    @staticmethod
    def identity():
        return SE3group(ca.DM([0,0,0,0,0,0]))
    
    @property
    def to_matrix(self):
        horz1 = ca.horzcat(self.R, self.p)
        horz2 = ca.DM([0,0,0,1]).T
        return ca.vertcat(horz1, horz2)

    @property
    def inv(self):  # input a matrix of SX form from casadi
        horz1 = ca.horzcat(self.R.T, ca.mtimes(-self.R.T, self.p))
        horz2 = ca.DM([0,0,0,1]).T
        return ca.vertcat(horz1, horz2)
    
    def product(self, other):
        horz1 = ca.horzcat(ca.mtimes(self.R, other.R), ca.mtimes(self.R, self.p+other.p))
        horz2 = ca.DM([0,0,0,1]).T
        return ca.vertcat(horz1, horz2)

    @property
    def Ad_matrix(self): # Ad matrix of v(6x1) for SE3 Lie Group
        p = self.p
        px = ca.SX(3,3)
        px[0,1] = -p[2]
        px[0,2] = p[1]
        px[1,0] = p[2]
        px[1,2] = -p[0]
        px[2,0] = -p[1]
        px[2,1] = p[0]
        horz1 = ca.horzcat(self.R, ca.mtimes(px, self.R))
        horz2 = ca.horzcat(ca.DM.zeros(3,3), self.R)
        return ca.vertcat(horz1, horz2)
    
    @classmethod
    def to_vec(cls, X):
        R = X[0:3, 0:3]
        theta = Euler.from_dcm(DCM(R)).param
        p = X[0:3,3]
        return ca.vertcat(p,ca.reshape(theta, (3,1)))

    @classmethod
    def log(cls, G: "SE3group") -> "se3algebra": # SE3 matrix to se3 matrix
        X = G.to_matrix
        R = X[:3, :3] # get the SO3 Lie group matrix
        theta = ca.arccos((ca.trace(R) - 1) / 2)
        wSkew = DCM.log(DCM(R)).wedge
        x = ca.SX.sym('x')
        C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < 1e-7, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
        C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < 1e-7, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
        V_inv = (
            ca.SX.eye(3)
            - wSkew / 2
            + (1 / theta**2) * (1 - C1(theta) / (2 * C2(theta))) * wSkew @ wSkew
        )

        t = ca.SX(3, 1)
        t[0] = X[0, 3]
        t[1] = X[1, 3]
        t[2] = X[2, 3]

        uInv = V_inv @ t
        return se3algebra(ca.vertcat(uInv, so3.vee(wSkew)))
    
    @classmethod
    def exp(cls, g:"se3algebra") -> "SE3group": # Lie algebra to Lie group # vw is v in wedge form (se3 lie algebra)
        v = g.param # v = [x,y,z,theta1,theta2,theta3]
        v_so3 = v[3:6]  # grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        X_so3 = so3(v_so3).wedge   # wedge operator for so3
        theta = ca.norm_2(v[3:6])  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)

        # translational components u
        u = ca.SX(3, 1)
        u[0, 0] = v[0]
        u[1, 0] = v[1]
        u[2, 0] = v[2]

        R = DCM.exp(so3(v_so3)) #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational
        x = ca.SX.sym('x')
        C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < 1e-7, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
        C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < 1e-7, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
        C = ca.Function('c', [x], [ca.if_else(ca.fabs(x) < 1e-7, 1/6 - x ** 2 /120 + x ** 4 / 5040, (1 - C1(x)) / x ** 2)]) #(1 - C1(theta)) / theta**2

        V = ca.SX.eye(3) + C2(theta) * X_so3 + C(theta) * X_so3 @ X_so3

        return SE3group(ca.vertcat(V@u, Euler.from_dcm(R).param))
    
se3 = se3algebra
SE3 = SE3group