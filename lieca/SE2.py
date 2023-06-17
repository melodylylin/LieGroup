import casadi as ca
from .base import EPS, LieAlgebra, LieGroup
from .SO2 import so2, SO2, wrap
    
class se2algebra(LieAlgebra):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.theta = wrap(param[2])
        self.w = so2(self.theta).wedge
        self.v = ca.reshape(param[0:2], (2,1))
        self.param = ca.vertcat(self.v, self.theta[0])
    
    def add(self, other):
        return se2algebra(self.param + other.param)

    def rmul(self, scalar):
        return se2algebra(scalar * self.param)
    
    def neg(self):
        return se2algebra(-self.param)

    @property
    def wedge(self):
        horz1 = ca.horzcat(self.w, self.v)
        horz2 = ca.SX([0, 0, 0]).T
        return ca.vertcat(horz1, horz2)
    
    @property
    def ad_matrix(self):
        x, y, theta = self.v[0], self.v[1], self.theta[0]
        ad = ca.SX(3,3)
        ad[0,1] = -theta
        ad[0,2] = y
        ad[1,0] = theta
        ad[1,2] = -x
        return ad
    
    @classmethod
    def vee(cls, w):
        assert w.shape == (3,3)
        x = w[0,2]
        y = w[1,2]
        theta = w[1,0]
        return cls(ca.DM([x,y,theta])).param


class SE2group(LieGroup):
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (3,1) or param.shape == (3,)
        self.theta = wrap(param[2])
        self.R = SO2(self.theta).to_matrix
        self.p = ca.reshape(param[0:2], (2,1))
        self.param = ca.vertcat(self.p, self.theta[0])
    
    @staticmethod
    def identity():
        return SE2group(ca.DM([0,0,0]))
    
    @property
    def to_matrix(self):
        horz1 = ca.horzcat(self.R, self.p)
        horz2 = ca.DM([0,0,1]).T
        return ca.vertcat(horz1, horz2)
    
    @property
    def inv(self):
        horz1 = ca.horzcat(self.R.T, ca.mtimes(-self.R.T, self.p))
        horz2 = ca.DM([0,0,1]).T
        return ca.vertcat(horz1, horz2)
    
    def product(self, other):
        x = ca.vertcat(ca.mtimes(self.R, self.p+other.p), self.theta+other.theta)
        return SE2group(x)

    @property
    def Ad_matrix(self):
        v = ca.vertcat(self.p[1], -self.p[0])
        horz1 = ca.horzcat(self.R, v)
        horz2 = ca.DM([0,0,1]).T
        return ca.vertcat(horz1, horz2)
    
    @classmethod
    def to_vec(cls, X):
        R = X[0:2,0:2]
        theta = SO2.to_vec(R)
        p = X[0:2,2]
        return ca.vertcat(p,theta)

    @classmethod
    def log(cls, G: "SE2group") -> se2algebra:
        v = ca.reshape(G.p, (2,1)) 
        theta = G.param[2]
        x = ca.SX.sym('x')
        C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < 1e-3, 1 - x**2/6 + x**4/120, ca.sin(x)/x)])
        C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < 1e-3, x/2 - x**3/24 + x**5/720, (1 - ca.cos(x))/x)])
        a = C1(theta)
        b = C2(theta)
        V_inv = ca.SX(2,2)
        V_inv[0,0] = a
        V_inv[0,1] = b
        V_inv[1,0] = -b
        V_inv[1,1] = a
        V_inv = V_inv/(a**2 + b**2)
        p = V_inv@v
        return se2algebra(ca.vertcat(p, theta))
    
    @classmethod
    def exp(cls, g: "se2algebra") -> "SE2group":

        theta = g.theta[0]
        x = ca.SX.sym('x')
        C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < 1e-3, 1 - x**2/6 + x**4/120, ca.sin(x)/x)])
        C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < 1e-3, x/2 - x**3/24 + x**5/720, (1 - ca.cos(x))/x)])
        a = C1(theta)
        b = C2(theta)
        V = ca.SX(2,2)
        V[0,0] = a
        V[0,1] = -b
        V[1,0] = b
        V[1,1] = a
        p = V@(g.v)
        return SE2group(ca.vertcat(p,theta))
    
se2 = se2algebra
SE2 = SE2group

# def diff_correction(e: se2, n=100):
#     # computes (1 - exp(-ad_x)/ad_x = sum k=0^infty (-1)^k/(k+1)! (ad_x)^k
#     ad = e.ad_matrix
#     ad_i = np.eye(3)
#     s = np.zeros((3, 3))
#     for k in range(n):
#         s += ((-1)**k/math.factorial(k+1))*ad_i
#         ad_i = ad_i @ ad
#     return -np.linalg.inv(s)@((-e).exp.Ad_matrix)

# def se2_diff_correction(e: se2): # U
#     x = e.x
#     y = e.y
#     theta = e.theta
#     with np.errstate(divide='ignore',invalid='ignore'):
#         a = ca.if_else(abs(theta) > 1e-3, -theta*np.sin(theta)/(2*(np.cos(theta) - 1)), 1 - theta**2/12 - theta**4/720)
#         b = ca.if_else(abs(theta) > 1e-3, -(theta*x*np.sin(theta) + (1 - np.cos(theta))*(theta*y - 2*x))/(2*theta*(1 - np.cos(theta))), -y/2 + theta*x/12 - theta**3*x/720)
#         c = ca.if_else(abs(theta) > 1e-3, -(theta*y*np.sin(theta) + (1 - np.cos(theta))*(-theta*x - 2*y))/(2*theta*(1 - np.cos(theta))), x/2 + theta*y/12 + theta**3*y/720)
#     return -np.array([
#         [a, theta/2, b],
#         [-theta/2, a, c],
#         [0, 0, 1]
#     ])

# def se2_diff_correction_inv(e: se2): # U_inv
#     x = e.x
#     y = e.y
#     theta = e.theta
#     with np.errstate(divide='ignore',invalid='ignore'):
#         a = ca.if_else(abs(theta) > 1e-3, np.sin(theta)/theta, 1 - theta**2/6 + theta**4/120)
#         b = ca.if_else(abs(theta) > 1e-3, (1  - np.cos(theta))/theta, theta/2 - theta**3/24)
#         c = ca.if_else(abs(theta) > 1e-3, -(x*(theta*np.cos(theta) - theta + np.sin(theta) - np.sin(2*theta)/2) + y*(2*np.cos(theta) - np.cos(2*theta)/2 - 3/2))/(theta**2*(1 - np.cos(theta))), y/2 + theta*x/6 - theta**2*y/24 - theta**3*x/120 + theta**4*y/720)
#         d = ca.if_else(abs(theta) > 1e-3, -(x*(-2*np.cos(theta) + np.cos(2*theta)/2 + 3/2) + y*(theta*np.cos(theta) - theta + np.sin(theta) - np.sin(2*theta)/2))/(theta**2*(1 - np.cos(theta))), -x/2 + theta*y/6 + theta**2*x/24 - theta**3*y/120 - theta**4*x/720)
#     return -np.array([
#         [a, -b, c],
#         [b, a, d],
#         [0, 0, 1]
#     ])