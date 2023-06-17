import casadi as ca
from .base import LieAlgebra, LieGroup, EPS, wrap

class so2algebra(LieAlgebra): # euler angle body 3-2-1
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (1,1) or param.shape == (1,)
        self.param = wrap(param)

    def add(self, other):
        return so2algebra(self.param + other.param)

    def rmul(self, scalar):
        return so2algebra(scalar * self.param)
    
    def neg(self):
        return so2algebra(-self.param)

    @property
    def wedge(self):
        algebra = ca.SX.zeros(2, 2)
        algebra[0, 1] = -self.param
        algebra[1, 0] = self.param
        return algebra
    
    @property
    def ad_matrix(self):
        raise NotImplementedError("")

    @classmethod
    def vee(cls, w):
        theta = w[1,0]
        return ca.SX(theta)
    

class SO2group(LieGroup): # input: theta, output: cosine matrix 2x2
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (1, 1) or self.param.shape == (1, )
        self.param = wrap(param)

    @staticmethod
    def identity():
        return SO2group(0)

    @property
    def to_matrix(self):
        theta = self.param[0]
        matrix = ca.SX.zeros(2, 2)
        matrix[0, 0] = ca.cos(theta)
        matrix[0, 1] = -ca.sin(theta)
        matrix[1, 0] = ca.sin(theta)
        matrix[1, 1] = ca.cos(theta)
        return matrix

    @property
    def inv(self):
        return SO2group(-self.param).to_matrix

    def product(self, other: "SO2group"):
        theta = self.param + other.param
        return SO2group(ca.SX(theta))
    
    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X):
        theta = ca.atan2(X[1, 0], X[0, 0])
        return ca.SX(theta)
    
    @classmethod
    def log(cls, G: "SO2group") -> "so2algebra":
        return so2algebra(G.param)
    
    @classmethod
    def exp(cls, g: "so2algebra") -> "SO2group": # so2 -> SO2 matrix
        return SO2group(g.param)# return SO2 matrix


so2 = so2algebra
SO2 = SO2group    
