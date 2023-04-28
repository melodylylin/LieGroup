import numpy as np
from base import LieAlgebra, LieGroup, EPS

def wrap(x):
    return np.where(np.abs(x) >= np.pi, (x + np.pi) % (2 * np.pi) - np.pi, x)

class so2algebra(LieAlgebra): # euler angle body 3-2-1
    def __init__(self, param):
        super().__init__(param)
        assert param.shape == (1,1) or param.shape == (1,)
        self.param = np.reshape(wrap(param), ())

    def add(self, other):
        return so2algebra(self.param + other.param)

    def rmul(self, scalar):
        return so2algebra(scalar * self.param)
    
    def neg(self):
        return so2algebra(-self.param)

    @property
    def wedge(self):
        theta = self.param
        return np.array([
            [0, -theta],
            [theta, 0]
        ])
    
    @property
    def ad_matrix(self):
        raise NotImplementedError("")

    @classmethod
    def vee(cls, w):
        theta = w[1,0]
        return np.array([theta])
    

class SO2group(LieGroup): # input: theta, output: cosine matrix 2x2
    def __init__(self, param):
        super().__init__(param)
        assert self.param.shape == (1, 0) or self.param.shape == (1, )
        self.param = np.reshape(wrap(param), ())

    @staticmethod
    def identity():
        return np.eye(2)

    @property
    def to_matrix(self):
        return np.array([
            [np.cos(self.param), -np.sin(self.param)],
            [np.sin(self.param), np.cos(self.param)]
        ])

    @property
    def inv(self):
        return self(-self.param).to_matrix

    @property
    def product(self, other: "SO2group"):
        raise NotImplementedError("")
    
    @property
    def Ad_matrix(self):
        return self.to_matrix
    
    @classmethod
    def to_vec(cls, X):
        theta = np.arctan2(X[1, 0], X[0, 0])
        return np.array([theta])
    
    @classmethod
    def log(cls, G: "SO2group") -> "so2algebra":
        return so2algebra(G.param).wedge
    
    @classmethod
    def exp(cls, g: "so2algebra") -> "SO2group": # so2 -> SO2 matrix
        return SO2group(g.param).to_matrix # return SO2 matrix


so2 = so2algebra
SO2 = SO2group    
