import unittest

from pathlib import Path
import cProfile
from pstats import Stats

import casadi as ca

from base import EPS
# from cyecca.lie.r import LieGroupR, LieAlgebraR
from SO2 import so2, SO2
from SO3 import so3, DCM, Quat, Euler, MRP
from SE3 import se3, SE3
from SE2 import se2, SE2


class ProfiledTestCase(unittest.TestCase):
    def setUp(self):
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self) -> None:
        p = Stats(self.pr)
        p.strip_dirs()
        p.sort_stats("cumtime")
        profile_dir = Path(".profile")
        profile_dir.mkdir(exist_ok=True)
        p.dump_stats(profile_dir / self.id())


class Test_LieAlgebraSO2(ProfiledTestCase):
    def test_ctor(self):
        v = ca.DM([1])
        G1 = so2(ca.DM([1]))

# class Test_LieGroupSO2(ProfiledTestCase):
#     def test_ctor(self):
#         v = ca.DM([1])
#         G1 = SO2(ca.DM([1]))

#     def test_identity(self):
#         e = SO2.identity()
#         G2 = SO2(ca.DM([1]))
#         self.assertEqual(SO2(e * G2), G2)
#         self.assertEqual(SO2(G2 * e), G2)
#         self.assertEqual(G2, G2)

#     def test_addition(self):
#         g = so2(ca.DM([.3]))
#         self.assertEqual(g + g, so2(g.param*2))
#         self.assertEqual(g - g, so2(ca.DM([ 0])))
#         self.assertEqual(-g, so2(ca.DM([-.3])))

#     def test_exp_log(self):
#         g = so2(ca.DM([.3]))
#         self.assertEqual(g, SO2.log(SO2.exp(g)))

# class Test_LieGroupSE2(ProfiledTestCase):
#     def test_ctor(self):
#         v = ca.DM([1])
#         G1 = se2(ca.DM([1,0,0]))

#     def test_identity(self):
#         e = SE2.identity()
#         G2 = SE2(ca.DM([0,1,0]))
#         self.assertEqual(SE2(SE2.to_vec(e * G2)), G2)
#         self.assertEqual(SE2(SE2.to_vec(G2 * e)), G2)
#         self.assertEqual(G2, G2)

#     def test_addition(self):
#         g = se2(ca.DM([1,2,0.4]))
#         self.assertEqual(g + g, se2(g.param*2))
#         self.assertEqual(g - g, se2(ca.DM([0,0,0])))
#         self.assertEqual(-g, se2(ca.DM([-1,-2,-0.4])))

#     def test_exp_log(self):
#         g = se2(ca.DM([1,2,0.4]))
#         self.assertEqual(g, SE2.log(SE2.exp(g)))


class Test_LieGroupSO3(ProfiledTestCase):
    def test_ctor(self):    
        v = ca.DM([1])
        G1 = MRP(ca.DM([1, 0, 0, 0]))

    def test_identity(self):
        e = MRP.identity()
        G2 = MRP(ca.DM([0, 1, 0, 0]))
        self.assertEqual(MRP(e * G2), G2)
        self.assertEqual(MRP(G2 * e), G2)
        self.assertEqual(G2, G2)

    def test_addition(self):
        g = so3(ca.DM([1, 2, 3]))
        self.assertEqual(g + g, so3(g.param*2))
        self.assertEqual(g - g, so3(ca.DM([0, 0, 0])))
        self.assertEqual(-g, so3(ca.DM([-1, -2, -3])))

    def test_exp_log(self):
        g = so3(ca.DM([.5, .2, .3]))
        self.assertEqual(g, Euler.log(Euler.exp(g)))

# class Test_LieGroupSE3(ProfiledTestCase):
#     def test_ctor(self):    
#         v = ca.DM([1])
#         G1 = SE3(ca.DM([1, 0, 0, 0, 0, 0]))

#     def test_identity(self):
#         e = SE3.identity()
#         G2 = SE3(ca.DM([0, 1, 0, 0, 0, 0]))
#         self.assertEqual(SE3(SE3.to_vec(e * G2)), G2)
#         self.assertEqual(SE3(SE3.to_vec(G2 * e)), G2)
#         self.assertEqual(G2, G2)

#     def test_addition(self):
#         g = se3(ca.DM([1,2,3,.1, .2, .3]))
#         self.assertEqual(g + g, se3(g.param*2))
#         self.assertEqual(g - g, se3(ca.DM([0,0,0,0, 0, 0])))
#         self.assertEqual(-g, se3(ca.DM([-1,-2,-3,-.1, -.2, -.3])))

#     def test_exp_log(self):
#         g = se3(ca.DM([1,2,3,.1, .2, .3]))
#         self.assertEqual(g, SE3.log(SE3.exp(g)))


if __name__ == "__main__":
    unittest.main()