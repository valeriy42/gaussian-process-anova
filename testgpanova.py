import unittest
import numpy as np
from sklearn.datasets import make_regression
from gpanova import GpAnova
from sklearn.preprocessing import MinMaxScaler

def relative_error(a, b):
    return np.abs((a-b)/b)

class TestGpAnova(unittest.TestCase):

    def setUp(self):
        self.seed = 1001
        self.dim = 2
        self.n_samples = 20
        np.random.seed(self.seed)
        theta_0 = np.random.uniform(0.5, 2.0, 1)[0]
        theta_1 = np.random.uniform(0.1, 2.0, self.dim)
        X, y = make_regression(n_samples=self.n_samples, n_features=self.dim,
                            n_informative=self.dim, bias=0.5, random_state=self.seed)
        X = MinMaxScaler().fit_transform(X)
        self.GpAnova = GpAnova(X, y, theta_0, theta_1)
        self.mc_samples = 10**6

    def test_f0(self):
        f0_actual = self.GpAnova.f0()
        X = np.random.uniform(
            size=self.dim*self.mc_samples).reshape((-1, self.dim))
        f0_expected = np.mean(self.GpAnova.kernel_eval(X))
        self.assertLess(relative_error(f0_actual, f0_expected), 0.005)

    def test_total_variance(self):
        tv_actual = self.GpAnova.total_variance()
        X = np.random.uniform(
            size=self.dim*self.mc_samples).reshape((-1, self.dim))
        tv_expected = np.mean((self.GpAnova.kernel_eval(X) - self.GpAnova.f0())**2)
        self.assertLess(relative_error(tv_actual, tv_expected), 0.005)

    def test_f_t(self):
        X = np.random.uniform(size=self.mc_samples*self.dim).reshape((-1,self.dim))
        x = np.random.uniform(size=self.dim)
        for d in range(self.dim):
            Xt = np.copy(X)
            Xt[:,d] = x[d]
            f_t_expected = np.mean(self.GpAnova.kernel_eval(Xt)-self.GpAnova.f0())
            f_t_actual = self.GpAnova.f_t(x[d], d)
            self.assertLess(relative_error(f_t_actual, f_t_expected), 0.005,
             msg='actual: {} expected: {}'.format(f_t_actual, f_t_expected))

    def test_main_effects(self):
        X = np.random.uniform(
            size=self.dim*self.mc_samples).reshape((-1, self.dim))
        for d in range(self.dim):
            main_effect_actual = self.GpAnova.main_effect(d)
            main_effect_expected = np.mean((self.GpAnova.f_t(X, d))**2)
            self.assertLess(relative_error(main_effect_actual, main_effect_expected), 0.005)



if __name__ == "__main__":
    unittest.main()
