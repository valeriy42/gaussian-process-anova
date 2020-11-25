import unittest
import numpy as np
from sklearn.datasets import make_regression
from gpanova import GpAnova
from sklearn.preprocessing import MinMaxScaler
import sobol_seq

def relative_error(a, b):
    return np.abs((a-b)/b)


class TestGpAnova(unittest.TestCase):

    def setUp(self):
        self.seed = 1000
        self.dim = 2
        self.n_samples = 20
        self.tolerance = 0.01
        np.random.seed(self.seed)
        theta_0 = np.random.uniform(0.5, 2.0, 1)[0]
        theta_1 = np.random.uniform(0.1, 2.0, self.dim)
        X, y = make_regression(n_samples=self.n_samples, n_features=self.dim,
                               n_informative=self.dim, bias=0.5)
        X = MinMaxScaler().fit_transform(X)
        self.GpAnova = GpAnova(X, y, theta_0, theta_1)
        self.mc_samples = 10**4
        self.Xmc = sobol_seq.i4_sobol_generate(self.dim, self.mc_samples)

    def test_f0(self):
        f0_actual = self.GpAnova.f0()
        f0_expected = np.mean(self.GpAnova.kernel_eval(self.Xmc))
        self.assertLess(relative_error(f0_actual, f0_expected), self.tolerance)

    def test_total_variance(self):
        tv_actual = self.GpAnova.total_variance()
        tv_expected = np.mean(
            (self.GpAnova.kernel_eval(self.Xmc) - self.GpAnova.f0())**2)
        self.assertLess(relative_error(tv_actual, tv_expected), self.tolerance)

    def test_f_t(self):
        x = np.random.uniform(size=self.dim)
        for d in range(self.dim):
            Xt = np.copy(self.Xmc)
            Xt[:, d] = x[d]
            f_t_expected = np.mean(
                self.GpAnova.kernel_eval(Xt)-self.GpAnova.f0())
            f_t_actual = self.GpAnova.f_t(x[d], d)
            self.assertLess(relative_error(f_t_actual, f_t_expected), self.tolerance,
                            msg='actual: {} expected: {}'.format(f_t_actual, f_t_expected))

    def test_main_effects(self):
        for d in range(self.dim):
            main_effect_actual = self.GpAnova.main_effect(d)
            main_effect_expected = np.mean((self.GpAnova.f_t(self.Xmc, d))**2)
            self.assertLess(relative_error(
                main_effect_actual, main_effect_expected), self.tolerance)


if __name__ == "__main__":
    unittest.main()
