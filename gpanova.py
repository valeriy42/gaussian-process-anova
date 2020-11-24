from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import pairwise_kernels
import numpy as np
from scipy.special import erf

class GpAnova:
    def kernel_eval(self, input):
        return self.theta_0**2*pairwise_kernels(input, self.X, metric=self.kernel) @ self.z

    def f0(self):
        f_0 = self.theta_0**2 * self.z.T @ np.prod(np.sqrt(np.pi)/(2*self.c) 
        * (erf(self.c*(1-self.X)) + erf(self.c*self.X)), axis=1)
        return f_0

    def prod_ij(self, i, j ):
        return self.z[i] * self.z[j] * np.prod(np.sqrt(np.pi/2)/(2*self.c)
             * np.exp(-0.5*self.c**2*(self.X[i,:]- self.X[j, :])**2)
            * (erf(self.c/np.sqrt(2)*(self.X[i,:] + self.X[j, :])) 
              - erf(self.c/np.sqrt(2)*(self.X[i,:] + self.X[j, :]-2))) )

    def total_variance(self):
        s = 0
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                s += self.prod_ij(i, j)
        sigma2 = self.theta_0**4* s - self.f0()**2
        return sigma2

    def prod_d(self, i, d): 
        return np.sqrt(np.pi)/(2*self.c[d])*(erf(self.c[d]*(1-self.X[i,d])) + erf(self.c[d]* self.X[i,d])) 

    def P_i(self, i, t):
        p = 1
        for d in range(self.X.shape[1]):
            if d != t:
                p *= self.prod_d(i, d)
        return p

    def prod_ijt(self, i,j,t):
        return np.sqrt(np.pi/2)/(2*self.c[t]) \
                * np.exp(-0.5*self.c[t]**2*(self.X[i,t]- self.X[j, t])**2) \
                * (erf(self.c[t]/np.sqrt(2)*(self.X[i,t] + self.X[j, t])) \
                - erf(self.c[t]/np.sqrt(2)*(self.X[i,t] + self.X[j, t]-2)))

    def main_effect(self, t):
        f0 = self.f0()
        s1 = 0
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[0]):
                s1 += self.z[i]*self.z[j]*self.P_i(i, t)*self.P_i(j, t)*self.prod_ijt(i, j, t)
        s1 *= self.theta_0**4
        s2 = 0
        for i in range(self.X.shape[0]):
            s2 += self.z[i] * self.prod_d(i, t) * self.P_i(i, t)
        s2 *= -2*self.theta_0**2*f0
        return s1 + s2 + f0**2

    def f_t(self, x,t):
        s = 0
        for i in range(self.X.shape[0]):
            s += self.z[i]*np.exp(-self.c[t]**2*(x - self.X[i, t])**2)*self.P_i(i, t)
        return self.theta_0**2 * s-self.f0()

    def __init__(self, X, y, theta_0, theta_1):
        self.X = X
        y = y
        ε = 1e-2
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.c = np.sqrt(theta_1**2 + ε)
        length_scale = 1.0/(np.sqrt(2)*self.c)
        self.kernel = RBF(length_scale = length_scale)

        K = theta_0**2*pairwise_kernels(X, metric=self.kernel)
        self.z = np.linalg.solve(K, y)

if __name__ == "__main__":
    pass