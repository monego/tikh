from scipy.linalg import norm
from itertools import islice, count
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.optimize import fmin_cg
from scipy.signal import convolve2d


class Tikhonov:

    def __init__(self, ɑ_min, ɑ_max, Δɑ, ɛ):
        self.ɑ_min = ɑ_min
        self.ɑ_max = ɑ_max
        self.Δɑ = Δɑ
        self.ɛ = ɛ

    def Morosov(self, B, K, center, op):
        """Calculates the optimum alpha by the Morosov criterion.

        Parameters
        ----------
        B: float
        Noisy image

        K:
        Point spread function

        center:
        Center of the PSF

        op:
        Tikhonov operator

        Returns
        -------
        alpha: float
        Optimum alpha
        """

        ɑ = np.arange(self.ɑ_min, self.ɑ_max + self.Δɑ*1.1, self.Δɑ*1.1)
        t_ɑ = ɑ.size
        error_mi = np.zeros(t_ɑ)

        for i in range(t_ɑ):
            X = self.tikhonov(B, K, center, op, ɑ[i])
            error_mi[i] = np.sum((X - B)**2)
            error_mi[i] = np.sqrt(error_mi[i])
            if error_mi[i] < self.ɛ:
                ɑ_m = ɑ[i]

        return ɑ_m

    def tikhonov(self, B, K, center, op, alpha):
        """
        Execute Tikhonov method. dim = image size in each dimension.

        Parameters
        ---------
        B : array dim x dim
            Input image.
        K : array dim x dim
            K as calculated by the PSF.
        center : list or tuple
            A list or tuple of two values given the center pixel from the PSF
        op : array 3 x 3.
            Tikhonov operator order.
        alpha : float
            Regularization parameter.

        Returns
        -------
        Im : array dim x dim
            Restored image.
        """

        # B = image (with noise) [512,512]
        # K = value found by the PSF [512,512]
        # center = center of the point spread function = [256,256]
        # op = 3x3 matrix
        # alpha = scalar

        des = np.array([2, 2])

        # Calculate the FFT on the image, flatten it.
        B_tf = np.fft.fft2(B).flatten()

        # Shift K over both dimensions.
        K_tf = np.fft.fft2(np.roll(K, 1-center, axis=(0, 1))).flatten()

        # Create a zero matrix the size of the image.
        P = np.zeros(B.shape)

        # B.shape = (512, 512)
        # op.shape = (3, 3)
        # op.shape[0] = 3
        # op.shape[1] = 3
        # 0:3 elements in every dimension will receive the "op" matrix
        P[0:op.shape[0], 0:op.shape[1]] = op

        # Calculate fft of this matrix too, roll P on both dimensions
        P_tf = np.fft.fft2(np.roll(P, 1-des, axis=(0, 1))).flatten()

        den = np.conjugate(K_tf)*K_tf + np.abs(alpha**2)*(np.conjugate(P_tf)*P_tf)
        X = np.conjugate(K_tf) * B_tf
        W = X/den
        W = np.reshape(W, B.shape)
        Im = np.fft.ifft2(W).real

        return Im


# The results from the entropy denoising are subpar.
# TODO: Re-check the implementation.
class Entropy:

    def __init__(self, img, s, alpha, A):
        self._dims = (512, 512)
        self._img = resize(img, self._dims, anti_aliasing=True)
        self._s = s
        self._A = A
        self._i, self._j = self._img.shape
        self._chi = 0.1
        self._alpha = alpha
        self._Nx = len(self._img.flatten())
        self._Omega1 = np.zeros((self._Nx, self._Nx))
        self._Omega2 = np.zeros((self._Nx, self._Nx))
        self._Smax0 = np.log(self._Nx)
        self._Smax1 = np.log(self._Nx - 1)
        self._Smax2 = np.log(self._Nx - 2)

        for i in islice(count(), 0, self._Nx-1):
            self._Omega1[i][i] = -1.
            if i + 1 < self._Nx:
                self._Omega1[i][i+1] = 1.

        for i in islice(count(), 1, self._Nx-1):
            self._Omega2[i][i] = -2.
            if i + 1 < self._Nx:
                self._Omega2[i][i+1] = 1.
            if i - 1 >= 0:
                self._Omega2[i][i-1] = 1.

    def _Omega_e0(self, x0):
        """
        0-order entropy regularization
        """
        fmin = np.min(x0)
        p = x0 - fmin + self._chi
        psum = np.sum(p)
        s = p/psum
        return 1 - np.sum(s*np.log(s))/self._Smax0

    def _Omega_e1(self, x0):
        """
        1-order entropy regularization
        """

        fmin = np.min(x0)
        fmax = np.max(x0)
        p = self._Omega1.dot(x0) + (fmax - fmin) + self._chi
        if np.min(p) < 0:
            print('alert')
            os.sys.exit('1')
        psum = np.sum(p[0:self._Nx-1])
        s = p/psum
        if np.min(s) < 0:
            print('alert')
            os.sys.exit('1')
        return 1.0 - np.sum(s*np.log(s))/self._Smax1

    def _Omega_e2(self, x0):
        """
        2-order entropy regularization
        """

        if len(x0) != self._Nx:
            return None
        else:
            fmin = np.min(x0)
            fmax = np.max(x0)
            p = self._Omega2.dot(x0) + 2*(fmax - fmin) + self._chi
            if np.min(p) < 0:
                print(p)
                print('alert')
                os.sys.exit('1')
            psum = np.sum(p[1:self._Nx-1])
            s = p/psum
            return 1.0 - np.sum(s*np.log(s))/self._Smax2

    def _callbackF(self, x0):
        print(self._minf(x0))

    def _minf(self, x):
        return norm(
            convolve2d(self._A, x.reshape(self._dims), mode='same')
            - self._img, ord='fro') + self._alpha*self._Omega_e2(x)

    def entropy(self, order):

        x0 = self._img

        sol = fmin_cg(f=self._minf, x0=x0, gtol=1e-2, maxiter=5)

        assert not np.array_equal(sol.reshape(self._dims), x0), "x0 is equal to solution"

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(x0, cmap=plt.cm.gray)
        ax[1].imshow(sol.reshape(self._dims) - x0, cmap=plt.cm.gray)
        plt.savefig("tmp.png", bbox_inches='tight')

        return sol.reshape(self._dims)
