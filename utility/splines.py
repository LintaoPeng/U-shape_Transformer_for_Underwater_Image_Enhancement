#!/usr/bin/env python3
import torch
import numpy as np


class SplineInterpolator(torch.nn.Module):
    """Module performing spline interpolation.

    Splines are defined by a set of n nodes.  x coordinates of the
    nodes are assumed to be equispaced in the [0, 1] range.
    y coordinates of the nodes are part of the input.

    Given a different set of x coordinates, the module compute
    the interpolated y coordinates.

    """
    
    def __init__(self, nodes, dtype=torch.float32):
        """Create the object.

        Parameters
        ----------
        nodes : int
            number of nodes.
        dtype
            type of internal data.
        """
        super().__init__()
        A = self._precalc(nodes)
        self.register_buffer("A", torch.tensor(A, dtype=dtype))

    def _precalc(self, n):
        # Helper function computing the internal matrix A.
        h = 1.0 / (n - 1)
        mat = 4 * np.eye(n - 2)        
        np.fill_diagonal(mat[1:, :-1], 1)
        np.fill_diagonal(mat[:-1, 1:], 1)
        A = 6 * np.linalg.inv(mat) / (h ** 2)
        z = np.zeros(n - 2)
        A = np.vstack([z, A, z])

        B = np.zeros([n - 2, n])
        np.fill_diagonal(B, 1)
        np.fill_diagonal(B[:, 1:], -2)
        np.fill_diagonal(B[:, 2:], 1)
        A = np.dot(A, B)
        return A.T
        
    def _coefficients(self, y):
        # Helper function computing the coefficients of the polynomials
        # For the given y coordinates of the nodes.
        n = self.A.size(1)
        h = 1.0 / (n - 1)
        M = torch.mm(y, self.A)
        a = (M[:, 1:] - M[:, :-1]) / (6 * h)
        b = M[:, :-1] / 2
        c = (y[:, 1:] - y[:, :-1]) / h - (M[:, 1:] + 2 * M[:, :-1]) * (h / 6)
        return (a, b, c, y[:, :-1])

    def _apply(self, x, coeffs):
        # Helper function interpolating the splines at x.
        # coeffs is the list of coefficients of the polynomials.
        n = self.A.size(1)
        xv = x.view(x.size(0), -1)
        xi = torch.clamp(xv * (n - 1), 0, n - 2).long()
        xf = xv - xi.float() / (n - 1)
        a, b, c, d = (torch.gather(cc, 1, xi) for cc in coeffs)
        z = d + c * xf + b * (xf ** 2) + a * (xf ** 3)
        return z.view_as(x)

    def forward(self, y, x):
        """Interpolate values using splines.

        Parameters
        ----------
        y : tensor (b, n)
            y coordinates for the nodes (one set for each batch).
        x : tensor (b, m1, m2, ..., md)
            values to interpolats (one set for each batch).

        Returns
        -------
        tensor (b, m1, m2, ..., md)
            interpolated values.
        """
        return self._apply(x, self._coefficients(y))

    
def _demo():
    import matplotlib.pyplot as plt
    n = 10
    b = 5
    sp = SplineInterpolator(n)
    y = torch.rand((b, n))
    x = torch.rand((b, 20 * n, 10))
    z = sp(y, x)
    ax = np.linspace(0, 1, n)
    for i in range(b):
        plt.figure()
        plt.plot(ax, y[i, :].cpu().numpy(), 'r.', markersize=25)
        plt.plot(x[i, :].cpu().numpy(), z[i, :].cpu().numpy(), 'b.')
    plt.show()
    

if __name__ == '__main__':
    _demo()
