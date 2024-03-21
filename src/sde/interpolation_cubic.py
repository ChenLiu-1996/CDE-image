import torch
from torchcde import interpolation_base
from . import misc


def _natural_cubic_spline_coeffs_without_missing_values(x, t):
    # x should be a tensor of shape (length, ...)
    # x should be a tensor of shape (length,)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    assert len(t.shape) == 1
    assert x.shape[0] == t.shape[0]

    x_dims = len(x.shape) - 1
    length = t.shape[0]

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")

    elif length == 2:
        a = x[:1, ...]
        time_elapsed = t[1:] - t[:1]
        for _ in range(x_dims):
            time_elapsed = time_elapsed[..., None]

        assert len(time_elapsed.shape) == len(x.shape)
        b = (x[1:, ...] - x[:1, ...]) / time_elapsed

        two_c = torch.zeros(1, *x.shape[1:], dtype=x.dtype, device=x.device)
        three_d = torch.zeros(1, *x.shape[1:], dtype=x.dtype, device=x.device)

    else:
        # Set up some intermediate values
        time_diffs = t[1:] - t[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        for _ in range(x_dims):
            time_diffs_reciprocal_squared = time_diffs_reciprocal_squared[..., None]

        three_path_diffs = 3 * (x[1:, ...] - x[:-1, ...])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty(length, dtype=x.dtype, device=x.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(x)
        system_rhs[:-1, ...] = path_diffs_scaled
        system_rhs[-1, ...] = 0
        system_rhs[1:, ...] += path_diffs_scaled

        # Move the time axis to the last dimension to accommodate the solver.
        knot_derivatives = misc.tridiagonal_solve(
            torch.transpose(system_rhs, 0, -1),
            time_diffs_reciprocal, system_diagonal, time_diffs_reciprocal)
        # Move the time axis back.
        knot_derivatives = torch.transpose(knot_derivatives, 0, -1)

        # Do some algebra to find the coefficients of the spline
        for _ in range(x_dims):
            time_diffs_reciprocal = time_diffs_reciprocal[..., None]

        a = x[:-1, ...]
        b = knot_derivatives[:-1, ...]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[:-1, ...]
                 - 2 * knot_derivatives[1:, ...]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[:-1, ...]
                          + knot_derivatives[1:, ...])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d


# The mathematics of this are adapted from  http://mathworld.wolfram.com/CubicSpline.html, although they only treat the
# case of each piece being parameterised by [0, 1]. (We instead take the length of each piece to be the difference in
# time stamps.)
def natural_cubic_spline_coeffs(x, t):

    assert not torch.isnan(x).any()
    a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(x=x, t=t)

    coeffs = torch.cat([a, b, two_c, three_d], dim=0)  # for simplicity put them all together
    return coeffs


class CubicSpline(interpolation_base.InterpolationBase):
    """
    Calculates the cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        # T is the time dimension (same length as t).
        x = torch.rand(T, C, H, W)
        coeffs = natural_cubic_coeffs(x)
        # ...at this point you can save coeffs, put it through PyTorch's Datasets and DataLoaders, etc...
        spline = CubicSpline(coeffs)
        point = torch.tensor(0.4)
        # will be a tensor of shape (C, H, W)
        out = spline.derivative(point)
    """

    def __init__(self, coeffs, t, **kwargs):
        super().__init__(**kwargs)

        # NOTE: The first dimension of `coeffs` contains 4 different coefficients of the same dimensionality,
        # and hence `coeffs.size(0)` shall be an integer multiple of 4.
        channels = coeffs.size(0) // 4
        if channels * 4 != coeffs.size(0):  # check that it's a multiple of 4
            raise ValueError("Passed invalid coeffs.")
        a, b, two_c, three_d = (coeffs[:channels, ...], coeffs[channels:2 * channels, ...],
                                coeffs[2 * channels:3 * channels, ...], coeffs[3 * channels:, ...])

        self.register_buffer('_t', t)
        self.register_buffer('_a', a)
        self.register_buffer('_b', b)
        self.register_buffer('_two_c', two_c)
        self.register_buffer('_three_d', three_d)

    @property
    def grid_points(self):
        return self._t

    @property
    def interval(self):
        return torch.stack([self._t[0], self._t[-1]])

    def _interpret_t(self, t):
        t = torch.as_tensor(t, dtype=self._b.dtype, device=self._b.device)
        maxlen = self._b.size(0) - 1
        # clamp because t may go outside of [t[0], t[-1]]; this is fine
        index = torch.bucketize(t.detach(), self._t.detach()).sub(1).clamp(0, maxlen)
        # will never access the last element of self._t; this is correct behaviour
        fractional_part = t - self._t[index]
        return fractional_part, index

    def evaluate(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = 0.5 * self._two_c[index, ...] + self._three_d[index, ...] * fractional_part / 3
        inner = self._b[index, ...] + inner * fractional_part
        return self._a[index, ...] + inner * fractional_part

    def derivative(self, t):
        fractional_part, index = self._interpret_t(t)
        fractional_part = fractional_part.unsqueeze(-1)
        inner = self._two_c[index, ...] + self._three_d[index, ...] * fractional_part
        deriv = self._b[index, ...] + inner * fractional_part
        return deriv