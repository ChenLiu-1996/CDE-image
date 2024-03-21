import torch
import torchdiffeq
import torchsde
import warnings


class _VectorField(torch.nn.Module):
    def __init__(self, X, func):
        super().__init__()

        self.X = X
        self.func = func

        # torchsde backend
        self.sde_type = getattr(func, "sde_type", "stratonovich")
        self.noise_type = getattr(func, "noise_type", "additive")

    # torchdiffeq backend
    def forward(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.X.derivative(t)

        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(t, z)

        # NOTE: This is completely different from the original formulation.
        # I don't even know if this is reasonable.
        # Not using matrix-vector multiplication because instead being
        # \mathbb{R}^{w} -> \mathbb{R}^{v}
        # we are trying to do
        # \mathbb{R}^{c x h x w} -> \mathbb{R}^{c x h x w}
        # which cannot be simply solved by a matrix multiplication.
        out = (vector_field * control_gradient.unsqueeze(0)).squeeze(0)

        return out

    # torchsde backend
    f = forward

    def g(self, t, z):
        return torch.zeros_like(z).unsqueeze(0)


def cdeint(X, func, z0, t, adjoint=True, backend="torchdiffeq", **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(s, z_s) dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        X: The control. This should be a instance of `torch.nn.Module`, with a `derivative` method. For example
            `torchcde.CubicSpline`. This represents a continuous path derived from the data. The
            derivative at a point will be computed via `X.derivative(t)`, where t is a scalar tensor. The returned
            tensor should have shape (..., input_channels), where '...' is some number of batch dimensions and
            input_channels is the number of channels in the input path.
        func: Should be a callable describing the vector field f(t, z). If using `adjoint=True` (the default), then
            should be an instance of `torch.nn.Module`, to collect the parameters for the adjoint pass. Will be called
            with a scalar tensor t and a tensor z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `X` arguments as above. The '...' corresponds to some number of batch dimensions. If it
            has a method `prod` then that will be called to calculate the matrix-vector product f(t, z) dX_t/dt, via
            `func.prod(t, z, dXdt)`.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate. Defaults to True.
        backend: Either "torchdiffeq" or "torchsde". Which library to use for the solvers. Note that if using torchsde
            that the Brownian motion component is completely ignored -- so it's still reducing the CDE to an ODE --
            but it makes it possible to e.g. use an SDE solver there as the ODE/CDE solver here, if for some reason
            that's desired.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq (the most common are `rtol`, `atol`,
            `method`, `options`) or the sdeint solver of torchsde.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(s, z_s)dX_s, where t_i = t[i].
        This will be a tensor of shape (..., len(t), hidden_channels).

    Raises:
        ValueError for malformed inputs.

    Note:
        Supports tupled input, i.e. z0 can be a tuple of tensors, and X.derivative and func can return tuples of tensors
        of the same length.

    Warnings:
        Note that the returned tensor puts the sequence dimension second-to-last, rather than first like in
        `torchdiffeq.odeint` or `torchsde.sdeint`.
    """

    # Reduce the default values for the tolerances because CDEs are difficult to solve with the default high tolerances.
    if 'atol' not in kwargs:
        kwargs['atol'] = 1e-6
    if 'rtol' not in kwargs:
        kwargs['rtol'] = 1e-4
    if adjoint:
        if 'adjoint_atol' not in kwargs:
            kwargs['adjoint_atol'] = kwargs['atol']
        if 'adjoint_rtol' not in kwargs:
            kwargs['adjoint_rtol'] = kwargs['rtol']

    assert isinstance(z0, torch.Tensor)

    if adjoint and 'adjoint_params' not in kwargs:
        for buffer in X.buffers():
            # Compare based on id to avoid PyTorch not playing well with using `in` on tensors.
            if buffer.requires_grad:
                warnings.warn("One of the inputs to the control path X requires gradients but "
                              "`kwargs['adjoint_params']` has not been passed. This is probably a mistake: these "
                              "inputs will not receive a gradient when using the adjoint method. Either have the input "
                              "not require gradients (if that was unintended), or include it (and every other "
                              "parameter needing gradients) in `adjoint_params`. For example:\n"
                              "```\n"
                              "coeffs = ...\n"
                              "func = ...\n"
                              "X = CubicSpline(coeffs)\n"
                              "adjoint_params = tuple(func.parameters()) + (coeffs,)\n"
                              "cdeint(X=X, func=func, ..., adjoint_params=adjoint_params)\n"
                              "```")

    vector_field = _VectorField(X=X, func=func)
    if backend == "torchdiffeq":
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
    elif backend == "torchsde":
        sdeint = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        out = sdeint(sde=vector_field, y0=z0, ts=t, **kwargs)
    else:
        raise ValueError(f"Unrecognised backend={backend}")

    return out