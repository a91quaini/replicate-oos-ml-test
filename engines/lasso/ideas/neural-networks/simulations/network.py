"""Trained bounded probability networks and independently checkable gradients.

Every hidden weight is optimized. Multiple starts supply a local numerical fit,
not a certificate of global empirical-risk minimization.
"""
import numpy as np
from scipy.optimize import minimize


class Network:
    def __init__(self, inputs, width):
        self.inputs, self.width = int(inputs), int(width)
        self.dimension = (inputs+1)*width+width+inputs+1

    def unpack(self, theta):
        edge = (self.inputs+1)*self.width
        return theta[:edge].reshape(self.inputs+1, self.width), theta[edge:edge+self.width], theta[edge+self.width:]

    def forward(self, theta, x):
        d = np.column_stack((np.ones(len(x)), x))
        w, a, b = self.unpack(theta)
        h = np.tanh(d@w)
        z = np.tanh((h@a)/np.sqrt(self.width)+d@b)
        return .5+.49*z, (d, h, a, z)

    def penalty_vector(self, theta):
        """Penalize every coordinate except the affine output intercept."""
        penalized = np.asarray(theta).copy()
        penalized[-self.inputs-1] = 0.
        return penalized

    def objective(self, theta, x, y, weight_decay=0.0):
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        pred, (d, h, a, z) = self.forward(theta, x)
        residual = pred-y
        g = .49*residual*(1-z*z)/len(y)
        gw = d.T@((g[:, None]*a[None, :])*(1-h*h))/np.sqrt(self.width)
        ga = h.T@g/np.sqrt(self.width)
        gb = d.T@g
        value, gradient = float(.5*np.mean(residual**2)), np.r_[gw.ravel(), ga, gb]
        # Keep the lambda=0 arithmetic exactly identical to the archived fit.
        if weight_decay:
            penalized = self.penalty_vector(theta)
            value += float(.5*weight_decay*(penalized@penalized))
            gradient += weight_decay*penalized
        return value, gradient

    def fit(self, x, y, seed, maxiter=600, starts=2, weight_decay=0.0):
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        rng = np.random.default_rng(seed)
        candidates = []
        for _ in range(starts):
            init = rng.normal(0, .25, self.dimension)
            init[-self.inputs-1:] = 0
            init[-self.inputs-1] = np.arctanh(np.clip((y.mean()-.5)/.49, -.8, .8))
            fit = minimize(lambda z: self.objective(z, x, y, weight_decay), init, jac=True, method='L-BFGS-B',
                bounds=[(-3., 3.)]*self.dimension,
                options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6, maxls=40))
            if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all() or not np.isfinite(fit.jac).all():
                raise ArithmeticError('Nonfinite neural fit')
            candidates.append(fit)
        best = min(candidates, key=lambda z: z.fun)
        self.theta = best.x.copy()
        unpenalized, _ = self.objective(self.theta, x, y)
        penalized = self.penalty_vector(self.theta)
        penalized_norm_squared = float(penalized@penalized)
        self.diagnostics = dict(converged=bool(best.success), status=int(best.status), message=str(best.message),
            iterations=int(best.nit), objective=float(best.fun),
            projected_gradient=float(np.max(np.abs(best.x-np.clip(best.x-best.jac, -3, 3)))),
            restart_spread=float(max(z.fun for z in candidates)-min(z.fun for z in candidates)),
            all_start_objectives=[float(z.fun) for z in candidates],
            all_start_converged=[bool(z.success) for z in candidates],
            weight_decay=float(weight_decay), unpenalized_objective=unpenalized,
            penalty_value=float(.5*weight_decay*penalized_norm_squared),
            penalized_parameter_norm_squared=penalized_norm_squared,
            parameter_norm_squared=float(self.theta@self.theta),
            mean_squared_error=2*unpenalized, training_size=len(y),
            width=self.width, parameters=self.dimension, output_lower=.01, output_upper=.99,
            global_optimization_certified=False)
        return self

    def predict(self, x):
        return self.forward(self.theta, x)[0]
