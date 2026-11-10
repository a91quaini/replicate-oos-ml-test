"""One/two hidden layer extension of the archived bounded probability network."""
import numpy as np
from network import Network
from selection import KAPPAS, split_record, independent_seed_copy


class MultilayerNetwork(Network):
    def __init__(self, inputs, architecture):
        self.inputs = int(inputs)
        self.architecture = tuple(int(w) for w in architecture)
        if not self.architecture or min(self.architecture) < 1:
            raise ValueError('Positive hidden layer widths required')
        self.width = self.architecture[-1]
        self.shapes = [(a+1, b) for a, b in
                       zip((self.inputs,)+self.architecture[:-1], self.architecture)]
        self.dimension = sum(a*b for a,b in self.shapes)+self.width+self.inputs+1

    def forward(self, theta, x):
        d = np.column_stack((np.ones(len(x)), x))
        layers, position, h = [], 0, x
        for shape in self.shapes:
            matrix = theta[position:position+np.prod(shape)].reshape(shape)
            position += np.prod(shape)
            augmented = np.column_stack((np.ones(len(x)), h))
            h = np.tanh(augmented @ matrix)
            layers.append((augmented, matrix, h))
        a, b = theta[position:position+self.width], theta[position+self.width:]
        z = np.tanh(h @ a / np.sqrt(self.width) + d @ b)
        return .5+.49*z, (d, layers, a, z)

    def objective(self, theta, x, y, weight_decay=0.):
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('Invalid weight decay')
        pred, (d, layers, a, z) = self.forward(theta, x)
        residual = pred-y
        g = .49*residual*(1-z*z)/len(y)
        ga = layers[-1][2].T @ g / np.sqrt(self.width)
        gb = d.T @ g
        upstream = g[:, None]*a[None, :]/np.sqrt(self.width)
        gradients = []
        for augmented, matrix, h in reversed(layers):
            local = upstream*(1-h*h)
            gradients.append((augmented.T @ local).ravel())
            upstream = local @ matrix[1:].T
        gradient = np.concatenate(list(reversed(gradients))+[ga, gb])
        value = float(.5*np.mean(residual**2))
        if weight_decay:
            penalized = self.penalty_vector(theta)
            value += .5*weight_decay*float(penalized @ penalized)
            gradient += weight_decay*penalized
        return value, gradient


def new_network(inputs, architecture):
    # Preserve the exact archived arithmetic for the one-layer specifications.
    return (Network(inputs, architecture[0]) if len(architecture)==1
            else MultilayerNetwork(inputs, architecture))


def fit_network(x, y, architecture, regularization, seed, gap, maxiter=600, starts=2):
    def fit(xx, yy, strength):
        return new_network(x.shape[1], architecture).fit(
            xx, yy, independent_seed_copy(seed), maxiter=maxiter,
            starts=starts, weight_decay=strength)
    if regularization == 'none':
        final = fit(x, y, 0.)
        final.diagnostics['selection'] = dict(mode='none', selected_kappa=0.,
            selected_lambda=0., candidates=[], split=None)
    elif regularization == 'validation':
        split = split_record(len(y), gap)
        prefix, validation = slice(0, split['fit_stop']), slice(split['validation_start'], len(y))
        candidates = []
        for kappa in KAPPAS:
            net = fit(x[prefix], y[prefix], kappa/len(y))
            candidates.append(dict(kappa=kappa, weight_decay=kappa/len(y),
                validation_mse=float(np.mean((net.predict(x[validation])-y[validation])**2)),
                converged=net.diagnostics['converged'],
                all_starts_converged=all(net.diagnostics['all_start_converged'])))
        minimum = min(c['validation_mse'] for c in candidates)
        chosen = max((c for c in candidates if np.isclose(c['validation_mse'], minimum,
                     rtol=1e-10, atol=1e-12)), key=lambda c:c['kappa'])
        final = fit(x, y, chosen['weight_decay'])
        final.diagnostics['selection'] = dict(mode='chronological_validation', split=split,
            selected_kappa=chosen['kappa'], selected_lambda=chosen['weight_decay'],
            candidates=candidates)
    else:
        raise ValueError('Unknown regularization')
    final.diagnostics['architecture'] = list(architecture)
    return final
