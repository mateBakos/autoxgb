import numpy as np
from hyperopt import hp, tpe, fmin,  STATUS_OK, Trials


class Space:
    """Space in one dimension"""
    def __init__(self, scope, scale='linear', granularity=None, rounding=None):
        """Initializes a one-dimensional search space"""
        self._scope = scope
        self._scale = scale
        self._granularity = granularity
        self._rounding = rounding

    @property
    def scope(self):
        return self._scope

    @property
    def granularity(self):
        return self._granularity


class BayesianOptimization:
    """Bayesian optimization"""

    def __init__(self, search_space, objective, max_evals, algo):
        """Initializes Bayesian optimization instance."""
        self._search_space = search_space
        self._objective = objective
        self._max_evals = max_evals
        self._algo = algo

    def get_numpy_space(self):
        space = self._search_space
        real_space = {}
        for param in space.keys():
            if space[param]._scale == 'linear':
                real_space[param] = np.linspace(space[param].scope[0], space[param].scope[1], space[param].granularity)
            if space[param]._scale == 'log':
                real_space[param] = np.logspace(space[param].scope[0], space[param].scope[1], space[param].granularity)
            # if space[param]._rounding:
            real_space[param] = np.round(real_space[param], space[param]._rounding)
        return real_space

    def hyperopt_objective(self, unit_params):

        # get real space
        real_space = self.get_numpy_space()
        real_params = {key: real_space[key][int(unit_params[key]-1)] for key in self._search_space.keys()}

        # perform evaluation
        result = self._objective(real_params).squeeze()

        return {'loss': result, 'status': STATUS_OK}

    def run_analysis(self):
        """Runs the Bayesian optimization."""

        # Create trials object to store information on optimization process
        trials = Trials()

        # Create the hyperopt format arguments
        hyperopt_space = {
            key: hp.quniform(key, 1, self._search_space[key].granularity, 1) for key in list(self._search_space.keys())
        }
        hyperopt_evals = self._max_evals # -len(warmstart_configs)

        # Run the hyperopt optimization
        best = fmin(
            fn=self.hyperopt_objective,
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=hyperopt_evals,
            trials=trials
        )

        return trials