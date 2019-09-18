import numpy as np
import pandas as pd
from hyperopt import hp, tpe, fmin,  STATUS_OK, Trials, rand


class Config:
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


class BayesianHopt:
    """Bayesian hyperparameter optimization"""

    def __init__(self, identifier, search_space, objective, max_evals, algo):
        """Initializes Bayesian hyperparameter optimization instance."""
        self._identifier = identifier
        self._search_space = search_space
        self._objective = objective
        self._max_evals = max_evals
        if algo == 'tpe':
            self._algo = tpe.suggest
        elif algo == 'random':
            self._algo = rand.suggest
        self.results = None

    @property
    def identifier(self):
        return self._identifier

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
        result, walltime, crossval = self._objective(real_params)

        return {'loss': result, 'status': STATUS_OK, 'walltime': walltime, 'crossval': crossval, 'params': real_params}

    def run_bayesian_hopt(self, show_progressbar=True):
        """Runs the Bayesian hyperparameter optimization."""

        # Create trials object to store information on optimization process
        trials = Trials()

        # Create the hyperopt format arguments
        space = self._search_space
        hyperopt_space = {key: hp.quniform(key, 1, space[key].granularity, 1) for key in list(space.keys())}
        hyperopt_evals = self._max_evals # -len(warmstart_configs)

        # Run the hyperopt optimization
        fmin(
            fn=self.hyperopt_objective,
            space=hyperopt_space,
            algo=self._algo,
            max_evals=hyperopt_evals,
            trials=trials,
            show_progressbar=show_progressbar
        )

        results = pd.DataFrame()
        for trial in trials.trials:
            result = trial['result']
            params = result.pop('params')
            result = pd.concat([pd.Series(result), pd.Series(params)], keys=['results', 'configs'])
            results = results.append(result, ignore_index=True)
        results.columns = pd.MultiIndex.from_tuples(results.columns)

        # add results as attribute to the instance
        self.results = results

        return results
