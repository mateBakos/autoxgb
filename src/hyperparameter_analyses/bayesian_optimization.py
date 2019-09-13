import numpy as np
import pandas as pd
from hyperopt import hp, tpe, fmin,  STATUS_OK, rand, Trials


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

    def hyperopt_lookup_objective(self, name):
        def objective(params):
            # real search space
            real_space = {key: np.linspace(
                self._search_space[key].scope[0],
                self._search_space[key].scope[1],
                self._search_space[key].granularity) for key in self._search_space.keys()}
            real_params = {key: real_space[key][params[key]] for key in self._search_space.keys()}
            # import lookup table
            lookup_table = pd.read_csv('../../data/metadata/raw/' + name + '.csv', index_col=0, header=[0, 1])
            # get row index
            idx = lookup_table.index[
                (lookup_table['hyperparameters']['max_depth'] == real_params['max_depth']) &
                (lookup_table['hyperparameters']['learning_rate'] == real_params['learning_rate']) &
                (lookup_table['hyperparameters']['min_child_weight'] == real_params['min_child_weight']) &
                (lookup_table['hyperparameters']['subsample'] == real_params['subsample']) &
                (lookup_table['hyperparameters']['num_trees'] == real_params['num_trees'])
                ]
            result = lookup_table.iloc[idx]['diagnostics']['mae'].squeeze()

            return {'loss': result, 'status': STATUS_OK}

        return objective

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
            fn=self._objective,
            space=hyperopt_space,
            algo=tpe.suggest,
            max_evals=hyperopt_evals,
            trials=trials
        )
