import pandas as pd
from src.pipeline_optimization.bayesian_hopt import Config


def thesis_lookup_objective(name):

    def objective(params):
        # import lookup table
        lookup_table = pd.read_csv('../../data/metadata/raw/' + name + '.csv', index_col=0, header=[0, 1])
        lookup_table.loc[:, ('hyperparameters', 'learning_rate')] = lookup_table['hyperparameters']['learning_rate'].round(13)

        idx = lookup_table.index[
            (lookup_table['hyperparameters']['max_depth'] == params['max_depth']) &
            (lookup_table['hyperparameters']['learning_rate'] == params['learning_rate']) &
            (lookup_table['hyperparameters']['min_child_weight'] == params['min_child_weight']) &
            (lookup_table['hyperparameters']['subsample'] == params['subsample']) &
            (lookup_table['hyperparameters']['num_trees'] == params['num_trees'])
        ]
        result = lookup_table.iloc[idx]['diagnostics']['mae'].squeeze()
        walltime = lookup_table.iloc[idx]['diagnostics']['walltime'].squeeze()
        crossval = lookup_table.iloc[idx]['crossval_diag']['mae'].squeeze()

        return result, walltime, crossval

    return objective


def thesis_search_space():
    search_space = {
        'num_trees': Config(scope=[100, 800], granularity=6, rounding=1),
        'learning_rate': Config(scope=[-2.5, -0.5], granularity=10, scale='log', rounding=13),
        'max_depth': Config(scope=[5, 20], granularity=8, rounding=0),
        'min_child_weight': Config(scope=[5, 40], granularity=3, rounding=1),
        'subsample': Config(scope=[0.5, 1.0], granularity=3, rounding=2)
    }
    return search_space
