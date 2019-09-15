import pandas as pd
from src.hyperparameter_analyses.bayesian_optimization import Space


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
        return result

    return objective


def thesis_search_space():
    search_space = {
        'num_trees': Space(scope=[100, 800], granularity=6, rounding=1),
        'learning_rate': Space(scope=[-2.5, -0.5], granularity=10, scale='log', rounding=13),
        'max_depth': Space(scope=[5, 20], granularity=8, rounding=0),
        'min_child_weight': Space(scope=[5, 40], granularity=3, rounding=1),
        'subsample': Space(scope=[0.5, 1.0], granularity=3, rounding=2)
    }
    return search_space
