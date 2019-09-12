import pandas as pd
from hyperopt import STATUS_OK


def create_lookup_objective(name):

    def objective(params):
        # import lookup table
        lookup_table = pd.read_csv('../../data/metadata/raw/' + name + '.csv', index_col=0, header=[0, 1])
        # get row index
        idx = lookup_table.index[
            (lookup_table['hyperparameters']['max_depth'] == params['max_depth']) &
            (lookup_table['hyperparameters']['learning_rate'] == params['learning_rate']) &
            (lookup_table['hyperparameters']['min_child_weight'] == params['min_child_weight']) &
            (lookup_table['hyperparameters']['subsample'] == params['subsample']) &
            (lookup_table['hyperparameters']['num_trees'] == params['num_trees'])
            ]
        result = lookup_table.iloc[idx]['diagnostics']['mae'].squeeze()

        return {'loss': result, 'status': STATUS_OK}

    return objective
