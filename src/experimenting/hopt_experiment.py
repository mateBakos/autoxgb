import pandas as pd
from tqdm import tqdm


class HoptExperiment:
    def __init__(self, hopts, iterations=1):
        self._hopts = hopts
        self._iterations = iterations
        self.results = None

    def run_hopt_experiment(self):

        results = [[hopt.run_bayesian_hopt(show_progressbar=False) for i in tqdm(range(self._iterations), desc=hopt.identifier + ' duplicates')] for hopt in self._hopts]

        df = [item['results']['loss'] for sublist in results for item in sublist]
        indices = pd.MultiIndex.from_product(
            iterables=[[hopt.identifier for hopt in self._hopts], range(self._iterations)],
            names=['hopt', 'duplicate_nr']
        )
        results = pd.DataFrame(df, index=indices).stack().unstack(1).transpose()
        results = results.rename_axis(columns=['hopt', 'iterations'])

        return results
