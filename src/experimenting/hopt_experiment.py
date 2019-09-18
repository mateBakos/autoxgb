class HoptExperiment:
    def __init__(self, hopts, iterations=1):
        self._hopts = hopts
        self._iterations = iterations
        self.results = None

    def run_hopt_experiment(self):

        results = [[hopt.run_bayesian_hopt() for hopt in self._hopts] for i in range(self._iterations)]

        return results
