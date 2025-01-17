import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_search_performance(results, xaxis='iterations', all_losses=False, crossvalidation=False):
    # create figure
    fig = go.Figure()

    # define x-axis
    if xaxis == 'iterations':
        idx = results.index
    if xaxis == 'walltime':
        idx = pd.Series([results.loc[:i, ('results', 'walltime')].sum() for i in range(len(results))])

    # compute best-so-far series
    rolling_min = pd.Series([results.loc[:i, ('results', 'loss')].min() for i in range(len(results))])
    fig.add_trace(go.Scatter(x=idx, y=rolling_min, mode='lines', name='Best so far'))

    # optionally show all computed evaluations
    if all_losses:
        loss = results.loc[:, ('results', 'loss')]
        fig.add_trace(go.Scatter(x=idx, y=loss, mode='markers', name='Iteration result'))

    # optionally show the crossvalidation performance
    if crossvalidation:
        rolling_crossval = pd.Series([results.loc[:i, ('results', 'crossval')].min() for i in range(len(results))])
        fig.add_trace(go.Scatter(x=idx, y=rolling_crossval, mode='lines', name='Crossvalidation so far'))

    # optionally show all computed evaluations of crossvalidation performance
    if all_losses & crossvalidation:
        crossval = results.loc[:, ('results', 'crossval')]
        fig.add_trace(go.Scatter(x=idx, y=crossval, mode='markers', name='Crossvalidation result'))

    fig.show()


def compare_search_performance(results_dict, xaxis='iterations'):
    # create figure
    fig = go.Figure()

    for analysis in results_dict.keys():
        # define x-axis
        if xaxis == 'iterations':
            idx = results_dict[analysis].index
        if xaxis == 'walltime':
            idx = pd.Series([results_dict[analysis].loc[:i, ('results', 'walltime')].sum() for i in range(len(results_dict[analysis]))])

        # compute best-so-far series
        rolling_min = pd.Series([results_dict[analysis].loc[:i, ('results', 'loss')].min() for i in range(len(results_dict[analysis]))])
        fig.add_trace(go.Scatter(x=idx, y=rolling_min, mode='lines', name=analysis))

    fig.show()


def visualize_hopt_exp_results(hopt_exp_results, show='mean'):

    fig = go.Figure()

    # transform to best so far dataframe
    form = hopt_exp_results.unstack(0).unstack(1)
    best_lists = [[form.iloc[j][:i + 1].min() for i in form.columns] for j in range(len(form))]
    best_so_far = pd.DataFrame(data=best_lists, columns=form.columns, index=form.index).stack(0).unstack([0, 2])

    if show == 'mean':
        data = best_so_far.mean().unstack(0)

    if show == 'rank':
        data = best_so_far.stack(1).rank(axis=1).mean(level='iterations')

    for identifier in hopt_exp_results.columns.levels[0].values:
        fig.add_trace(go.Scatter(y=data[identifier], name=identifier))

    fig.update_layout(
        xaxis=go.layout.XAxis(title='Iterations'),
        yaxis=go.layout.YAxis(title=show)
    )

    fig.show()


def visualize_hopt_exp_perf_distribution(hopt_exp_results, iterations):

    fig = go.Figure()

    data = hopt_exp_results.stack(0)[iterations].unstack()

    for identifier in hopt_exp_results.columns.levels[0]:
        fig.add_trace(go.Box(
            y=data[identifier],
            name=identifier,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker_size=3,
            line_width=1
        ))

    fig.update_layout(yaxis=go.layout.YAxis(title='MAE'), showlegend=False)

    fig.show()


def visualize_search_performance_heatmap(hopt_exp_results):

    identifiers = hopt_exp_results.columns.levels[0]

    fig = make_subplots(rows=1, cols=len(identifiers), subplot_titles=identifiers)

    for j, identifier in enumerate(identifiers):
        data = hopt_exp_results[identifier]
        x = list(data.columns.values) * len(data.index)
        y = data.values.flatten()
        fig.add_trace(go.Histogram2dContour(x=x, y=y, name=identifier), row=1, col=j + 1)

    fig.update_yaxes(title_text="Mean squared error")
    fig.update_xaxes(title_text="Iterations")

    fig.show()
