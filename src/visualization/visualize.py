import pandas as pd
import plotly.graph_objects as go


def visualize_search_performance(results):
    # compute results
    idx = results.index
    rolling_min = pd.Series([results.loc[:i,('results','loss')].min() for i in range(len(results))])
    loss = results.loc[:,('results','loss')]

    # show in figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx,y=rolling_min,mode='lines',name='Best so far'))
    fig.add_trace(go.Scatter(x=idx,y=loss,mode='markers',name='Iteration result'))
    fig.show()