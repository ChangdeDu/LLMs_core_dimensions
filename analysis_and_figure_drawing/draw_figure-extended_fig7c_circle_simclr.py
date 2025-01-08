import pandas as pd
import numpy as np
import plotly.express as px

data = pd.read_csv('data/DNNs/simclr/dim_naming_simclr/artificial_dimension_naming_for_simclr.csv')
fig = px.sunburst(
    data,
    path=[ 'Name1',  'Name2',  'Name3'],
    # title='Sunburst Plot for Categorical Dataset',
    width=500,  # Set the width
    height=500  # Set the height
)


fig.write_image('figures/sunburst_simclr.pdf')
