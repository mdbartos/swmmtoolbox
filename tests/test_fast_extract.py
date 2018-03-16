import pandas as pd
import numpy as np
from swmmtoolbox import swmmtoolbox as swmm

# Get some labels to extract
filename = 'frutal.out'
c = swmm.catalog(filename)
c = pd.DataFrame(c)
# Get nodes only
item_names = c[c[0] == 'node'][1].values
items = pd.Series(np.repeat(item_names, 3)).astype(str)
# Get depth head and inflow
var_indices = pd.Series(np.tile([0, 1, 4], len(item_names))).astype(str)
item_types = pd.Series(np.repeat('node', len(items))).astype(str)
# Construct label strings
labels = (item_types + ',' + items + ',' + var_indices).values.tolist()

# Extract data
result = swmm.fast_extract(filename, *labels)
