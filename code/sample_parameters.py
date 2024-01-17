import pandas as pd
import numpy as np

filename = 'a_b_samples_uniform_2.csv'

amin, amax = 0.0, 2.0
bmin,bmax  = 0.2, 2.0
N=1000
df = pd.DataFrame()
df['a'] = np.random.uniform(amin, amax, N)
df['b'] = np.random.uniform(bmin, bmax, N)
df.to_csv(filename)