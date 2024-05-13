import pandas as pd
import numpy as np

filename = 'all_params_sample_25k.csv'



N=25000
df = pd.DataFrame()

amin, amax = 0.0, 2.0
df['StringZ:aLund'] = np.random.uniform(amin, amax, N)

bmin,bmax  = 0.2, 2.0
df['StringZ:bLund'] = np.random.uniform(bmin, bmax, N)

rFactCmin,rFactCmax  = 0.0, 2.0
df['StringZ:rFactC'] = np.random.uniform(rFactCmin, rFactCmax, N)

rFactBmin, rFactBmax = 0., 2.0
df['StringZ:rFactB'] = np.random.uniform(rFactBmin, rFactBmax, N)

df['StringZ:aExtraSQuark'] = np.random.uniform(0.,2.,N)
df['StringZ:aExtraDiquark']=np.random.uniform(0.,2.,N)
df['StringPT:sigma']=np.random.uniform(0.,1.,N)
df['StringPT:enhancedFraction']=np.random.uniform(0.,1.,N)
df['StringPT:enhancedWidth']=np.random.uniform(1.0,10.0,N)
df['StringFlav:ProbStoUD']=np.random.uniform(0.,10.,N)
df['StringFlav:probQQtoQ'] = np.random.uniform(0.,10.,N)
df['StringFlav:probSQtoQQ']=np.random.uniform(0.,10.,N)
df['StringFlav:ProbQQ1toQQ0']=np.random.uniform(0.,10.,N)
df['TimeShower:alphaSvalue']=np.random.uniform(0.06,0.25,N)
df['TimeShower:pTmin']=np.random.uniform(0.1,2.0,N)





df.to_csv(filename)