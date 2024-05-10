import os, sys, re
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import importlib
from joblib import  Memory

#source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh
from glob import glob
from tqdm import tqdm
from yoda2numpy_all import Yoda2Numpy

# update fonts
FONTSIZE = 14
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=True)


memory = Memory('/afs/cern.ch/work/a/aalkadhi/private/TUNES/Pythia8_SBI_Tune/cluster')


# @memory.cache
def get_data():
    yoda2numpy = Yoda2Numpy()
    files = list(glob('ALEPH_YODAS/*.yoda'))
    # M = len(files)
    M = 1000
    generated_indices = []
    for file in files[:M]:
        index = file.split('_')[-1].split('.')[0]
        generated_indices.append(int(index))    
    generated_indices.sort()
    print(generated_indices)
    # # --- SIM
    print(f'looping over {M:d} sim yoda files...\n')
    dfsims = []
    for ii in tqdm(generated_indices):    
        # index here should match the index of the file
        dfsims.append( yoda2numpy.todf( yoda2numpy('sim', index=ii) ) )

    # # --- NEW
    # print(f'looping over {M:d} new yoda files...\n')
    # # dfnews = []
    # # for ii in tqdm(range(M)):
    # #     dfnews.append( yoda2numpy.todf( yoda2numpy('new', index=ii) ) )

    print()
    # key = '/ALEPH_1996_S3486095/d01-x01-y01'
    # dfsim = dfsims[0][key]
    
    dfdata = yoda2numpy.todf( yoda2numpy('dat') )
    
    return dfdata, dfsims, generated_indices


def get_hist_names(dfdata):
    data_keys = list(dfdata.keys())
    mc_keys = [data_key[4:] for data_key in data_keys]
    
    return data_keys, mc_keys


def filter_keys(dfdata, dfsims, data_keys, mc_keys):
    new_data_keys = []
    new_mc_keys = []
    for data_key, mc_key in zip(data_keys, mc_keys):
        dfdat = dfdata[data_key]
        yval = dfdat['yval']
        yerr = dfdat['yerr-']
        
        if len(yval) < 2: continue
        if not (mc_key in dfsims[0]): continue
        
        new_data_keys.append(data_key)
        new_mc_keys.append(mc_key)
        
    return new_data_keys, new_mc_keys


def test_statistic(data_keys, mc_keys, dfdata, dfpred, which = 0):
    Y=0.0
    nbins=0
    # if which=0, we are using dfdata=dfdata, 
    #if which =1 we are using dfdata=dfbest or dfdata=dfsim
    
    for data_key, mc_key in zip(data_keys, mc_keys):
        if which == 0:
            df = dfdata[data_key]
            data = df['yval']
            data_err = df['yerr-']
        else:
            df = dfdata[mc_key]
            data = df['sumw']
            data_err = np.sqrt(df['sumw2'])
            
        ndat = len(df)
        pred_df = dfpred[mc_key]
        npred = len(pred_df)
        
        assert ndat == npred
        
        pred = pred_df['sumw']
        pred_err2 = pred_df['sumw2']
        
        stdv = np.sqrt(data_err**2 + pred_err2)
        stdv = np.where(stdv < 1e-3, 1, stdv)
        
        X = (((data-pred)/stdv)**2).sum()
        Y += X
        nbins += ndat
        
    return np.sqrt(Y/nbins)


if __name__ == '__main__':
    dfdata, dfsims, generated_indices = get_data()
    print(generated_indices)
    # print(dfsims)
    print('DATA DATAFRAME')
    print(dfdata['/REF/ALEPH_1996_S3486095/d01-x01-y01'].head())
    print('FIRST SIM DATAFRAME')
    print(dfsims[0]['/ALEPH_1996_S3486095/d01-x01-y01'].head())
    
    data_keys, mc_keys = get_hist_names(dfdata)
    filtered_data_keys, filtered_mc_keys = filter_keys(dfdata, dfsims, data_keys, mc_keys)
    
    
    X0 = []
    for ii in range(0,len(generated_indices)):
        X0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[ii], which = 0))
    X0=np.array(X0)
    
    df_all_params = pd.read_csv('all_params_sample_25k.csv')
    df_generated_indices = np.array(generated_indices) -1
    df_all_params=df_all_params.iloc[df_generated_indices].reset_index(drop=True)
    print(len(df_all_params))
    print(df_all_params.head())
    
    K_best = np.argmin(X0)   # file with minimum value of test statistic
    dfbest = dfsims[K_best]
    
    a_best = df_all_params['StringZ:aLund'].iloc[K_best]
    b_best = df_all_params['StringZ:bLund'].iloc[K_best]
    
    print(f'a best= {a_best}, b best = {b_best}')
    