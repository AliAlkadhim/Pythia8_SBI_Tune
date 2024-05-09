import os, sys, re
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import importlib
from joblib import  Memory


from glob import glob
from tqdm import tqdm
from yoda2numpy import Yoda2Numpy

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


memory = Memory('/home/ali/Downloads')

@memory.cache
def get_data():
    yoda2numpy = Yoda2Numpy()
    files = glob('rivet_histograms/newseeds/*.yoda')
    M = len(files)
    # M =3
    # --- SIM
    print(f'looping over {M:d} sim yoda files...\n')
    dfsims = []
    for ii in tqdm(range(M)):    
        dfsims.append( yoda2numpy.todf( yoda2numpy('sim', index=ii) ) )

    # --- NEW
    print(f'looping over {M:d} new yoda files...\n')
    dfnews = []
    for ii in tqdm(range(M)):
        dfnews.append( yoda2numpy.todf( yoda2numpy('new', index=ii) ) )

    print()
    key = '/ALEPH_1996_S3486095/d01-x01-y01'
    dfsim = dfsims[0][key]
    
    dfdata = yoda2numpy.todf( yoda2numpy('dat') )
    
    return dfdata, dfsims, dfnews, M

def get_hist_names(dfdata):
    data_keys = list(dfdata.keys())
    mc_keys = [data_key[4:] for data_key in data_keys]
    
    return data_keys, mc_keys
    

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
    
    
def make_hists(dfdata, dfbest, filtered_data_keys, filtered_mc_keys):
    hists = []
    for data_key, mc_key in zip(filtered_data_keys, filtered_mc_keys):
        data=dfdata[data_key]
        data_val = data['yval']
        # data_err = data['yerr-']
        
        pred = dfbest[mc_key]
        pred_val = pred['sumw']
        # pred_err = np.sqrt(pred_val['sumw2'])
        edges = dfbest[mc_key]['xlow']
        
        hist_name = data_key.split('/')[-1]
        hists.append((hist_name, edges, data_val, pred_val))
    return hists
    
def plt_sim_data_hist(ax, hist):
    name, edges, data_val, pred_val = hist
    
    xmin = 0
    xmax = edges.max()
    ax.set_ylim(xmin, xmax)
    
    ymin = 0
    ymax = 1.25 * data_val.max()
    ax.set_ylim(ymin, ymax)

    ax.text(xmin + 0.7*(xmax-xmin), ymin + 0.2*(ymax-ymin), name)
    
    ax.step(y=data_val, x=edges, label='data')
    ax.step(y=pred_val, x=edges, label='pred')
    ax.legend()
            
def plot_dist(hist_names, hists, filename='fig_bestfit_dist.png'):
    
        
    nhists= len(hist_names)
    ncols = 3
    nrows = nhists // ncols
    nhists= nrows * ncols
    
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 20), edgecolor='k')
    ax = ax.ravel()
    for hist_ind, hist in enumerate(hists[:nhists]):
        plt_sim_data_hist(ax[hist_ind], hist)
    plt.show()
        
    # plt.savefig(filename)   
    
    
def plot_cdf(df, a=None, b=None,
             xbins=10, xmin=0.5, xmax=1.5, 
             ybins=10, ymin=0.5, ymax=1.5, 
             filename='fig_cdf_via_hist.png',
             fgsize=(5, 5), 
             ftsize=18):
    
    # approximate cdf via histogramming
    xrange = (xmin, xmax)
    yrange = (ymin, ymax)
        
    # weighted histogram   (count the number of ones per bin)
    hw, xe, ye = np.histogram2d(df.a, df.b, 
                                bins=(xbins, ybins), 
                                range=(xrange, yrange), 
                                weights=df.Z0)

    # unweighted histogram (count number of ones and zeros per bin)
    hu, _, _ = np.histogram2d(df.a, df.b, 
                              bins=(xbins, ybins), 
                              range=(xrange, yrange)) 
    P =  hw / (hu + 1.e-10)    
  
    # flatten arrays so that p, x, and y are 1d arrays
    # of the same length.    
    # get bin centers
    x   = (xe[1:] + xe[:-1])/2
    y   = (ye[1:] + ye[:-1])/2
    X,Y = np.meshgrid(x, y)
    x   = X.flatten()
    y   = Y.flatten()
    
    # WARNING: must transpose P so that X, Y, and P have the
    # same shape
    P   = P.T
    p   = P.flatten()
                
    # Now make plots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fgsize)
    
    ax.set_xlim(xmin, xmax)
    #ax.set_xticks(np.linspace(xmin, xmax, 6))
    ax.set_xlabel(r'$a$', fontsize=ftsize)
    
    ax.set_ylim(ymin, ymax)
    #ax.set_yticks(np.linspace(xmin, xmax, 9))
    ax.set_ylabel(r'$b$',  fontsize=ftsize)
    
    mylevels = np.array([0.68, 0.8, 0.9])

    colormap = 'rainbow'
    
    cs = ax.contour(X, Y, P, 
                    extent=(xmin, xmax, ymin, ymax),
                    levels=mylevels,
                    linewidths=2,
                    linestyles='dashed',
                    cmap=colormap)

    ax.clabel(cs, cs.levels, 
              inline=True, 
              fontsize=18, fmt='%4.2f', 
              colors='black')

    if a != None:
        if b != None:
            print(f'a(best): {a_best:10.3f} b(best): {b_best:10.3f}')
            ax.plot([a_best], [b_best], 
                    markerfacecolor='red',
                    markersize=20, 
                    marker='.')
            
    ax.grid()

    plt.tight_layout()
    # plt.savefig(filename)    
    plt.show()
    
if __name__ == '__main__':
    dfdata, dfsims, dfnews, M = get_data()
    print('DATA DATAFRAME')
    print(dfdata['/REF/ALEPH_1996_S3486095/d01-x01-y01'].head())
    print('FIRST SIM DATAFRAME')
    print(dfsims[0]['/ALEPH_1996_S3486095/d01-x01-y01'].head())

    data_keys, mc_keys = get_hist_names(dfdata)
    filtered_data_keys, filtered_mc_keys = filter_keys(dfdata, dfsims, data_keys, mc_keys)

    X0 = []
    for ii in range(M):
        X0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfnews[ii], which = 0))
    X0=np.array(X0)
    
    df_ab = pd.read_csv('a_b_samples_uniform_1.csv')[:M]
    K_best = np.argmin(X0)   # file with minimum value of test statistic
    dfbest = dfnews[K_best]

    a_best = df_ab.a.iloc[K_best]
    b_best = df_ab.b.iloc[K_best]
    
    hists = make_hists(dfdata, dfbest, filtered_data_keys, filtered_mc_keys)
    
    # plot_dist(filtered_data_keys, hists, filename='fig_bestfit_dist.png')
    
    # randomly shuffle integers 0 to M-1
    jj = np.array(list(range(M)))
    np.random.shuffle(jj)
    # 1. test statistic with "observed" data
    l0 = []
    for ii in range(M):
        l0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata=dfbest, dfpred=dfnews[ii], which = 1))
    l0=np.array(l0)
    
    # 2. test statistic with simulated "observed" data
    lp = []
    for ii in range(M):
        lp.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata=dfsims[jj[ii]], dfpred=dfnews[ii], which = 1))
    lp=np.array(lp)
    # 3. test statistic with simulated data and predictions at the same parameter point
    l = []
    for ii in range(M):
        l.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata=dfsims[ii], dfpred=dfnews[ii], which = 1))
    l=np.array(l)

    df_ab['l0'] = l0
    df_ab['lp'] = lp
    df_ab['l']  = l
    df_ab['Z0'] = (l <= l0).astype(int)
    df_ab['Zp'] = (l <= lp).astype(int)
    
    print(df_ab.head())

    plot_cdf(df_ab, a_best, b_best)