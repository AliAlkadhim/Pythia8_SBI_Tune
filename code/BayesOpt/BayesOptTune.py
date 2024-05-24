import os, sys, re
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import importlib

from bayes_opt import BayesianOptimization, UtilityFunction 


from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render


# init_notebook_plotting()
# import plotly.io as pio
# pio.renderers.default = "jupyterlab"


#`source /cvmfs/sft.cern.ch/lcg/views/LCG_102/x86_64-centos7-gcc11-opt/setup.sh`
from glob import glob
from tqdm import tqdm
from yoda2numpy_BayesOpt import Yoda2Numpy

from pythia_SBI_utils import *


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


BAYES_OPT=False
AX=True


def make_pythia_card(aLund, bLund):
    
    cards_dir = os.path.join(os.getcwd(), "BO_Cards")
    filename = f"ALEPH_1996_S3486095_BO_card.cmnd"
    file_path = os.path.join(cards_dir, filename)
    with open(file_path,'w') as f:
        first_block="""Main:numberOfEvents = 100000          ! number of events to generate
Next:numberShowEvent = 0           ! suppress full listing of first events
# random seed
Random:setSeed = on
Random:seed= 0
! 2) Beam parameter settings.
Beams:idA = 11                ! first beam,  e- = 11
Beams:idB = -11                ! second beam, e+ = -11
Beams:eCM = 91.2               ! CM energy of collision
# Pythia 8 settings for LEP
# Hadronic decays including b quarks, with ISR photons switched off
WeakSingleBoson:ffbar2gmZ = on
23:onMode = off
23:onIfAny = 1 2 3 4 5
PDF:lepton = off
SpaceShower:QEDshowerByL = off\n\n"""
        f.write(first_block)
        # f.write(f"Random:seed={indx+1}")
        f.write(f"StringZ:aLund = {aLund}\n\n")
        f.write(f"StringZ:bLund = {bLund}\n\n")

PARAM_DICT = {
        'StringZ:aLund' : [0.0, 2.0],
        'StringZ:bLund': [0.2, 2.0],
        # 'StringZ:rFactC':[0.0, 1.994052.0],
        # 'StringZ:rFactB': [0., 2.0],
        # 'StringZ:aExtraSQuark':[0.,2],
        # 'StringZ:aExtraDiquark':[0.,2.],
        # 'StringPT:sigma':[0.,1.],
        # 'StringPT:enhancedFraction':[0.,1.],
        # 'StringPT:enhancedWidth':[1.0,4.0],
        # 'StringFlav:ProbStoUD':[0,4.0],
        # 'StringFlav:probQQtoQ':[0,4.0],
        # 'StringFlav:probSQtoQQ':[0,4.0],
        # 'StringFlav:ProbQQ1toQQ0':[0,4.0],
        # 'TimeShower:alphaSvalue':[0.06,0.25],
        # 'TimeShower:pTmin':[0.1,2.0]


}


def get_pbounds(PARAM_DICT):
    pbounds = {}
    for key, value in PARAM_DICT.items():
        p_name = key.split(':')[1]
        p_bound = tuple(value)
        pbounds[p_name] = p_bound
    return pbounds

def true_objective_func(aLund, bLund):
    
    # step 1: write .cmnd file 
    make_pythia_card(aLund, bLund)
    #step 2 run main42 and rivet
    os.system("""./main42 BO_Cards/ALEPH_1996_S3486095_BO_card.cmnd ALEPH_1996_S3486095_card.fifo &
    
    rivet -o ALEPH_1996_S3486095_hist_0.yoda -a ALEPH_1996_S3486095 ALEPH_1996_S3486095_card.fifo

    rm ALEPH_1996_S3486095_card.fifo
    mv ALEPH_1996_S3486095_hist_0.yoda ALEPH_YODAS_BayesOpt/""")
    
    
    dfdata, dfsims, generated_indices = get_data()
    print('DATA DATAFRAME')
    print(dfdata['/REF/ALEPH_1996_S3486095/d01-x01-y01'].head())
    print('FIRST SIM DATAFRAME')
    print(dfsims[generated_indices[0]]['/ALEPH_1996_S3486095/d01-x01-y01'].head())
    
    data_keys, mc_keys = get_hist_names(dfdata)

    filtered_data_keys, filtered_mc_keys = filter_keys(dfdata, dfsims, data_keys, mc_keys)
    X0 = {}
    for ii, gen_ind in enumerate(generated_indices):
        # X0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[gen_ind], which = 0))
        # try:
        #     X0.append(test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[ii], which = 0))
        try:
            X0[gen_ind] = test_statistic(filtered_data_keys,filtered_mc_keys, dfdata, dfsims[gen_ind], which = 0)
        except Exception:
            print('test statistic error in file index: ', gen_ind)
            
    if BAYES_OPT:        
        objective_func = - X0[0]
    else:
        objective_func = X0[0]
    os.system("rm ALEPH_YODAS_BayesOpt/ALEPH_1996_S3486095_hist_0.yoda")
        
    print(f"objective function = {objective_func}")
    return objective_func
        
    #step 3: copy the output yoda file to the ALEPH_YODAS_BayesOpt/
    #step 4: calculate the chi2 between the simulation at this point and the data
if __name__=='__main__':
    # make_pythia_card(1,2,1)
    # true_objective_func(aLund=1, bLund=2)
    if BAYES_OPT:
        PBOUNDS = get_pbounds(PARAM_DICT)
        print(PBOUNDS)
        optimizer = BayesianOptimization(
            f=true_objective_func,
            pbounds=PBOUNDS,
            verbose=2, 
            random_state=1
        )
        # kind: {'ucb', 'ei', 'poi'}
        #     * 'ucb' stands for the Upper Confidence Bounds method
        #     * 'ei' is the Expected Improvement method
        #     * 'poi' is the Probability Of Improvement criterion.
            
        # acquisition_function = UtilityFunction(kind='ucb',
        #                         kappa=2.576,
        #                         xi=0.0,
        #                         kappa_decay=1,
        #                         kappa_decay_delay=0)
        
        # acquisition_function = UtilityFunction(kind='poi')
        acquisition_function = UtilityFunction(kind='ei')
        
        optimizer.maximize(
            init_points=16,
            n_iter=6,
            acquisition_function=acquisition_function
        )
        print('BEST PARAMETERS', optimizer.max)
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \t{}".format(i, res))
            
            
    if AX:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="Ax_Tune_Pythia",
            parameters = [
                {
                    "name": "aLund",
                    "type": "range",
                    "bounds": [0.0, 2.0],
                }, 
                {
                    "name": "bLund",
                    "type": "range",
                    "bounds": [0.2, 2.0],
                },
            ],
            objectives = {"true_objective_func": ObjectiveProperties(minimize=True)},
        )
        
        N_ITER = 45
        for i in range(N_ITER):
            parameterization, trial_index = ax_client.get_next_trial()
            print(parameterization)
            ax_client.complete_trial(trial_index=trial_index, raw_data=true_objective_func(
                aLund=parameterization["aLund"], bLund=parameterization["bLund"]))
        
        
        best_parameters, values = ax_client.get_best_parameters()

        print("BEST PARAMETERS: ", best_parameters)
        
        # render(ax_client.get_contour_plot())
    