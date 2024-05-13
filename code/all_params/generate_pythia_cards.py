import pandas as pd
import os
base_dir="/afs/cern.ch/work/a/aalkadhi/public/Pythia_SBI_Tune"
filename= os.path.join(base_dir,"all_params_sample_25k.csv")

orig_datacard = os.path.join(base_dir,"template_run","ALEPH_1996_S3486095_card.cmnd")
df = pd.read_csv(filename)

for rowind, row in df.iterrows():
    
    os.system('cp %s PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (orig_datacard, rowind) )
    
    os.system("sed -i 's;Random:seed=0;Random:seed=%s;g' PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd" % (rowind+1, rowind))
    
    os.system('echo StringZ:aLund = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringZ:aLund'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringZ:bLund = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringZ:bLund'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringZ:rFactC = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringZ:rFactC'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringZ:rFactB = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringZ:rFactB'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringZ:aExtraSQuark = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringZ:aExtraSQuark'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringZ:aExtraDiquark = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringZ:aExtraDiquark'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringPT:sigma = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringPT:sigma'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringPT:enhancedFraction = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringPT:enhancedFraction'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringPT:enhancedWidth = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringPT:enhancedWidth'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringFlav:ProbStoUD = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringFlav:ProbStoUD'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringFlav:probQQtoQ = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringFlav:probQQtoQ'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringFlav:probSQtoQQ = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringFlav:probSQtoQQ'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo StringFlav:ProbQQ1toQQ0 = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['StringFlav:ProbQQ1toQQ0'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo TimeShower:alphaSvalue = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['TimeShower:alphaSvalue'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    os.system('echo TimeShower:pTmin = %s >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % (row['TimeShower:pTmin'], rowind) )
    os.system('echo >> PYTHIA_CARDS_25K/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    
    