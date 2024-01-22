import pandas as pd
import os
filename="a_b_samples_uniform_1.csv"
orig_datacard = 'ALEPH_1996_S3486095_Cards/Aplanarity.cmnd'
df = pd.read_csv(filename)
for rowind, row in df.iterrows():
    print(f"a={row['a']}", '\t', f"b={row['b']}")
    
    os.system('cp %s ALEPH_1996_S3486095_Cards/ALEPH_1996_S3486095_card_%s.cmnd' % (orig_datacard, rowind) )
    os.system('echo StringZ:aLund = %s >> ALEPH_1996_S3486095_Cards/ALEPH_1996_S3486095_card_%s.cmnd' % (row['a'], rowind) )
    
    os.system('echo >> ALEPH_1996_S3486095_Cards/ALEPH_1996_S3486095_card_%s.cmnd' % rowind)
    os.system('echo StringZ:bLund = %s >> ALEPH_1996_S3486095_Cards/ALEPH_1996_S3486095_card_%s.cmnd' % (row['b'], rowind) )
