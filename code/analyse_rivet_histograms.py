import yoda
import os
hist_data_path = os.path.join(os.getcwd(), 'rivet_histograms', 'data', 'ALEPH_1996_S3486095.yoda')
hist_data = yoda.read(hist_data_path)
print(hist_data)
