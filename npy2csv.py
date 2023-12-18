import numpy as np
import pandas as pd
array = np.load(r'C:\Users\frank\Desktop\jd5374\dataset\Adrian\0\saved_obs\rel_mat.npy')
df = pd.DataFrame(array)
df.to_csv('output_file.csv', index=False)
