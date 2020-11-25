import pandas as pd
import numpy as np

for j in range(17):
    for i in range(25):
        df = pd.DataFrame(data=np.load(f"{j:02d}_{i:02d}_coeffs.npy").reshape(24*7,-1).T, index=np.load(f"{j:02d}_{i:02d}_times.npy"))
        index = pd.date_range('1/1/2018', periods=2*365, freq='D')
        df = df.reindex(index, method='nearest', limit=1).interpolate(method='linear', limit_direction='both')
        coeffs = df.to_numpy().T
        print(coeffs.shape, coeffs.dtype)
        np.save(f"{j:02d}_{i:02d}_intcoeffs", coeffs)
