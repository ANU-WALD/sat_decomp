import pandas as pd
import numpy as np

import torch
import torch.nn as nn

device = torch.device("cpu")

class AE(nn.Module):
    def __init__(self, n_comps, n_coeffs):
        super(AE, self).__init__()
        self.n_coeffs = n_coeffs
        self.n_comps = n_comps
        self.D = nn.Linear(1, self.n_comps*404*404, bias=False)
        self.conv1 = nn.Conv2d(self.n_comps, 2*self.n_comps, kernel_size=5, bias=False)
        self.coeffs = nn.Linear(1, 2*self.n_comps*self.n_coeffs, bias=False)

    def forward(self, x):
        base = self.D(x)
        conv1 =  torch.tanh(self.conv1(base.reshape(1, self.n_comps, 404, 404)))
        coeffs = self.coeffs(x).view(2*self.n_comps, self.n_coeffs)
        return torch.einsum('ki,kj->ji', conv1.view(2*self.n_comps,400*400), coeffs)


for j in range(17):
    for i in range(25):
        print(i,j)

        # 1.- Save mean as f32
        mean = np.load(f"{j:02d}_{i:02d}_mean.npy")
        np.save(f"{j:02d}_{i:02d}_meanf32.npy", mean.astype(np.float32))
        
        # 2.- Reshape and save nn parameters
        net = torch.load(f"{j:02d}_{i:02d}_net.pt", map_location='cpu')
        net.eval()
        base = net.D.weight.detach().numpy().reshape(1,12,404,404)
        base = np.moveaxis(base,1,-1)
        np.save(f"{j:02d}_{i:02d}_basenp", base)

        ker = net.conv1.weight.detach().numpy()
        ker = np.moveaxis(ker,1,-1)
        ker = np.moveaxis(ker,0,-1)
        np.save(f"{j:02d}_{i:02d}_kernp", base)


        # 3.- Interpolate coeffs to daily values
        df = pd.DataFrame(data=np.load(f"{j:02d}_{i:02d}_coeffs.npy").reshape(24*7,-1).T, index=np.load(f"{j:02d}_{i:02d}_times.npy"))
        index = pd.date_range('1/1/2018', periods=2*365, freq='D')
        df = df.reindex(index, method='nearest', limit=1).interpolate(method='linear', limit_direction='both')
        coeffs = df.to_numpy().T
        np.save(f"{j:02d}_{i:02d}_coeffsnp", coeffs)
