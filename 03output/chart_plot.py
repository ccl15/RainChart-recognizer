# output npy file to rain chart fig 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


class ChartStack:
    def __init__(self, n):
        self.fs = 14
        self.n = n
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 3*n))
        self.axes = axes
        self.fig = fig

    def plot_gray(self, j, data, left_tt='', right_tt=''):
        ax = self.axes if self.n == 1 else self.axes[j]
        im = ax.imshow(data, cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(left_tt,  loc='left',  fontsize = self.fs)
        ax.set_title(right_tt, loc='right', fontsize = self.fs)
        self.fig.colorbar(im,ax=ax, fraction=0.013)

    def plot_rgb(self, j, data, left_tt='', right_tt=''):
        ax = self.axes if self.n == 1 else self.axes[j]
        im = ax.imshow(data)
        ax.set_title(left_tt,  loc='left',  fontsize = self.fs)
        ax.set_title(right_tt, loc='right', fontsize = self.fs)

    def save_fig(self, save_dir, save_name):
        plt.savefig(f'{save_dir}/{save_name}.png', dpi=200, bbox_inches='tight')
        plt.close()



#%%  load data -----------------------------  
exp_name, sub_exp_name = 'Unet_1_color_v2', 'pass2_data_mm05_M180'
save_dir = f'{exp_name}/{sub_exp_name}'
Path(save_dir).mkdir(parents=True, exist_ok=True)

# load label
# test = np.load('/home/ccl/rain_chart/01data_process/test_color.npy', allow_pickle='TRUE').item()

# load original test data
#data_file = '/home/ccl/rain_chart/01data_process/data/test_color.npy'  #!!!
#test = np.load(data_file, allow_pickle='TRUE').item()

# load predict
pred = np.load(f'{exp_name}/{sub_exp_name}.npy', allow_pickle=True).item()

#%% plot -----------------------------------------------------------
for fig_name in tqdm(pred.keys()):
    #whole_test = np.hstack(test[fig_name][...,:-1])
    whole_pred = np.hstack(pred[fig_name])
    # plot  
    chart = ChartStack(1)
    #chart.plot_rgb(0, whole_test, fig_name)
    chart.plot_gray(0, whole_pred, fig_name)
    chart.save_fig(save_dir, fig_name)
   
