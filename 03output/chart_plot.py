# output npy file to rain chart fig 
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm



# load data
test = np.load('/home/ccl/rain_chart/01data_process/test_color.npy', allow_pickle='TRUE').item()

# load predict
# exp_name = input('exp_name/sub_exp_name:')
def load_pred(exp_name, sub_exp_name):
    pred_file = f'/home/ccl/rain_chart/03output/{exp_name}/{sub_exp_name}.npy'
    return np.load(pred_file, allow_pickle=True).item()

# pred1 = load_pred('Unet_1_data_wide/BCE01_5em6')
# pred2 = load_pred('Unet_1_data_wide/MSE01_1em3')


#%%
def plot_one(fig_name, whole):
    fs = 16
    plt.figure(figsize=(10,3))
    plt.imshow(whole,  cmap='gray_r', vmin=0, vmax=1)
    plt.title(fig_name, loc='left', fontsize = fs)
    plt.title('label', loc='right', fontsize = fs)
    plt.colorbar(fraction = 0.013)
    plt.savefig(f'Fig_label/{fig_name}.png',dpi=200)
    plt.close()

def plt_two(whole_pred1, whole_pred2):
    fs = 16
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    
    im = axes[0].imshow(whole_pred1, cmap='gray_r', vmin=0, vmax=1)
    axes[0].set_title(fig_name, loc='left', fontsize = fs)
    axes[0].set_title('BCE', loc='right', fontsize = fs)
    fig.colorbar(im,ax=axes[0], fraction=0.013)

    im = axes[1].imshow(whole_pred2, cmap='gray_r', vmin=0, vmax=1)
    axes[1].set_title(fig_name, loc='left', fontsize = fs)
    axes[1].set_title('MSE', loc='right', fontsize = fs)
    fig.colorbar(im, ax=axes[1], fraction=0.013)
    
    plt.savefig(f'BCEvsMSE/{fig_name}.png',dpi=200)
    plt.close()
    

# satack pieces to one fig
figs_name = list(test.keys())
for i in range(0,len(figs_name)):
    fig_name = figs_name[i]
    # whole_pred1 = np.hstack(pred1[fig_name])
    # whole_pred2 = np.hstack(pred2[fig_name])
    # whole_orig = np.hstack(test[fig_name][...,0])
    whole_true = np.hstack(test[fig_name][...,-1])
    
    # plot  
    plot_one(fig_name, whole_true)
