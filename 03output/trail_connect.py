# connect probability
import numpy as np
import matplotlib.pyplot as plt

#%%
def plot_pred_trail(fig_name, whole):
    trail = get_trail(whole)
    
    
    fs = 16
    plt.figure(figsize=(10,3))
    plt.imshow(whole_pred,  cmap='gray_r', vmin=0, vmax=1)
    plt.plot(trail[0,1], trail[0,0], c='b', marker='*', markersize=10)
    # trail = np.sort
    plt.scatter(trail[:,1],  trail[:,0], c='r', s=1 )
    
    plt.title(fig_name, loc='left', fontsize = fs)
    # plt.title(di, loc='right', fontsize = fs)
    plt.savefig(f'mouse/{fig_name}.png',dpi=200)
    plt.close()


def get_trail(probs):
    I, J = probs.shape
    # Find the row with the highest sum of probabilities
    grid = np.where(probs > 0.5, 1, 0)
    i0 = np.argmax(np.sum(grid, axis=1))
    j0 = np.argmax(probs[i0,:])

    trail = [(i0, j0)]
    di = 10
    th = 0.005

    # forward
    i_now = i0
    for j_now in range(j0+1, J):
        if i_now < I-di and i_now > di:
            i_now += np.argmax(probs[i_now-di:i_now+1, j_now])-di
        else:
            i_now = np.argmax(probs[:, j_now])
        trail.append( (i_now, j_now) )
        if probs[i_now, j_now] < th:
            i_now = -1
    # backward    
    i_now = i0
    for j_now in range(j0-1, 0, -1):
        if i_now < I-di and i_now > di:
            i_now += np.argmax(probs[i_now:i_now+di+1, j_now])
        else:
            i_now = np.argmax(probs[:, j_now])
        trail.append( (i_now, j_now) )
        if probs[i_now, j_now] < th:
            i_now = -1

    return np.array(trail)

#%% load predict
def load_pred(exp_name, sub_exp_name):
    pred_file = f'{exp_name}/{sub_exp_name}.npy'
    return np.load(pred_file, allow_pickle=True).item()


# main 
exp_name, sub_exp_name = 'Unet_1_color_v2', 'MSE01_1em3_M036'
pred = load_pred(exp_name, sub_exp_name)

# fig_name ='466900-19440813'
for fig_name in pred.keys():
    whole_pred = np.hstack(pred[fig_name])
    plot_pred_trail(fig_name, whole_pred)
