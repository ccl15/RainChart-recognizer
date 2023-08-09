import pandas as pd 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
plt.rcParams.update({'font.size': 12})

class DataLoader:
    def __init__(self):
        test_list_file = '/home/ccl/rain_chart/01data_process/data/test_list.txt'
        with open(test_list_file, 'r') as f:
            self.test_list = [line[5:20] for line in f.readlines()]

    def __call__(self, file, label):
        D = np.load(file, allow_pickle=True).item()
        
        if label:
            result = np.concatenate([D[key][...,-1] for key in self.test_list])
        else:
            result = np.concatenate([D[key] for key in self.test_list])
        print(file, result.shape)
        return np.squeeze(result)
    

def confusion_matrix(yt, yp, th_list):
    def cm_calculate(th):
        # conver predict data to 0, 1
        yp_01 = (yp > th).astype(np.int32)
        true = pd.Categorical(yt, categories=[0,1])
        pred = pd.Categorical(yp_01, categories=[0,1])
        cm = pd.crosstab(true, pred, rownames=['True'], colnames=['Predicted'])
        cm = cm.sort_index(ascending=False).sort_index(axis=1, ascending=False)
        
        matrix = cm.values
        precision = matrix[0,0] / matrix[:,0].sum()
        recall = matrix[0,0] / matrix[0,:].sum()
        f1 = 2 * precision * recall / (precision + recall)
        return cm, precision, recall, f1
    
    yt = yt.ravel().astype(np.int32)
    yp = yp.ravel()
    
    with open(f'{save_folder}/{sub_exp}.txt', 'w') as f:
        for th in th_list:
            cm, precision, recall, f1 = cm_calculate(th)
            
            f.write(f'th: {th:.1f}\n')
            cm.to_csv(f, sep='\t')
            f.write('\n')
            f.write('preci, recall, f1\n')
            f.write(f'{precision:.3f}, {recall:.3f}, {f1:.3f}\n')
            f.write('\n\n')

def plot_roc_curve(yt, yp):
    fpr, tpr, ths = roc_curve(yt.ravel(), yp.ravel())
    aucs = auc(yt.ravel(), yp.ravel())

    plt.plot(fpr, tpr, lw=3)
    plt.plot([0,1],[0,1], '--', lw =1)
    plt.text(0.6, 0.1, f'auc:{aucs:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0,1,0,1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{save_folder}/{sub_exp}.png', bbox_inches='tight', dpi=200)
    
def plot_PR_cureve(yt, yp):
    p, r, t = precision_recall_curve(yt.ravel(), yp.ravel())
    plt.plot(r,p, lw=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0,1,0,1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{save_folder}/PR_{sub_exp}.png', bbox_inches='tight', dpi=200)
    
    
# input setting
# exp_name = input('exp_name/sub_exp_name:')
exp_name ='Unet_color_v3'
sub_exp_names = ['U1_pass_M']
th_list = np.arange(0.3, 0.51, 0.1)


# load true
data_loader = DataLoader()
true_file = '/home/ccl/rain_chart/01data_process/data/test_color.npy' #!!!
test_true = data_loader(true_file, True)
# load sub exp
for sub_exp in sub_exp_names:
    exp_file = f'{exp_name}/{sub_exp}.npy'
    test_pred = data_loader(exp_file, False)

    save_folder = f'{exp_name}/matrix'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    confusion_matrix(test_true, test_pred, th_list)
    #plot_roc_curve(test_true, test_pred)
    plot_PR_cureve(test_true, test_pred)
