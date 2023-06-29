import pandas as pd 
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self):
        test_list_file = '/home/ccl/rain_chart/01data_process/data/test_list.txt'
        with open(test_list_file, 'r') as f:
            self.test_list = [line[5:20] for line in f.readlines()]

    def __call__(self, file, label):
        print('load', file)
        D = np.load(file, allow_pickle=True).item()
        
        if label:
            result = np.concatenate([D[key][...,-1] for key in self.test_list])
        else:
            result = np.concatenate([D[key] for key in self.test_list])
        print(result.shape)
        return np.squeeze(result)
    

def confusion_matrix(yt, yp, output_file, th_list):
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
    
    with open(output_file, 'w') as f:
        for th in th_list:
            cm, precision, recall, f1 = cm_calculate(th)
            
            f.write(f'th: {th:.1f}\n')
            cm.to_csv(f, sep='\t')
            f.write('\n')
            f.write('preci, recall, f1\n')
            f.write(f'{precision:.3f}, {recall:.3f}, {f1:.3f}\n')
            f.write('\n\n')


# input setting
# exp_name = input('exp_name/sub_exp_name:')
exp_name ='Unet_1_color_v2'
sub_exp_names = ['pass2_data_mm05_M180']
th_list = np.arange(0.4, 0.51, 0.1)



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
    txt_file = f'{save_folder}/{sub_exp}.txt'
   
    confusion_matrix(test_true, test_pred, txt_file, th_list)
