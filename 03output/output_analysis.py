import pandas as pd 
import numpy as np


def confusion_matrix_and_calculate(y_pred, y_true, path='', th=0.5):
    # conver predict data to 0, 1
    y_pred_01 = np.zeros(y_pred.shape)
    y_pred_01[y_pred>th] = 1
    true = pd.Categorical(y_true.ravel(), categories=[0,1])
    pred = pd.Categorical(y_pred_01.ravel(), categories=[0,1])
    cm = pd.crosstab(true, pred, rownames=['True'], colnames=['Predicted'])
    cm = cm.sort_index(ascending=False).sort_index(axis=1, ascending=False)
    print(cm)
    
    matrix = cm.values
    precision = matrix[0,0] / sum(matrix[:,0])
    recall = matrix[0,0] / sum(matrix[0,:])
    f1 = 2 * precision * recall / (precision + recall)
    print(f'precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}')
    
    if path: 
        with open(path, 'w') as f:
            cm.to_csv(f, sep='\t')
            f.write('\n')
            f.write('preci, recall, f1\n')
            f.write(f'{precision:.3f}, {recall:.3f}, {f1:.3f}')
    
def load_and_stack(file, test=False):
    D = np.load(file, allow_pickle=True).item()
    keys = sorted(list(D.keys()))
    
    result = D[keys[0]][...,1] if test else D[keys[0]]
    for key in keys[1:]:
        new = D[key][...,1] if test else D[key]
        result = np.concatenate((result, new))
    print(result.shape, test)
    return np.squeeze(result)

true_file = '/home/ccl/rain_chart/01data_process/wide_test.npy'
test_true = load_and_stack(true_file, True)

# exp_name = input('exp_name/sub_exp_name:')
exps = ['Unet_1_data_wide/MSE01_1em3']
for exp_name in exps:
    exp_file = f'/home/ccl/rain_chart/03output/{exp_name}/pred.npy'
    # txt_path = f'/home/ccl/rain_chart/03output/{exp_name}/confusion.txt'
    test_pred = load_and_stack(exp_file, False)
    
    for th in np.arange(0.6, 0.9, 0.1 ):
        print(th)
        confusion_matrix_and_calculate(test_pred, test_true, th = th)

