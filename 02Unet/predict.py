import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from modules.experiment_helper import parse_exp_settings
from modules.training_helper import create_model_by_exp_settings
import numpy as np
from pathlib import Path



def main(exp_path, only_this_sub):
    exp_list = parse_exp_settings(exp_path, only_this_sub)
    # load test data
    data_file = '/home/ccl/rain_chart/01data_process/test_color.npy'
    test_data = np.load(data_file, allow_pickle='TRUE').item()
    
    for sub_exp_settings in exp_list:
        # load model
        exp_name = sub_exp_settings['experiment_name']
        sub_exp_name = sub_exp_settings['sub_exp_name']
        model_save_path = f'saved_models/{exp_name}/{sub_exp_name}/Unet'
        model = create_model_by_exp_settings(sub_exp_settings['model'], model_save_path)
        print('Start predict sub-exp:', sub_exp_name)
        
        # set output 
        pred_set = dict()
        for key in list(test_data.keys()):
            pred_set[key] = model(test_data[key][...,:3])
            
        save_folder = f'/home/ccl/rain_chart/03output/{exp_name}/'
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        np.save(f'{save_folder}/{sub_exp_name}.npy', pred_set)
        print('Save predict file:', exp_name, sub_exp_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path')
    parser.add_argument('-sub','--only_this_sub', default='')
    args = parser.parse_args()

    main(args.exp_path, args.only_this_sub)


