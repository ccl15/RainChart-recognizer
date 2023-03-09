import cv2, random, h5py
from glob import glob
from skimage.measure import block_reduce
import numpy as np
from tqdm import tqdm

def read_label(name):
    img3 = cv2.imread(name)
    mask = (img3[...,0]<20) & (img3[...,1]<20) & (img3[...,2]>240)
    img = np.zeros(img3.shape[:2])
    img[mask] = 1
    return img

def CenterCrop_and_Split(image, dx=80):
    # crop
    total_y, total_x, d2 = image.shape
    crop_x = (total_x//dx) * dx
    crop_y = 640
    x0 = total_x //2 - crop_x //2
    y0 = total_y //2 - crop_y //2
    crop_img = image[y0:y0+crop_y, x0:x0+crop_x,:]
    # split
    split_img = np.hsplit(crop_img, total_x//dx)
    return np.array(split_img)


    
def pair_and_processed_to_dataset(label_folder, output_name=''):
    img_folder = '/NAS-DS1515P/users1/RAINFALL_STRIP_CHARTS'
    img_label_names = sorted(glob(f'{img_folder}/Train_Label/{label_folder}/RAIN-*'))
    
    # Stations = []; Dates = []; Labels = []; Images = []
    Dataset = dict()
    bs = 2 # pooling block size
    for name_lab in tqdm(img_label_names):
        string = name_lab.split('RAIN-')[1]
        station = string[:6]
        date = string[7:15]
        name_inp = f'{img_folder}/DATA/RAIN-{station}/RAIN-{station}-{date[:4]}/RAIN-{station}-{date}.jpg'
        
        img_lab = read_label(name_lab)
        img_inp = cv2.imread(name_inp)/255
        
        if not img_lab.shape == img_inp.shape[:2]:
            print(station, date, 'Error')
            continue
        
        pooled_lab = block_reduce(img_lab, block_size=(bs,bs), func=np.min)
        pooled_inp = block_reduce(img_inp, block_size=(bs,bs,1), func=np.mean)

        pool_pair_img = np.dstack((pooled_inp, pooled_lab))
        Dataset[f'{station}-{date}'] = CenterCrop_and_Split( pool_pair_img, dx=80 )  #(n,640,80,2)
    if output_name:
        np.save(f'{output_name}.npy', Dataset)
    return Dataset
                

 
#%%
def stack_set(D, set_keys):
    result = D[set_keys[0]]
    Info_list = [set_keys[0]] * len(D[set_keys[0]])
    
    for key in set_keys[1:]:
        result = np.concatenate((result, D[key]))
        Info_list.extend( [key] * len(D[key]) )
    print(len(Info_list), result.shape)
    return result, Info_list

    
def split_dataset(D, db_name):
    # D = np.load(f'{db_name}.npy', allow_pickle='TRUE').item()

    keys = list(D.keys())
    overlat = ['467770-19880813', '467770-19880812']
    for element in overlat:
        keys.remove(element)
    
    n = len(keys)
    n_train = int(0.8 * n)
    random.shuffle(keys)
    train_keys = keys[:n_train]
    valid_keys = keys[n_train:]
    train, train_info = stack_set(D, train_keys)
    valid, valid_info = stack_set(D, valid_keys)
    
    # save to h5
    with h5py.File(f'{db_name}.h5', 'w') as f:
        f.create_dataset('train', data=train)
        f.create_dataset('valid', data=valid)
        f.create_dataset('train_info', data=train_info)
        f.create_dataset('valid_info', data=valid_info)
        


#%%
if __name__ == '__main__':
    dataset = pair_and_processed_to_dataset('uuchen/test', 'test_color')
    # split_dataset(dataset, 'test_color')

