import os, importlib, cv2
from pathlib import Path
from skimage.measure import block_reduce
import numpy as np

def CenterCrop_and_Split(image, dx = 80, dy = 640):
    # crop
    total_y, total_x, dim = image.shape
    crop_x = (total_x//dx) * dx
    crop_y = dy
    x0 = total_x //2 - crop_x //2
    y0 = total_y //2 - crop_y //2
    crop_img = image[y0:y0+crop_y, x0:x0+crop_x,:]
    # split
    split_img = np.hsplit(crop_img, total_x//dx)
    return np.array(split_img)

def input_data_processing(img_path, color):
    if color:
        image = cv2.imread(img_path.as_posix())/255
    else:
        image = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)/255
    image = block_reduce(image, block_size=(2,2,1), func=np.mean)
    image = CenterCrop_and_Split(image)
    return image
    
if __name__ == '__main__':
    # environmental setting
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    
    # model setting and load model
    color = True # modify to False if gray scale !!!
    model_name = 'models.Unet_1'
    saved_weight = 'models/M030' if color else 'models/...'
    model = importlib.import_module(model_name).Model()
    model.load_weights(saved_weight).expect_partial()
    
    # fig setting. modeify here!
    input_dir = Path('/NAS-DS1515P/users1/RAINFALL_STRIP_CHARTS/Train_Label/uuchen/test')
    input_imgs = input_dir.glob('*.jpg')
    Output_dir = Path('./output')
    Path(Output_dir).mkdir(parents=True, exist_ok=True)
    
    
    # start predict 
    for img_path in input_imgs:
        image = input_data_processing(img_path, color)
        pred = np.squeeze(model.predict(image))
        
        whole_pred = np.hstack(pred)
        out_path = Output_dir / (img_path.stem + '.npy')
        np.save(out_path, whole_pred)
    
