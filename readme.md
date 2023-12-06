# RainChart-recognizer


## Requirements

0. Install pipenv (if not already installed)
```bash
pip install pipenv
```
1. Install Packages from Pipfile
```bash
pipenv install
```

## Data process
Create data files
1. Go to **01data_process**.
2. modify data path at fig2dat.py
```bash
pipenv run python fig2dat.py
```

## Training
1. Go to **02Unet**. Run this command:
```
pipenv run python main.py <experiment_path> -G <GPU_number>
```

## Evaluate testing data
1. predict testing data
``` bash
pipenv run python predict <experiment_path> -s <sub-exp name> -w <weight name>
```
2. evaluatie result

```bash  
cd 03output
vi output_analysis.py
```

## Use recongnizer
Sample code of useing recongnizer **04run_pred/predict.py**.
