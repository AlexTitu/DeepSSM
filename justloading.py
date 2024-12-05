import numpy as np
import torch
from test import calculate_total_score

models_dir = ['./dev_8/general', './dev_2/general', f'./dev_2/general_unet', f'./dev_3/general', f'./dev_4/general',
              f'./dev_5/general', f'./dev_6/general',  f'./dev_7/general']

for model in models_dir:
    previous_state = torch.load(f"{model}/train_state_dict_CAE_normed.pt")
    auc = np.load(f'{model}/auc_values.npy').tolist()
    pauc = np.load(f'{model}/pauc_values.npy').tolist()
    mean = calculate_total_score(auc, pauc)
