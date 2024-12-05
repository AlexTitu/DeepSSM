import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
from utils import DCASE2024MachineDataset


# Define the function H with threshold
def H(x, threshold=0):
    return 1 if x > threshold else 0


# Function to calculate AUC
def calculate_auc(normal_scores, anomalous_scores, threshold=0):
    N_d = len(normal_scores)
    N_plus = len(anomalous_scores)

    auc_sum = 0

    for x_j in anomalous_scores:
        for x_i in normal_scores:
            auc_sum += H(x_j - x_i, threshold)

    auc = auc_sum / (N_d * N_plus)
    return auc


# Function to calculate pAUC
def calculate_pauc(normal_scores, anomalous_scores, p, threshold=0):
    N_d = len(normal_scores)
    N_plus = len(anomalous_scores)

    pN_d = int(np.floor(p * N_d))

    if pN_d == 0 or N_plus == 0:
        return 0  # Handle edge case when counts are zero

    normal_scores = sorted(normal_scores, reverse=True)[:pN_d]

    pauc_sum = 0
    for x_j in anomalous_scores:
        for x_i in normal_scores:
            pauc_sum += H(x_j - x_i, threshold)

    pauc = pauc_sum / (pN_d * N_plus)
    return pauc


# Function to calculate the total score Ω
def calculate_total_score(auc_values, pauc_values):
    return hmean(auc_values + pauc_values)


# switching off autograd for eval
def test_loop(model, dataLoader, steps, sourceLen, device, lossFn, scores):
    with torch.no_grad():
      # set the model in eval mode
      model.eval()
      totalTestLoss = 0

      for mel_specs, tags, origin in dataLoader:
        mel_specs = mel_specs.to(device)
        tags = tags.to(device)

        pred_mel_specs = model(mel_specs)
        loss_val = lossFn(pred_mel_specs, mel_specs)
        totalTestLoss += loss_val.cpu().detach().item()

        if origin[0] == 'source':
            if tags.cpu().detach().item() == 0:
                scores['normal']['source'].append(loss_val.cpu().detach().item())
            else:
                scores['anomalous']['source'].append(loss_val.cpu().detach().item())
        else:
            if tags.cpu().detach().item() == 0:
                scores['normal']['target'].append(loss_val.cpu().detach().item())
            else:
                scores['anomalous']['target'].append(loss_val.cpu().detach().item())

    # print(f"Total loss mean: {totalTestLoss/steps}")

    return scores


# -------------------------------------
# ------- Main Test Function ----------
# -------------------------------------

def test_model(models_dir, model, machine_types, trainDataLoader, trainSteps, lossFn, device):
    # plot the training and val losses
    if models_dir in ['./dev_2/general_unet', './dev_2/general', "./dev/general", "./dev_3/general", './dev_8/general']:
        print(f"[INFO] testing the model for {models_dir}")
        previous_state = torch.load(f"{models_dir}/train_state_dict_CAE_normed.pt")
        previous_model = torch.load(f"{models_dir}/CAE_normed.pt")
        model.load_state_dict(previous_model)

        plt.style.use("ggplot")
        H = previous_state['train_loss_history']

        # Plotting loss on train and evaluation
        plt.figure("total_loss").clear()
        plt.plot(H["total_train_loss"], label="total_train_loss", linestyle="solid")
        plt.plot(H["total_val_loss"], label="total_val_loss", linestyle="solid")
        plt.title("Evolutia functiei de cost in timpul antrenarii")
        plt.xlabel("# Epoca")
        plt.ylabel("Cost")
        plt.legend(loc="upper right")
        plt.savefig(f"{models_dir}/train_val_graph_CAE_normed.png")

        # Parameters
        domains = ['source', 'target']  # predefine domains
        p = 0.1  # false positive rate for pAUC

        auc_values = []
        pauc_values = []

        # Calculate AUC and pAUC for each combination of machine type, section, and domain
        for m in machine_types:
            scores = {
                'normal': {'source': [], 'target': []},
                'anomalous': {'source': [], 'target': []}
            }
            dev_test_dataset = DCASE2024MachineDataset('./DCASE2024', ('dev', 'test'), machine=m,
                                                       extension='.npy', standardize=None, isTesting=True)
            print(f'Number of test Samples: {len(dev_test_dataset)}')

            testDataLoader = DataLoader(dev_test_dataset, batch_size=1)

            testSteps = len(testDataLoader.dataset) // 1

            scores = test_loop(model, testDataLoader, testSteps, dev_test_dataset.sourceLen(), device, lossFn, scores)

            # Calculate pAUC across both domains for each machine type
            all_normal_scores = scores['normal']['source'] + scores['normal']['target']
            all_anomalous_scores = scores['anomalous']['source'] + scores['anomalous']['target']
            pauc = calculate_pauc(all_normal_scores, all_anomalous_scores, p)
            pauc_values.append(pauc)

            for d in domains:
                auc = calculate_auc(scores['normal'][d], all_anomalous_scores)
                auc_values.append(auc)


        np.save(f'./{models_dir}/auc_values.npy', auc_values)
        np.save(f'./{models_dir}/pauc_values.npy', pauc_values)
        # Calculate the total score
        total_score = calculate_total_score(auc_values, pauc_values)

        print(f'Total Score (Ω): {total_score}')
        np.save(f'./{models_dir}/total_score.npy', total_score)

    else:

        print(f"[INFO] testing the model for {models_dir}")
        previous_state = torch.load(f"{models_dir}/train_state_dict_CAE_normed.pt")
        previous_model = torch.load(f"{models_dir}/CAE_normed.pt")
        model.load_state_dict(previous_model)

        plt.style.use("ggplot")
        H = previous_state['train_loss_history']

        # Plotting loss on train and evaluation
        plt.figure("total_loss").clear()
        plt.plot(H["total_train_loss"], label="total_train_loss", linestyle="solid")
        plt.plot(H["total_val_loss"], label="total_val_loss", linestyle="solid")
        plt.title("Evolutia functiei de cost in timpul antrenarii")
        plt.xlabel("# Epoca")
        plt.ylabel("Cost")
        plt.legend(loc="upper right")
        plt.savefig(f"{models_dir}/train_val_graph_CAE_normed.png")

        # switching off autograd for eval
        with torch.no_grad():
            # set the model in eval mode
            model.eval()
            trainLosses = []

            for mel_specs, _ in trainDataLoader:
                mel_specs = mel_specs.to(device)

                pred_mel_specs = model(mel_specs)
                loss_val = lossFn(pred_mel_specs, mel_specs)
                trainLosses.append(loss_val.cpu().detach().item())

        # Assume trainLosses is a list of the reconstruction errors on your training data
        train_losses = np.array(trainLosses)
        mean = np.mean(train_losses)
        std_dev = np.std(train_losses)
        print(f"Total loss mean: {mean}")
        print(f"Total loss std: {std_dev}")

        # Set threshold as mean plus 3 standard deviations
        threshold = mean + 3 * std_dev

        H['threshold'] = threshold

        torch.save({
            'train_loss_history': H}, f"{models_dir}/train_history_values_normed.pt")

        # Parameters
        domains = ['source', 'target']  # predefine domains
        p = 0.1  # false positive rate for pAUC

        scores = {
            'normal': {'source': [], 'target': []},
            'anomalous': {'source': [], 'target': []}
        }

        auc_values = []
        pauc_values = []

        # Calculate AUC and pAUC for each combination of machine type, section, and domain

        dev_test_dataset = DCASE2024MachineDataset('./DCASE2024', ('dev', 'test'), machine=machine_types,
                                                   extension='.npy', standardize=None, isTesting=True)
        print(f'Number of test Samples: {len(dev_test_dataset)}')

        testDataLoader = DataLoader(dev_test_dataset, batch_size=1)

        testSteps = len(testDataLoader.dataset) // 1

        scores = test_loop(model, testDataLoader, testSteps, dev_test_dataset.sourceLen(), device, lossFn,
                           scores)

        # Calculate pAUC across both domains for each machine type
        all_normal_scores = scores['normal']['source'] + scores['normal']['target']
        all_anomalous_scores = scores['anomalous']['source'] + scores['anomalous']['target']
        pauc = calculate_pauc(all_normal_scores, all_anomalous_scores, p)
        pauc_values.append(pauc)

        for d in domains:
            auc = calculate_auc(scores['normal'][d], all_anomalous_scores)
            auc_values.append(auc)

        np.save(f'./{models_dir}/auc_values.npy', auc_values)
        np.save(f'./{models_dir}/pauc_values.npy', pauc_values)
        # Calculate the total score
        total_score = calculate_total_score(auc_values, pauc_values)

        print(f'Total Score (Ω): {total_score}')
        np.save(f'./{models_dir}/total_score.npy', total_score)


