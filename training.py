from utils import EarlyStopping
import torch
import time
import tqdm

def training_ssm(models_dir, model, optimizer, H, EPOCHS, trainDataLoader, trainSteps, device='cpu'):

    for epoch in tqdm.tqdm(range(EPOCHS)):
        total_loss = 0.0
        # Generate audio samples
        for batch_dict, _ in trainDataLoader:
            # Reset gradients
            optimizer.zero_grad()

            # Use the `step` function to compute the loss
            loss, _, _ = model.step(batch_dict)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss for monitoring
            total_loss += loss.item()
            print("New loss {}".format(loss))
        # Average loss per epoch
        avg_loss = total_loss / trainSteps
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    print("Training complete.")

def training_model(models_dir, model, optimizer, scheduler, lossFn, H, EPOCHS, trainDataLoader, trainSteps,
                   valDataLoader, valSteps, device='cpu'):

    early_stopper = EarlyStopping(patience=10, models_dir=models_dir)

    H['time'].append(time.time())
    print(f"[INFO] training the network for {models_dir}...")

    for e in range(EPOCHS):
        # set the model in training mode
        model.train()
        optimizer.zero_grad()

        train_loss = 0
        val_loss = 0

        for mel_specs, _ in trainDataLoader:
            # sending data to device
            mel_specs = mel_specs.to(device)
            # tags = tags.to(device)

            # perform forward pass and calculate loss
            pred_mel_specs = model(mel_specs)
            loss = lossFn(pred_mel_specs, mel_specs)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.cpu().detach().numpy()

        with torch.no_grad():
            model.eval()

            for mel_specs, _ in valDataLoader:
                # sending data to device
                mel_specs = mel_specs.to(device)
                # tags = tags.to(device)

                # perform forward pass and calculate loss
                pred_mel_specs = model(mel_specs)
                loss = lossFn(pred_mel_specs, mel_specs)

                val_loss += loss.cpu().detach().numpy()

        H['total_train_loss'].append(train_loss / trainSteps)
        H['total_val_loss'].append(val_loss / valSteps)
        H['time'].append(time.time())

        scheduler.step(H['total_val_loss'][e])

        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'train_loss_history': H}, f"{models_dir}/train_state_dict_CAE_normed.pt")

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{} ...".format(e + 1, EPOCHS))
        print("Train loss: {:.5f}".format(H['total_train_loss'][e]))
        print("Val loss: {:.5f}".format(H['total_val_loss'][e]))
        # checking if resulting loss in evaluation improved
        if early_stopper.earlyStop((e + 1), H['total_train_loss'][e], H['total_val_loss'][e], model):
            # if not improved - stop the training
            print("[INFO] Early Stopping the train process. Patience exceeded!")
            print("=============================================================")
            break

        print("=============================================================")

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] Total time taken to train the model: {:.2f}s".format(endTime - H['time'][0]))
    print("[INFO] Best loss was found in Epoch {} where the performance was {:.5f}. "
          "Model's parameters saved!".format(early_stopper.getBestEpoch(), early_stopper.getBestValLoss()))

    early_stopper.saveLossesLocally()
