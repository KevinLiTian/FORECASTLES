import pandas as pd
import torch
import numpy as np
from timeit import default_timer as timer
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import tqdm
from model import *
from data import *
from torchsummary import summary

train_on_gpu = torch.cuda.is_available()


def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_mse = 0.0
        valid_mse = 0.0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)
                    loss_mse = criterion(torch.exp(output), torch.exp(target))
                    valid_mse += loss_mse.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                valid_mse = valid_mse / len(valid_loader.dataset)

                history.append([train_loss, valid_loss])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                            f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \tValidation MSE: {valid_mse:.4f}'
                    )
                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.4f}'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=['train_loss', 'valid_loss'])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.4f}'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss'])
    return model, history


def plot_loss(history, filename):
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(
            history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average cross entropy')
    plt.title('Training, Validation and testing Losses')
    plt.savefig(filename)


def train_mlp():
    X_train_scaled, X_test_scaled, y_train, y_test, _ = load_data(include_capacity=True)
    train_dataset = DefaultDataset(X_train_scaled.to_numpy(), np.log(np.expand_dims(np.asarray(y_train)+1.0e-1, axis=-1)))
    test_dataset = DefaultDataset(X_test_scaled.to_numpy(), np.log(np.expand_dims(np.asarray(y_test)+1.0e-1, axis=-1)))

    x, y = train_dataset[10]
    input_shape = x.shape[0]
    model = MLP(input_shape, 1)
    summary(model, input_size=(input_shape,), batch_size=128)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-5)
    model, history = train(
        model,
        criterion,
        optimizer,
        train_dataloader,
        test_dataloader,
        save_file_name='./mlp_model_log_cap.pt',
        max_epochs_stop=20,
        n_epochs=100,
        print_every=1)

    plot_loss(history, "./loss.jpg")


def train_seq(combined_data_path):
    X_train_scaled, X_test_scaled, y_train, y_test, info = load_sequence_data(combined_data_path)
    train_dataset = SequenceDataset(X_train_scaled, np.log(np.asarray(y_train)))
    test_dataset = SequenceDataset(X_test_scaled, np.log(np.asarray(y_test)))
    print("Datasets loaded")
    x, y = train_dataset[10]
    input_shape = x.shape[1]
    print(f"input shape {input_shape}")
    model = TimeModel(input_shape,
                      hidden_dim=512,
                      num_layers=1,
                      num_heads=4,
                      output_dim=1,
                      seq_len=30).to("cuda")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)
    model, history = train(
        model,
        criterion,
        optimizer,
        train_dataloader,
        test_dataloader,
        save_file_name='./trained_models/trans_model_log2.pt',
        max_epochs_stop=5,
        n_epochs=100,
        print_every=1)

    plot_loss(history, "./loss.jpg")
    return model, history


def train_full_trans(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_mse = 0.0
        valid_mse = 0.0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, prev_days, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target, prev_days = data.cuda(), target.cuda(), prev_days.cuda()
            # print("avg shape", avg.shape)

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data, prev_days)
            # print("output shape", output.shape)
            # print("target", target.shape)
            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, prev_days, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target, prev_days = data.cuda(), target.cuda(), prev_days.cuda()

                    # Forward pass
                    output = model(data, prev_days)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)
                    loss_mse = criterion(torch.exp(output), torch.exp(target))
                    valid_mse += loss_mse.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                valid_mse = valid_mse / len(valid_loader.dataset)

                history.append([train_loss, valid_loss])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                            f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f} \tValidation MSE: {valid_mse:.4f}'
                    )
                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.4f}'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=['train_loss', 'valid_loss'])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.4f}'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss'])
    return model, history


def train_fullTrans(combined_data_path):
    X_train_scaled, X_test_scaled, y_train, y_test, info = load_sequence_data(combined_data_path)
    train_dataset = FullTransSequenceDataset(X_train_scaled, np.log(np.asarray(y_train)), window=30)
    test_dataset = FullTransSequenceDataset(X_test_scaled, np.log(np.asarray(y_test)), window=30)
    print("Datasets loaded")
    x, y, _ = train_dataset[10]
    input_shape = x.shape[1]
    print(f"input shape {input_shape}")
    model = Full_Transformer(input_shape,
                      curr_state_dim=1,
                      hidden_dim=256,
                      num_layers=2,
                      num_heads=4,
                      output_dim=1).to("cuda")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-5)
    model, history = train_full_trans(
        model,
        criterion,
        optimizer,
        train_dataloader,
        test_dataloader,
        save_file_name='./full_trans_model_4h_2l_256.pt',
        max_epochs_stop=20,
        n_epochs=200,
        print_every=1)

    plot_loss(history, "./loss.jpg")
    return model, history


def evaluate(model_path):
    X_train_scaled, X_test_scaled, y_train, y_test, df_copy = load_data(include_capacity=True)
    train_dataset = DefaultDataset(X_train_scaled.to_numpy(), np.log(np.expand_dims(np.asarray(y_train), axis=-1)))
    test_dataset = DefaultDataset(X_test_scaled.to_numpy(), np.log(np.expand_dims(np.asarray(y_test), axis=-1)))

    x, y = train_dataset[10]
    input_shape = x.shape[0]
    model = MLP(input_shape, 1)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    y_pred = model(torch.from_numpy(X_test_scaled.to_numpy()).float())
    y_pred = y_pred.detach().numpy()
    y_pred = np.squeeze(np.exp(y_pred))
    df_copy['PRED_SC'] = y_pred
    plt.figure(figsize=(8, 6))
    plt.plot(df_copy.groupby('OCCUPANCY_DATE')['SERVICE_USER_COUNT'].mean())
    plt.plot(df_copy.groupby('OCCUPANCY_DATE')['PRED_SC'].mean())
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Day')
    plt.ylabel('Average user count')
    plt.title('Evaluation of predicted user counts')
    plt.savefig('./trained_models/mlp_model_log_cap_eval.jpg')


def open_eval(model_path, combined_data_path):
    X_train_scaled, X_test_scaled, y_train, y_test, info = load_sequence_data(combined_data_path)
    cols = list(X_train_scaled.columns)
    cols.remove("OCCUPANCY_DATE")
    train_dataset = SlowSequenceDataset(X_train_scaled, np.log(np.asarray(y_train)))
    test_dataset = SlowSequenceDataset(X_test_scaled, np.log(np.asarray(y_test)))

    x, y, _, _ = train_dataset[10]
    input_shape = x.shape[1]
    print(f"input shape {input_shape}")
    model = TimeModel(input_shape,
                      hidden_dim=512,
                      num_layers=1,
                      num_heads=4,
                      output_dim=1,
                      seq_len=30).to("cuda")
    model.eval()
    model.load_state_dict(torch.load(model_path))
    sc = info["scaler"]
    # alt_cols = list(map(lambda x: x.replace('SERVICE_USER_COUNT', 'UNSCALED_SC'), cols))
    test_dataset.x["UNSCALED_SC"] = y_test
    for i in tqdm.tqdm(range(len(test_dataset))):
        _, _, idx, ss = test_dataset[i]
        ss[cols] = sc.inverse_transform(ss[cols])
        ss = ss.drop(columns=["SERVICE_USER_COUNT"])
        ss = ss.rename(columns={"UNSCALED_SC": "SERVICE_USER_COUNT"})
        input = sc.transform(ss[cols])
        y_pred = model(torch.unsqueeze(torch.from_numpy(input), dim=0).to("cuda").float())
        test_dataset.x["UNSCALED_SC", idx] = np.exp(torch.squeeze(y_pred.detach()).to("cpu").item())
    test_dataset.x["ACTUAL"] = y_test
    plt.figure(figsize=(8, 6))
    plt.plot(test_dataset.x.groupby('OCCUPANCY_DATE')['ACTUAL'].sum())
    plt.plot(test_dataset.x.groupby('OCCUPANCY_DATE')['UNSCALED_SC'].sum())
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Day')
    plt.ylabel('Average user count')
    plt.title('Evaluation of predicted user counts')
    plt.savefig('./trans_model_eval.jpg')


def open_eval_trans(model_path):
    X_train_scaled, X_test_scaled, y_train, y_test, info = load_sequence_data(combined_data_path)
    cols = list(X_train_scaled.columns)
    cols.remove("OCCUPANCY_DATE")
    train_dataset = SlowSequenceDataset(X_train_scaled, np.log(np.asarray(y_train)), window=30)
    X_test_scaled["UNSCALED_SC"] = y_test
    test_dataset = SlowSequenceDataset(X_test_scaled, np.log(np.asarray(y_test)), window=30)

    x, y, _, _ = train_dataset[10]
    input_shape = x.shape[1]
    print(f"input shape {input_shape}")
    model = Full_Transformer(input_shape,
                      curr_state_dim=1,
                      hidden_dim=256,
                      num_layers=2,
                      num_heads=4,
                      output_dim=1).to("cuda")
    model.eval()
    model.load_state_dict(torch.load(model_path))
    sc = info["scaler"]
    # alt_cols = list(map(lambda x: x.replace('SERVICE_USER_COUNT', 'UNSCALED_SC'), cols))
    for i in tqdm.tqdm(range(len(test_dataset))):
        _, _, idx, ss = test_dataset[i]
        ss[cols] = sc.inverse_transform(ss[cols])
        ss = ss.drop(columns=["SERVICE_USER_COUNT"])
        ss = ss.rename(columns={"UNSCALED_SC": "SERVICE_USER_COUNT"})
        dec_prompt = ss["SERVICE_USER_COUNT"]
        dec_prompt = np.expand_dims(np.log(dec_prompt.to_numpy()), axis=-1)
        input = sc.transform(ss[cols])
        y_pred = model(torch.unsqueeze(torch.from_numpy(input), dim=0).to("cuda").float(), torch.unsqueeze(torch.from_numpy(dec_prompt), dim=0).to("cuda").float())
        gnum = test_dataset.get_gnum(i)
        val = np.exp(torch.squeeze(y_pred[0, -1, 0].detach()).to("cpu").item())
        test_dataset.gnum_subsets[gnum]["UNSCALED_SC"][idx] = val
        test_dataset.x["UNSCALED_SC"][idx] = val
    test_dataset.x["ACTUAL"] = y_test
    info["X_test"]["PREDICTED_SERVICE_USER_COUNT"] = test_dataset.x["UNSCALED_SC"]
    plt.figure(figsize=(8, 6))
    plt.plot(test_dataset.x.groupby('OCCUPANCY_DATE')['ACTUAL'].sum())
    plt.plot(test_dataset.x.groupby('OCCUPANCY_DATE')['UNSCALED_SC'].sum())
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Day')
    plt.ylabel('Average user count')
    plt.title('Evaluation of predicted user counts')
    plt.savefig('./trans_model_eval.jpg')
    return info["X_test"]


if __name__ == "__main__":
    combined_data_path = "./shelter_neighbourhood_features_pca.csv"
    model, history = train_fullTrans(combined_data_path)

    # open_eval('/trained_models/trans_model_log2.pt', combined_data_path)
    # evaluate('./trained_models/mlp_model_log_cap.pt')
