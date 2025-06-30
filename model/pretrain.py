import logging
import pprint
import time
import json
import os
import torch
import torch.nn as nn
import wandb
import gc
from itertools import combinations
from torch.nn.functional import softmax
from picme_src import models, picme_utils
from picme_src.losses import InfoNCE
from picme_src.data import *
from picme_src.argparser import pretrain_arg_parser

torch.cuda.empty_cache()
gc.collect()

_DATA_DIR = "<your_data_dir>"
_MODALITY_SHAPES = {"demo": 44, "ts": 76}

def save_pair_weights(epoch, pair_weights, path_prefix, is_best=False):
    """Save the learned pair weights to a JSON file."""
    filename = f"{path_prefix}_pair_weights_epoch_current.json" if not is_best else f"{path_prefix}_pair_weights_epoch_best.json"
    with open(filename, 'w') as f:
        json.dump(pair_weights, f)
    print(f"Pair weights saved to {filename}")

def train_model_single_pair(dataloaders_dict, criterion, len_train, len_val, config, path, args):
    """
    Train a contrastive model for a single pair of modalities.
    
    Args:
        dataloaders_dict: Dictionary containing train and val dataloaders
        criterion: Loss function (InfoNCE)
        len_train: Length of training dataset
        len_val: Length of validation dataset
        config: Configuration from wandb
        path: Path for saving model checkpoints
        args: Command line arguments
    
    Returns:
        Trained model
    """
    modality1, modality2 = args.modality_pairs
    picme_utils.set_seed(config.random_seed)
    
    ts_input_dim = _MODALITY_SHAPES['ts']
    demo_input_dim = _MODALITY_SHAPES['demo']
    projection_dim = config.projection_dim
    
    model = models.ContrastiveModel(ts_input_dim, demo_input_dim, projection_dim)
    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = config.epochs
    optimizer = picme_utils.build_optimizer(model, config.optimizer, config.learning_rate, 0.01)

    since = time.time()

    best_acc = 0.0
    patience = 5 
    trigger = 0
    acc_dict = {}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                length = len_train
                model.train()  
            else:
                length = len_val
                model.eval()  

            running_loss = 0.0
            running_top1_acc = 0.0
            running_top5_acc = 0.0
            num_batches = 0

            for batch in dataloaders_dict[phase]:
                modality1_data, modality2_data = batch[0], batch[1]
                attention_mask1, attention_mask2, lengths = None, None, None
                
                # Handle attention masks and sequence lengths for different modalities
                if modality1 in ['text_rad', 'text_ds']:
                    modality1_data, attention_mask1 = modality1_data
                    modality1_data, attention_mask1 = modality1_data.to(device), attention_mask1.to(device) 

                if modality2 in ['text_rad', 'text_ds']:
                    modality2_data, attention_mask2 = modality2_data
                    modality2_data, attention_mask2 = modality2_data.to(device), attention_mask2.to(device)
                    
                if modality1 in ['ts']:
                    modality1_data, lengths = modality1_data
                    modality1_data, lengths = modality1_data.to(device), lengths.to(device)
                    
                if modality2 in ['ts']:
                    modality2_data, lengths = modality2_data
                    modality2_data, lengths = modality2_data.to(device), lengths.to(device)
                
                if modality1 not in ['ts', 'text_rad', 'text_ds']:
                    modality1_data = modality1_data.to(device)
                    
                if modality2 not in ['ts', 'text_rad', 'text_ds']:
                    modality2_data = modality2_data.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    feature1, feature2 = model(modality1_data, modality2_data, modality1, modality2, 
                                              attention_mask1, attention_mask2, lengths)
                    
                    # Calculate loss
                    loss1 = criterion(feature1, feature2)
                    loss2 = criterion(feature2, feature1)
                    loss = (loss1 + loss2) / 2
                    
                    # Backward pass and optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * modality1_data.size(0)
                    
                    # Calculate accuracy
                    top1, top5 = picme_utils.compute_accuracy(feature1, feature2)
                    running_top1_acc += top1
                    running_top5_acc += top5
                    num_batches += 1

            epoch_loss = running_loss / length
            epoch_top1_acc = running_top1_acc / num_batches
            epoch_top5_acc = running_top5_acc / num_batches
            
            print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Top-1 Accuracy: {epoch_top1_acc:.4f}, Top-5 Accuracy: {epoch_top5_acc:.4f}')

            if phase == 'val':
                wandb.log({"val_loss": epoch_loss, "val_acc1": epoch_top1_acc, "val_acc5": epoch_top5_acc})
                acc_dict[epoch] = epoch_top5_acc
                torch.save(model.state_dict(), path + "_current.pth")
                
                if epoch_top5_acc > best_acc:
                    best_acc = epoch_top5_acc
                    torch.save(model.state_dict(), path + "_best.pth")
                    
                if epoch % 5 == 0 or epoch == 0:
                    epoch_save_path = f"{path}_epoch_{epoch}.pth"
                    torch.save(model.state_dict(), epoch_save_path)
                    print(f"Model saved at epoch {epoch} to {epoch_save_path}")
                    
                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger += 1
                    if trigger >= patience:
                        return model
                else:
                    trigger = 0
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss, "train_acc1": epoch_top1_acc, "train_acc5": epoch_top5_acc, "epoch": epoch})

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model

def train_model_all_pairs(dataloaders_dict, criterion, len_train, len_val, config, path, args):
    """
    Train a contrastive model using all pairwise combinations of modalities.
    
    Args:
        dataloaders_dict: Dictionary containing train and val dataloaders
        criterion: Loss function (InfoNCE)
        len_train: Length of training dataset
        len_val: Length of validation dataset
        config: Configuration from wandb
        path: Path for saving model checkpoints
        args: Command line arguments
    
    Returns:
        Trained model
    """
    modalities = args.modalities
    picme_utils.set_seed(config.random_seed)
    
    ts_input_dim = _MODALITY_SHAPES['ts']
    demo_input_dim = _MODALITY_SHAPES['demo']
    projection_dim = config.projection_dim
    
    model = models.MultiModalContrastiveModel(
        ts_input_dim, 
        demo_input_dim, 
        projection_dim,
        num_modalities=len(modalities),
        ovo=False
    )
    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = config.epochs
    optimizer = picme_utils.build_optimizer(model, config.optimizer, config.learning_rate, 0.01)

    since = time.time()

    best_acc = 0.0
    patience = 5 
    trigger = 0
    acc_dict = {}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                length = len_train
                model.train()  
            else:
                length = len_val
                model.eval()  

            running_loss = 0.0
            running_top1_acc = 0.0
            running_top5_acc = 0.0
            running_top10_acc = 0.0
            num_batches = 0
            
            # For tracking pair weights
            num_pairs = len(list(combinations(range(len(modalities)), 2)))
            pair_weights_sum = torch.zeros(num_pairs, device=device)

            for batch in dataloaders_dict[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    modalities_data = []
                    modalities_type = []
                    mask_rad, mask_ds, ts_lengths = None, None, None
                    
                    for modality, data in zip(modalities, batch):
                        if modality == "text_rad":
                            data, mask_rad = data
                            mask_rad = mask_rad.to(device)
                        if modality == "text_ds":
                            data, mask_ds = data
                            mask_ds = mask_ds.to(device)
                        if modality == "ts":
                            data, ts_lengths = data
                            ts_lengths = ts_lengths.to(device)
                            
                        modalities_data.append(data.to(device))
                        modalities_type.append(modality)
                    
                    features = model(modalities_data, modalities_type, mask_rad, mask_ds, ts_lengths)
                    total_loss = torch.tensor(0.0, device=device)
                    
                    pair_weights = softmax(model.pair_weights, dim=0)
                    pair_weights_sum += pair_weights
                    
                    # Calculate loss for each pair
                    pair_idx = 0
                    for i in range(len(features)):
                        for j in range(i+1, len(features)):
                            loss_i_j = criterion(features[i], features[j])
                            loss_j_i = criterion(features[j], features[i])
                            pair_loss = (loss_i_j + loss_j_i) / 2
                            
                            # Weight the loss for this pair
                            weighted_loss = pair_weights[pair_idx] * pair_loss
                            total_loss += weighted_loss
                            pair_idx += 1
                    
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()
                    
                    running_loss += total_loss.item() * modalities_data[0].size(0)
                    
                    top1, top5, top10 = picme_utils.compute_multimodal_accuracy(*features)
                    running_top1_acc += top1
                    running_top5_acc += top5
                    running_top10_acc += top10
                    num_batches += 1

            epoch_loss = running_loss / length
            epoch_top1_acc = running_top1_acc / num_batches
            epoch_top5_acc = running_top5_acc / num_batches
            epoch_top10_acc = running_top10_acc / num_batches
            
            average_weights = pair_weights_sum / num_batches
            
            # Create a dictionary of pair weights for logging
            pair_idx = 0
            average_weights_dict = {}
            for i in range(len(modalities)):
                for j in range(i+1, len(modalities)):
                    pair_key = f"{modalities[i]}-{modalities[j]}"
                    average_weights_dict[pair_key] = average_weights[pair_idx].item()
                    pair_idx += 1
            
            print(average_weights_dict)
            print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Top-1 Accuracy: {epoch_top1_acc:.4f}, Top-5 Accuracy: {epoch_top5_acc:.4f}')

            if phase == 'val':
                wandb.log({"val_loss": epoch_loss, "val_acc1": epoch_top1_acc, "val_acc5": epoch_top5_acc})
                acc_dict[epoch] = epoch_top5_acc
                torch.save(model.state_dict(), path + "_current.pth")
                save_pair_weights(epoch, average_weights_dict, path, is_best=False)
                
                if epoch_top5_acc > best_acc:
                    best_acc = epoch_top5_acc
                    torch.save(model.state_dict(), path + "_best.pth")
                    save_pair_weights(epoch, average_weights_dict, path, is_best=True)
                    
                if epoch % 5 == 0 or epoch == 0:
                    epoch_save_path = f"{path}_epoch_{epoch}.pth"
                    torch.save(model.state_dict(), epoch_save_path)
                    print(f"Model saved at epoch {epoch} to {epoch_save_path}")
                    
                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger += 1
                    if trigger >= patience:
                        return model
                else:
                    trigger = 0
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss, "train_acc1": epoch_top1_acc, "train_acc5": epoch_top5_acc, "epoch": epoch})

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model

def train_model_ovo(dataloaders_dict, criterion, len_train, len_val, config, path, args):
    """
    Train a contrastive model using the OvO (One-vs-Others) approach.
    
    Args:
        dataloaders_dict: Dictionary containing train and val dataloaders
        criterion: Loss function (InfoNCE)
        len_train: Length of training dataset
        len_val: Length of validation dataset
        config: Configuration from wandb
        path: Path for saving model checkpoints
        args: Command line arguments
    
    Returns:
        Trained model
    """
    modalities = args.modalities
    picme_utils.set_seed(config.random_seed)
    
    ts_input_dim = _MODALITY_SHAPES['ts']
    demo_input_dim = _MODALITY_SHAPES['demo']
    projection_dim = config.projection_dim
    
    model = models.MultiModalContrastiveModel(
        ts_input_dim, 
        demo_input_dim, 
        projection_dim,
        num_modalities=len(modalities),
        ovo=True
    )
    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    num_epochs = config.epochs
    optimizer = picme_utils.build_optimizer(model, config.optimizer, config.learning_rate, 0.01)

    since = time.time()

    best_acc = 0.0
    patience = 5 
    trigger = 0
    acc_dict = {}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                length = len_train
                model.train()  
            else:
                length = len_val
                model.eval()  

            running_loss = 0.0
            running_top1_acc = 0.0
            running_top5_acc = 0.0
            running_top10_acc = 0.0
            num_batches = 0
            
            N_weights_sum = torch.zeros(len(modalities), device=device)

            for batch in dataloaders_dict[phase]:
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    modalities_data = []
                    modalities_type = []
                    mask_rad, mask_ds, ts_lengths = None, None, None
                    
                    for modality, data in zip(modalities, batch):
                        if modality == "text_rad":
                            data, mask_rad = data
                            mask_rad = mask_rad.to(device)
                        if modality == "text_ds":
                            data, mask_ds = data
                            mask_ds = mask_ds.to(device)
                        if modality == "ts":
                            data, ts_lengths = data
                            ts_lengths = ts_lengths.to(device)
                            
                        modalities_data.append(data.to(device))
                        modalities_type.append(modality)
                    
                    features = model(modalities_data, modalities_type, mask_rad, mask_ds, ts_lengths)
                    total_loss = torch.tensor(0.0, device=device)
                    
                    N_weights = softmax(model.N_weights, dim=0)
                    N_weights_sum += N_weights
                    
                    # One-vs-Others approach: each modality against all others
                    for i in range(len(features)):
                        one_vs_others_loss = 0
                        for j in range(len(features)):
                            if i != j:
                                loss = criterion(features[i], features[j])
                                one_vs_others_loss += loss
                        
                        # Weight the loss for this modality
                        weighted_loss = N_weights[i] * one_vs_others_loss
                        total_loss += weighted_loss
                        
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()
                    
                    running_loss += total_loss.item() * modalities_data[0].size(0)
                    
                    top1, top5, top10 = picme_utils.compute_multimodal_accuracy(*features)
                    running_top1_acc += top1
                    running_top5_acc += top5
                    running_top10_acc += top10
                    num_batches += 1

            epoch_loss = running_loss / length
            epoch_top1_acc = running_top1_acc / num_batches
            epoch_top5_acc = running_top5_acc / num_batches
            epoch_top10_acc = running_top10_acc / num_batches
            
            average_weights = N_weights_sum / num_batches
            average_weights_dict = {str(modality): weight.item() for modality, weight in zip(modalities, average_weights)}
            print(average_weights_dict)
             
            print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Top-1 Accuracy: {epoch_top1_acc:.4f}, Top-5 Accuracy: {epoch_top5_acc:.4f}')

            if phase == 'val':
                wandb.log({"val_loss": epoch_loss, "val_acc1": epoch_top1_acc, "val_acc5": epoch_top5_acc})
                acc_dict[epoch] = epoch_top5_acc
                torch.save(model.state_dict(), path + "_current.pth")
                save_pair_weights(epoch, average_weights_dict, path, is_best=False)
                
                if epoch_top5_acc > best_acc:
                    best_acc = epoch_top5_acc
                    torch.save(model.state_dict(), path + "_best.pth")
                    save_pair_weights(epoch, average_weights_dict, path, is_best=True)
                    
                if epoch % 5 == 0 or epoch == 0:
                    epoch_save_path = f"{path}_epoch_{epoch}.pth"
                    torch.save(model.state_dict(), epoch_save_path)
                    print(f"Model saved at epoch {epoch} to {epoch_save_path}")
                    
                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger += 1
                    if trigger >= patience:
                        return model
                else:
                    trigger = 0
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss, "train_acc1": epoch_top1_acc, "train_acc5": epoch_top5_acc, "epoch": epoch})

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


def main():
    args = pretrain_arg_parser()
    
    # Set random seeds
    picme_utils.set_seed(args.random_seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    config = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        notes=args.wandb_notes,
        tags=args.wandb_tags,
        config={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "projection_dim": args.projection_dim,
            "random_seed": args.random_seed,
            "pretraining_type": args.pretraining_type
        }
    ).config
    
    # Validate arguments based on pretraining type
    if args.pretraining_type == "single_pair":
        if not args.modality_pairs or len(args.modality_pairs) != 2:
            raise ValueError("Must specify exactly two modalities for single_pair pretraining")
    else:  # all_pairs or ovo
        if not args.modalities or len(args.modalities) < 2:
            raise ValueError("Must specify at least two modalities for all_pairs or ovo pretraining")
    
    # Load datasets based on pretraining type
    if args.pretraining_type == "single_pair":
        modality1, modality2 = args.modality_pairs
        train_dataset = load_dataset(modality1, modality2, split="train")
        val_dataset = load_dataset(modality1, modality2, split="val")
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
        )
        
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
        len_train, len_val = len(train_dataset), len(val_dataset)
        
    else:  # all_pairs or ovo
        modalities = args.modalities
        train_dataset = load_multimodal_dataset(modalities, split="train")
        val_dataset = load_multimodal_dataset(modalities, split="val")
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
        )
        
        dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
        len_train, len_val = len(train_dataset), len(val_dataset)
    
    # Initialize criterion
    criterion = InfoNCE(temperature=config.temperature)
    
    # Create checkpoint path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_name = f"{args.pretraining_type}_{timestamp}"
    if args.pretraining_type == "single_pair":
        checkpoint_name += f"_{args.modality_pairs[0]}_{args.modality_pairs[1]}"
    checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    
    # Train model based on pretraining type
    if args.pretraining_type == "single_pair":
        model = train_model_single_pair(dataloaders_dict, criterion, len_train, len_val, config, path, args)
    elif args.pretraining_type == "all_pairs":
        model = train_model_all_pairs(dataloaders_dict, criterion, len_train, len_val, config, path, args)
    elif args.pretraining_type == "ovo":
        model = train_model_ovo(dataloaders_dict, criterion, len_train, len_val, config, path, args)
    
    # Save final model
    torch.save(model.state_dict(), checkpoint_path + "_final.pth")
    print(f"Final model saved to {checkpoint_path}_final.pth")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
