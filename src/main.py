"""This script trains and evaluates the CLIP model."""
import argparse
import os
import yaml
import torch
import logging as log
import pandas as pd
import itertools
from typing import Dict, Any

from dataset import ImageCaptionDataset, build_loader
from model import CLIPModel
from train_valid import train_epoch, valid_epoch
from utils import save_pkl, get_dt_string, load_pkl

def main(args: argparse.Namespace) -> None:
    """
    Main function to train and evaluate the CLIP model.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    dataset = args.dataset
    train_config_file = args.config_file
    is_tiny = args.is_tiny
    print(args)

    log.basicConfig(
        filename=f'logs/main_{dataset}.log',
        level=log.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    with open(train_config_file) as file:
        config_train = yaml.safe_load(file)
    print(config_train)
    log.info(f'Config train: {config_train}')
    
    torch.manual_seed(config_train["manual_seed"])
    config_train["device"] = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    if is_tiny:
        df_train_file = 'small_df_train.csv'
        df_test_file = 'small_df_test.csv'
        df_dev_file = 'small_df_dev.csv'
        config_train['epochs'] = 1
    else:
        df_train_file = 'df_train.csv'
        df_test_file = 'df_test.csv'
        df_dev_file = 'df_dev.csv'

    text_dict_path = os.path.join(
        config_train['DATASET'][dataset]['root'],
        config_train['DATASET'][dataset]['text_dict'],
    )
    image_dict_path = os.path.join(
        config_train['DATASET'][dataset]['root'],
        config_train['DATASET'][dataset]['image_dict'],
    )
    df_train_path = os.path.join(
        config_train['DATASET'][dataset]['root'],
        df_train_file,
    )
    df_dev_path = os.path.join(
        config_train['DATASET'][dataset]['root'],
        df_dev_file,
    )

    image_dict = load_pkl(image_dict_path)
    text_dict = load_pkl(text_dict_path)

    df_train = pd.read_csv(df_train_path, index_col=0)
    df_dev = pd.read_csv(df_dev_path, index_col=0)

    experiment_id = f'{dataset}_{config_train["epochs"]}E_{config_train["batch_size"]}BS'
    best_model_path = f'{config_train["deliverables_folder"]}/models/best_model_{experiment_id}.pt'
    loss_data_path = f'{config_train["deliverables_folder"]}/meta_data/loss_info_{experiment_id}'

    train_loader = build_loader(
        dataset=ImageCaptionDataset,
        config=config_train,
        dataf=df_train,
        text_dict=text_dict,
        image_dict=image_dict,
        mode='train'
    )
    valid_loader = build_loader(
        dataset=ImageCaptionDataset,
        config=config_train,
        dataf=df_dev,
        text_dict=text_dict,
        image_dict=image_dict,
        mode='test'
    )
    print('Built loaders!')
    log.info('Built loaders!')

    model = CLIPModel(config=config_train).to(config_train["device"])

    params = [
        {
            'params': itertools.chain(
                model.image_projection_head.parameters(),
                model.text_projection_head.parameters()
            ),
            'lr': config_train['head_lr'],
            'weight_decay': config_train['weight_decay']
        }
    ]
    optimizer = torch.optim.AdamW(params)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config_train["patience"],
        factor=config_train["factor"]
    )
    print('Created CLIP model')
    log.info('Created CLIP model')

    step = "epoch"
    train_loss_values = []
    valid_loss_values = []
    best_loss = float('inf')
    for epoch in range(config_train["epochs"]):
        running_loss = 0.0
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(config_train, model, train_loader, optimizer, lr_scheduler, step)
        train_loss_values.append(train_loss.avg)
        
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(config_train, model, valid_loader)
            valid_loss_values.append(valid_loss.avg)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved Best Model to {best_model_path}! best loss - {best_loss}")
            log.info(f"Saved Best Model! best loss - {best_loss}")

        lr_scheduler.step(valid_loss.avg)
    print('Finished training!')
    log.info('Finished training!')

    # Build loss dict, save loss values
    loss_values = {
        'train_loss': train_loss_values,
        'valid_loss': valid_loss_values
    }

    save_pkl(file=loss_values, path=loss_data_path)
    print('Finished running the script!')
    log.info('Finished running the script!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to train and evaluate the CLIP model."
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='CUB',
        choices=['CUB', 'ABO', 'fashion200k', 'coco', 'f30k'],
        help='Dataset type'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/train_conf.yaml',
        help='Configuration file'
    )
    parser.add_argument(
        '--is_tiny',
        type=bool,
        default=False,
        help='Use tiny version of the dataset'
    )
    args = parser.parse_args()
    main(args)
