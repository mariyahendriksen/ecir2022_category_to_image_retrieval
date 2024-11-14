"""This module contains classes for loading video and text datasets."""

import argparse
import os
from typing import Dict, Any

import pandas as pd
import yaml

from utils import make_train_test_dev_dfs

def main(args: argparse.Namespace) -> None:
    """
    Main function to load dataset, split into train, test, dev sets, and save them.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    DATASET = args.dataset
    is_tiny = args.is_tiny
    config_file = args.config_file

    print(args)

    # Load config file
    with open(config_file) as file:
        config_full = yaml.safe_load(file)
    config = config_full[DATASET]
    print('Loaded configuration: ', config)

    if is_tiny:
        csv_file = config['csv_file_small']
        df_train_file = 'small_df_train.csv'
        df_test_file = 'small_df_test.csv'
        df_dev_file = 'small_df_dev.csv'
    else:
        csv_file = config['csv_file']
        df_train_file = 'df_train.csv'
        df_test_file = 'df_test.csv'
        df_dev_file = 'df_dev.csv'

    # Load dataframe
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(config['dataset_root'], csv_file),
        dtype=config['columns_dtypes'],
        index_col=0
    )
    print(f'Loaded df:\n{df.head()}\ndf shape: {df.shape}')

    # Split dataframe into train, test, dev sets
    df_train, df_test, df_dev = make_train_test_dev_dfs(df)
    print('Split df into train, test, dev')

    # Save dataframes
    df_train.to_csv(os.path.join(config["dataset_root"], df_train_file))
    df_test.to_csv(os.path.join(config["dataset_root"], df_test_file))
    df_dev.to_csv(os.path.join(config["dataset_root"], df_dev_file))
    print('Saved df train, test, dev to ', config['dataset_root'])
    print('Done with the script')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to load dataset, split into train, test, dev sets, and save them."
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='CUB',
        choices=['CUB', 'ABO', 'fashion200k', 'coco', 'f30k'],
        help='Dataset type'
    )
    parser.add_argument(
        '--is_tiny',
        type=bool,
        default=False,
        help='Use tiny version of the dataset'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='conf/data_conf.yaml',
        help='Configuration file'
    )
    args = parser.parse_args()
    main(args)
