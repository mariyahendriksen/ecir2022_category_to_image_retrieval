import argparse
from ast import parse
import yaml
import pandas as pd
import os
from utils import make_train_test_dev_dfs

def main(args):
    DATASET = args.dataset
    is_tiny = args.is_tiny
    config_file = args.config_file

    print(args)

    # load config file
    with open(config_file) as file:
        config_full = yaml.safe_load(file)
    # config_full = config_full['DATASET']
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


    # load df
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(config['dataset_root'], csv_file),
        dtype=config['columns_dtypes'],
        index_col=0
        )
    df.head()
    print(f'Loaded df:{df.head()}\ndf shape: {df.shape}')

    df_train, df_test, df_dev = make_train_test_dev_dfs(df)
    print('Split df into train, test, dev')

    # save dfs
    df_train.to_csv(
        path_or_buf=os.path.join(config["dataset_root"], df_train_file)
        )
    df_test.to_csv(
        path_or_buf=os.path.join(config["dataset_root"], df_test_file)
        )
    df_dev.to_csv(
        path_or_buf=os.path.join(config["dataset_root"], df_dev_file)
        )
    print('Saved df train, test, dev to ', config['dataset_root'])
    print('Done with the script')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='CUB',
                        choices=['CUB', 'ABO', 'fashion200k', 'coco', 'f30k'],
                        help='dataset type')
    parser.add_argument('--is_tiny', type=bool, default=False)
    parser.add_argument('--config_file', type=str, default='conf/data_conf.yaml',
                    help='Configuration file')
    args = parser.parse_args()
    main(args)