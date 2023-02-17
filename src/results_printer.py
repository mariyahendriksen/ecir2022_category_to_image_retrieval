import argparse
import yaml
import pandas as pd
import os
from utils import load_pkl

def subset_df(dataf):
    target_columns = dataf.columns.tolist()[2:5]
    return dataf[target_columns]


def main(args):
    DATASET = args.dataset
    results_root = args.results_root
    is_zero_shot = args.is_zero_shot
    # print(f'Working with {DATASET}; zero-shot? {is_zero_shot}')

    files = os.listdir(results_root)
    files_filtered = [file for file in files if DATASET in file]

    if is_zero_shot:
        files_filtered = [file for file in files_filtered if 'zero_shot' in file]

    print('Working with the following files: ', files_filtered)

    for file in files_filtered:
        path = os.path.join(results_root, file)
        df = load_pkl(path)

        df_subset = subset_df(df)
        # print(df_subset.describe())
        print(df_subset.mean())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='CUB',
                        choices=['CUB', 'ABO', 'fashion200k', 'coco', 'f30k'],
                        help='dataset type')
    parser.add_argument('--results_root', type=str, default='CLIP/data/results',
                    help='Configuration file')
    parser.add_argument('--is_zero_shot', type=bool, default=False,
                        help='Do we use CLIP in zero-shot way?')
    args = parser.parse_args()
    main(args)