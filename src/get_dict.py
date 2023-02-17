import yaml
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image
import pickle
import argparse


def get_image_emb(config, model, img_file):
    img_abs_path = os.path.join(config['dataset_root'], config['image_folder'], img_file)
    img_emb = model.encode(Image.open(img_abs_path))
    return img_emb

def get_text_emb(model, caption, add_prefix):
    prefix = 'a photo of a '
    if add_prefix:
        caption = prefix + caption
    
    cutoff_point = 150
    if len(caption) > cutoff_point:
        caption = caption[:cutoff_point]
    try:
        txt_emb = model.encode(caption)
        return txt_emb
    except Exception as e:
        print('Problem with caption: ', caption)
        print('Exception: ', e)

def main(args):
    CONTENT_TYPE = args.content_type # 'image' or 'text'
    DATASET = args.dataset
    config_file = args.config_file
    add_prefix = args.add_prefix
    print(f'Working with {DATASET}, CONTENT_TYPE:{CONTENT_TYPE}')

    # load config file
    with open(config_file) as file:
        config_full = yaml.safe_load(file)
    config = config_full[DATASET]
    print('Loaded configuration: ', config)
    # assert DATASET == config['dataset']

    # load df
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(config['dataset_root'], config['csv_file']),
        dtype=config['columns_dtypes'],
        index_col=0
        )
    df.head()
    print(f'Loaded df:{df.head()}\ndf shape: {df.shape}')

    # load embedding model
    model = SentenceTransformer(config['clip_version'])
    print(f'Loaded {config["clip_version"]} of clip')

    # collect id-file pairs
    idx = df.index.tolist()
    files = df[config['content_type'][CONTENT_TYPE]]
    assert len(idx) == len(files)

    # create a dictionary
    my_dict = dict()
    my_dict['meta'] = f"{CONTENT_TYPE} FEATURES FOR {DATASET} dataset"
    my_dict['data'] = {}
    print('Initialized a dictionary with this meta: ', my_dict['meta'])

    for id, file in tqdm(zip(idx, files)):
        if id not in my_dict['data']:
            my_dict['data'][id] = {}
            if CONTENT_TYPE == 'image':
                emb = get_image_emb(config, model, file)
            elif CONTENT_TYPE == 'text':
                emb = get_text_emb(model, file, add_prefix)
            my_dict['data'][id][config['clip_version']] = emb
        else:
            print(f'{id} is already in the dict: {my_dict["data"][id]}')

    dict_path = os.path.join(config["dataset_root"], f'{CONTENT_TYPE}_prefix{add_prefix}_dict.pkl')
    with open(dict_path, 'wb+') as f:
        pickle.dump(my_dict, f)
        print('Saved the file to: ', dict_path)
    
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./CLIP/conf/data_conf.yaml',
                        help='Configuration file')
    parser.add_argument('--content_type', type=str, default='image',
                        choices=['image', 'text'],
                        help='Modality type')
    parser.add_argument('--add_prefix', type=bool, default=False,
                        help='Add "a photo of a " prefix (text data only)?')
    parser.add_argument('--dataset', type=str,
                        default='ABO',
                        choices=['CUB', 'ABO', 'fashion200k', 'coco', 'f30k'],
                        help='dataset type')
    args = parser.parse_args()
    main(args)
