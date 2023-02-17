import argparse, yaml, os
import torch
import pandas as pd
from model import CLIPModel
from utils import load_pkl, save_pkl
from evaluation_utils import dict_to_tensor, dict_to_list, get_top_k_matches_for_queries, iterate_over_results_to_compute_recalls
from tqdm import tqdm
import pickle

def main(args):
    config_file = args.config_file
    ds_config_file = args.ds_config_file
    dataset = args.dataset
    is_zero_shot = args.is_zero_shot
    add_prefix = args.add_prefix
    # df_test_path = '/Users/mhendriksen/Desktop/repositories/CLIP/data/CUB/df_test.csv'

    print('Loading config file')
    # dataset_config_file = '/Users/mhendriksen/Desktop/repositories/CLIP/conf/local_cub_conf.yaml'
    with open(ds_config_file) as file:
        config_ds = yaml.safe_load(file)
    with open(config_file) as file:
        config = yaml.safe_load(file)
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_zero_shot:
        print('Zero-shot set-up, no need to load the model')
        model = None
        model_path = f'zero_shot_{dataset}'
    else:
        print(f'Non-zero-shot setup, loading pre-trained model froom {config["DATASET"][dataset]["model_path"]}...')
        model_path = config['DATASET'][dataset]["model_path"]
        model = CLIPModel(config)
        model.load_state_dict(torch.load(config['DATASET'][dataset]["model_path"], map_location=config["device"]))
        model.eval()
        # model = get_model(path=config['DATASET'][dataset]["model_path"], device=config["device"])

    print('Loading text and image dicts...')
    image_dict_path = os.path.join(
        config['DATASET'][dataset]['root'],
        config['DATASET'][dataset]['image_dict']
        )

    if add_prefix:
        text_dict_file = config['DATASET'][dataset]['text_dict_prefix']
    else:
        text_dict_file = config['DATASET'][dataset]['text_dict']
    text_dict_path = os.path.join(
        config['DATASET'][dataset]['root'],
        text_dict_file,
    )
    # text_dict_path = '/Users/mhendriksen/Desktop/repositories/datasets/CUB_200_2011/text_dict.pkl'
    # image_dict_path = '/Users/mhendriksen/Desktop/repositories/datasets/CUB_200_2011/image_dict.pkl'

    image_dict = load_pkl(image_dict_path)
    text_dict = load_pkl(text_dict_path)

    print('Loading the dataset...')
    df_test_path = os.path.join(
    config['DATASET'][dataset]['root'],
    config['DATASET'][dataset]['df_test'],
    )
    # df_test_path = '/Users/mhendriksen/Desktop/repositories/CLIP/data/CUB/df_test.csv'
    df_test = pd.read_csv(df_test_path, index_col=0)

    print('Get ids, images and captions from dev df... ')
    ids = df_test.index.tolist()
    images = df_test[config_ds[dataset]['content_type']['image']].tolist()
    captions = df_test[config_ds[dataset]['content_type']['text']].tolist()
    assert len(images) == len(captions)
    print(f'Got {len(images)} image-caption pairs')

    print('Text to image retrieval results...')
    print('Build a list of textual query embeddings and an images tensor...')
    text_query_embs = dict_to_list(
        indices=ids,
        dict=text_dict,
        trained_model=model,
        content_type='text'
    )
    images_tensor = dict_to_tensor(
        indices=ids,
        dict=image_dict,
        trained_model=model,
        content_type='image',
        clip_version='clip-ViT-L-14'
    )
    assert len(text_query_embs) == len(images_tensor)

    k=1000
    print(f'For each text query, retrieve top k={k} results...')
    text_to_image_results = get_top_k_matches_for_queries(
        indices=ids,
        queries_embs=text_query_embs,
        target_tensor=images_tensor,
        k=k,
        normalized=True
    )
    assert len(text_to_image_results) == len(ids) == len(images_tensor)

    print('Calculating recall@k for text-to-image results...')
    t2i_query_ids, t2i_top_k_results, t2i_recalls_at_1, t2i_recalls_at_5, t2i_recalls_at_10, t2i_recalls_at_50, t2i_recalls_at_100 = iterate_over_results_to_compute_recalls(
        ids=ids,
        results=text_to_image_results,
        k_list=[1, 5, 10, 50, 100]
    )

    t2i_results = pd.DataFrame(
        data={
        't2i_query_ids': t2i_query_ids,
            # text to image
            't2i_top_k_results': t2i_top_k_results,
            't2i_recalls_at_1': t2i_recalls_at_1,
            't2i_recalls_at_5': t2i_recalls_at_5,
            't2i_recalls_at_10': t2i_recalls_at_10,
            't2i_recalls_at_50': t2i_recalls_at_50,
            't2i_recalls_at_100': t2i_recalls_at_100,
        }
    )

    print('Text-to-image results:\n', t2i_results.describe())

    print('###############################')
    print('Image to text retrieval...')
    print('Build a list of visual query embeddings and a captions tensor...')
    image_query_embs = dict_to_list(
        indices=ids,
        dict=image_dict,
        trained_model=model,
        content_type='image'
    )
    text_tensor = dict_to_tensor(
        indices=ids,
        dict=text_dict,
        trained_model=model,
        content_type='text',
        clip_version='clip-ViT-L-14'
    )
    k=1000
    print(f'For each image query, retrieve top k={k} captions...')
    image_to_text_results = get_top_k_matches_for_queries(
        indices=ids,
        queries_embs=image_query_embs,
        target_tensor=text_tensor,
        k=k,
        normalized=True
    )
    assert len(image_to_text_results) == len(ids) == len(text_tensor)
    print('Calculating recall@k for image-to-text results...')
    i2t_query_ids, i2t_top_k_results, i2t_recalls_at_1, i2t_recalls_at_5, i2t_recalls_at_10, i2t_recalls_at_50, i2t_recalls_at_100 = iterate_over_results_to_compute_recalls(
        ids=ids,
        results=image_to_text_results,
        k_list=[1, 5, 10, 50, 100]
    )
    i2t_results = pd.DataFrame(
        data={
        'i2t_query_ids': i2t_query_ids,
            # text to image
            'i2t_top_k_results': i2t_top_k_results,
            'i2t_recalls_at_1': i2t_recalls_at_1,
            'i2t_recalls_at_5': i2t_recalls_at_5,
            'i2t_recalls_at_10': i2t_recalls_at_10,
            'i2t_recalls_at_50': i2t_recalls_at_50,
            'i2t_recalls_at_100': i2t_recalls_at_100,
        }
    )

    print('Image-to-text results:\n', i2t_results.describe())

    print('Saving results')
    if '/' in model_path:
        evaluation_id = f'{os.path.basename(os.path.splitext(model_path)[0])}'
    else:
        evaluation_id = model_path
    if add_prefix:
        evaluation_id += f'_prefix{add_prefix}'
    i2t_results_path = f'{config["deliverables_folder"]}/results/{evaluation_id}_image_to_text.pkl'
    t2i_results_path = f'{config["deliverables_folder"]}/results/{evaluation_id}_text_to_image.pkl'
    save_pkl(file=i2t_results, path=i2t_results_path)
    save_pkl(file=t2i_results, path=t2i_results_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='CLIP/conf/train_conf.yaml',
                    help='Configuration file')
    parser.add_argument('--ds_config_file', type=str, default='CLIP/conf/data_conf.yaml',
                    help='Configuration file')
    parser.add_argument('--dataset', type=str,
                        default='CUB',
                        choices=['CUB', 'ABO', 'fashion200k', 'coco', 'f30k'],
                        help='dataset type')
    parser.add_argument('--add_prefix', type=bool, default=False,
                        help='Add "a photo of a " prefix (text data only)?')
    parser.add_argument('--is_zero_shot', type=bool, default=False,
                        help='Do we use CLIP in zero-shot way?')
    args = parser.parse_args()
    main(args)
