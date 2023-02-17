from urllib.error import ContentTooShortError
import torch
from tqdm import tqdm
import torch.nn.functional as F
from model import CLIPModel


def get_model(path, device):
    model = CLIPModel(config)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def numpy_to_tensor(numpy_array, dim):
    tmp = []
    for tensor in numpy_array:
        tmp.append(tensor)
    return torch.stack(tmp, dim=dim)

def dict_to_tensor(indices, dict, trained_model, content_type, clip_version='clip-ViT-L-14'):
    collection = []
    for id in tqdm(indices):
        try:
            emb = torch.from_numpy(dict['data'][id][clip_version])
            emb = torch.unsqueeze(emb, 0)
            if trained_model:
                if content_type == 'text':
                    emb = trained_model.text_projection_head(emb)
                elif content_type == 'image':
                    emb = trained_model.image_projection_head(emb)
                else:
                    raise NotImplementedError
            collection.append(emb)
        except Exception as e:
            print('Problem with ', id)
            print('Exception: ', e)
            break
    # final_tensor = numpy_to_tensor(collection, dim=dim)
    final_tensor = torch.stack(collection, dim=1)
    final_tensor = torch.squeeze(final_tensor, dim=0)
    print('Final tensor shape:', final_tensor.shape)
    return final_tensor

def dict_to_list(indices, dict, trained_model, content_type, clip_version='clip-ViT-L-14'):
    list = []
    for id in tqdm(indices):
        try:
            emb = torch.from_numpy(dict['data'][id][clip_version])
            emb = torch.unsqueeze(emb, 0)
            if trained_model:
                if content_type == 'text':
                    emb = trained_model.text_projection_head(emb)
                elif content_type == 'image':
                    emb = trained_model.image_projection_head(emb)
                else:
                    raise NotImplementedError
            list.append(emb)
        except Exception as e:
            print('Problem with ', id)
            print('Exception: ', e)
    assert len(indices) == len(list)
    print(f'Got a list of {len(list)} embeddings')
    return list

def get_top_k_indices(query_emb, index, target_embeddings, k=5, normalized=True):
    """
    Given a query embedding, an index, a tensor of target embeddings, and a k 
    Retrieve indices of top-k items in target embeddings
    """
    if normalized:
        # normalize query and image embeddings
        query_embedding_n = F.normalize(query_emb, p=2, dim=-1)
        target_embeddings_n = F.normalize(target_embeddings, p=2, dim=-1)
    else:
        query_embedding_n = query_emb
        target_embeddings_n = target_embeddings

    # find top-k image embeddings
    dot_similarity = query_embedding_n @ target_embeddings_n.T
    values, indices = torch.topk(dot_similarity.squeeze(0), k)

    # map matches to indices
    matches = [index[idx] for idx in indices]

    return matches, values

def get_top_k_matches_for_queries(indices, queries_embs, target_tensor, k, normalized):
    retrieved_results = []
    for id, query_emb in tqdm(zip(indices, queries_embs)):
        matches, values = get_top_k_indices(
            query_emb=query_emb,
            index=indices,
            k=k,
            normalized=normalized,
            target_embeddings=target_tensor
        )
        retrieved_results.append(matches)
    assert len(indices) == len(queries_embs) == len(target_tensor)
    return retrieved_results


def recall_at_k(target, predictions, k=5, total_in_collection=1):
    if len(predictions) < k:
        print(f'Watch out, len(predictions) < k: {len(predictions)} < {k}')
    recall_at_k = predictions[:k].count(target) / total_in_collection
    return 100 * recall_at_k

def get_recalls(target, predictions, k_list=[1,5,10,50,100]):
    recalls = []
    for k in k_list:
        tmp_recall = recall_at_k(
            target=target,
            predictions=predictions,
            k=k,
            total_in_collection=1
            )
        recalls.append(tmp_recall)
    return recalls

def iterate_over_results_to_compute_recalls(ids, results, k_list=[1, 5, 10, 50, 100]):
    query_ids = []
    top_k_results = []
    recalls_at_1 = []
    recalls_at_5 = [] 
    recalls_at_10 = []
    recalls_at_50 = [] 
    recalls_at_100 = []
    for id, result in tqdm(zip(ids, results)):
        r_at_1, r_at_5, r_at_10, r_at_50, r_at_100 = get_recalls(
            id,
            result,
            k_list=k_list
            )
        query_ids.append(id)
        top_k_results.append(result)
        recalls_at_1.append(r_at_1)
        recalls_at_5.append(r_at_5)
        recalls_at_10.append(r_at_10)
        recalls_at_50.append(r_at_50)
        recalls_at_100.append(r_at_100)
    return query_ids, top_k_results, recalls_at_1, recalls_at_5, recalls_at_10, recalls_at_50, recalls_at_100


def get_mrr(target, predictions):
    if target in predictions:
        index = predictions.index(target) + 1
        mrr = 1 / index
        return mrr
    else:
        return 0.0