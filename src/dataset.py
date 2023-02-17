import torch
from utils import seed_worker


class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataf, text_dict, image_dict) -> None:
        self.config = config
        self.dataf = dataf
        self.text_dict = text_dict
        self.image_dict = image_dict

    def __getitem__(self, idx):
        key = self.dataf.iloc[idx].name
        image = self.image_dict["data"][key][self.config["clip_version"]]
        text = self.text_dict["data"][key][self.config["clip_version"]]

        return image, text

    def __len__(self):
        return self.dataf.shape[0]



def build_loader(dataset, config, dataf, text_dict, image_dict, mode='train'):
    # initialize a dataset
    dataset = dataset(
        config=config,
        dataf=dataf,
        text_dict=text_dict,
        image_dict=image_dict
    )
    # we use generator for reproducibility
    g = torch.Generator()
    g.manual_seed(config["manual_seed"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True if mode == 'train' else False,
        num_workers=config["num_workers"],
        worker_init_fn=seed_worker,
        generator=g
    )

    return dataloader