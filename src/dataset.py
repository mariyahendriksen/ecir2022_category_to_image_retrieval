from typing import Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils import seed_worker

class ImageCaptionDataset(Dataset):
    """
    A dataset class for image-caption pairs.
    """
    def __init__(
            self,
            config: Dict[str, Any],
            dataf,
            text_dict: Dict[str, Any],
            image_dict: Dict[str, Any]
        ) -> None:
        """
        Initializes the dataset with the given configuration, data frame,
        text dictionary, and image dictionary.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            dataf: Data frame containing the dataset information.
            text_dict (Dict[str, Any]): Dictionary containing text data.
            image_dict (Dict[str, Any]): Dictionary containing image data.
        """
        self.config = config
        self.dataf = dataf
        self.text_dict = text_dict
        self.image_dict = image_dict

        # Validate config
        required_keys = [
            "clip_version", "manual_seed", "batch_size", "num_workers"
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

    def __getitem__(self, idx: int):
        """
        Retrieves the image and text pair at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple: A tuple containing the image and text.
        """
        key = self.dataf.iloc[idx].name
        image = self.image_dict["data"][key][self.config["clip_version"]]
        text = self.text_dict["data"][key][self.config["clip_version"]]

        return image, text

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return self.dataf.shape[0]

def build_loader(
        dataset: Dataset,
        config: Dict[str, Any],
        dataf: pd.DataFrame,
        text_dict: Dict[str, Any],
        image_dict: Dict[str, Any],
        mode: str = 'train'
    ) -> DataLoader:
    """
    Builds a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset class to use.
        config (Dict[str, Any]): Configuration dictionary.
        dataf (pd.DataFrame): Data frame containing the dataset information.
        text_dict (Dict[str, Any]): Dictionary containing text data.
        image_dict (Dict[str, Any]): Dictionary containing image data.
        mode (str): Mode of the DataLoader ('train' or 'test').

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    # Initialize the dataset
    dataset = dataset(
        config=config,
        dataf=dataf,
        text_dict=text_dict,
        image_dict=image_dict
    )

    # Use generator for reproducibility
    g = torch.Generator()
    g.manual_seed(config["manual_seed"])

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=(mode == 'train'),
        num_workers=config["num_workers"],
        worker_init_fn=seed_worker,
        generator=g
    )

    return dataloader
