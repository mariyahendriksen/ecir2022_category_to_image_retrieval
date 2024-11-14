"""This module contains the implementation of the CLIP model."""
import torch
from torch import nn
from torch.nn import functional as F
from utils import cross_entropy

class Projection(nn.Module):
    """
    A projection head for transforming input features.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128):
        """
        Initializes the Projection head.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output features.
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        x = self.model(x)
        return F.normalize(x, dim=1)


class CLIPModel(nn.Module):
    """
    A CLIP model for learning joint representations of images and text.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the CLIP model.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.image_input_dim = self.config["image_input_dim"]
        self.image_hidden_dim = self.config["image_hidden_dim"]
        self.text_input_dim = self.config["text_input_dim"]
        self.text_hidden_dim = self.config["text_hidden_dim"]
        self.projection_output_dim = self.config["projection_output_dim"]
        self.temperature = self.config["temperature"]

        # Initialize text and image encoders
        self.image_projection_head = Projection(
            input_dim=self.image_input_dim,
            hidden_dim=self.image_hidden_dim,
            output_dim=self.projection_output_dim
        )
        self.text_projection_head = Projection(
            input_dim=self.text_input_dim,
            hidden_dim=self.text_hidden_dim,
            output_dim=self.projection_output_dim
        )

    def cross_entropy(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            reduction: str = 'none'
        ) -> torch.Tensor:
        """
        Computes the cross-entropy loss.

        Args:
            preds (torch.Tensor): Predictions.
            targets (torch.Tensor): Targets.
            reduction (str): Reduction method ('none' or 'mean').

        Returns:
            torch.Tensor: Computed loss.
        """
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def symmetric_contrastive_loss(
            self,
            image_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor
        ) -> torch.Tensor:
        """
        Computes the symmetric contrastive loss.

        Args:
            image_embeddings (torch.Tensor): Image embeddings.
            text_embeddings (torch.Tensor): Text embeddings.

        Returns:
            torch.Tensor: Computed loss.
        """
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        texts_similarity = text_embeddings @ text_embeddings.T
        images_similarity = image_embeddings @ image_embeddings.T
        targets = F.softmax(
            (texts_similarity + images_similarity) / 2 * self.temperature,
            dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CLIP model.

        Args:
            batch (torch.Tensor): Input batch, contains image and text features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and text embeddings.
        """
        # Extract feature representations of each modality
        image_embeddings = self.image_projection_head(batch[0])
        text_embeddings = self.text_projection_head(batch[1])

        return image_embeddings, text_embeddings
