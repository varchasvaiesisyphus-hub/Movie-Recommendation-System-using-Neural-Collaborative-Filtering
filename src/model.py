"""
model.py

Neural Collaborative Filtering (NeuMF) model definition.

WHY THIS MODULE EXISTS:
NeuMF (He et al., "Neural Collaborative Filtering", WWW 2017) improves on
plain matrix factorization by combining two complementary sub-models:

  1. GMF (Generalized Matrix Factorization): a linear, element-wise
     product of user/item embeddings -- captures simple, low-rank
     collaborative signal, similar to classic matrix factorization.

  2. MLP (Multi-Layer Perceptron): a stack of fully-connected layers over
     concatenated user/item embeddings -- captures higher-order,
     non-linear interactions that a linear dot product cannot express.

Fusing both branches' final representations before the output layer lets
the model learn both linear and non-linear user-item interaction patterns,
which is why NeuMF consistently outperforms either sub-model alone.
"""

from typing import List

import torch
import torch.nn as nn

from config import model_cfg


class GMF(nn.Module):
    """
    Generalized Matrix Factorization branch.

    Computes the element-wise (Hadamard) product of user and item
    embeddings, which is the generalization of classic matrix
    factorization's dot-product similarity.
    """

    def __init__(self, num_users: int, num_movies: int, embedding_dim: int) -> None:
        """
        Initialize GMF user/item embedding tables.

        Args:
            num_users: total number of unique encoded users.
            num_movies: total number of unique encoded movies.
            embedding_dim: dimensionality of the latent embedding space.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # WHY explicit init: normal initialization with a small std keeps
        # embeddings near zero at the start of training, which empirically
        # stabilizes early-epoch gradients for embedding-based models.
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the GMF interaction vector for a batch of (user, movie) pairs.

        Args:
            user_ids: LongTensor of shape (batch_size,).
            movie_ids: LongTensor of shape (batch_size,).

        Returns:
            torch.Tensor: element-wise product of embeddings, shape
            (batch_size, embedding_dim).
        """
        user_vec = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)
        return user_vec * movie_vec


class MLP(nn.Module):
    """
    Multi-Layer Perceptron branch.

    Concatenates user and item embeddings and passes them through a stack
    of fully-connected layers with ReLU activations and dropout, allowing
    the model to learn non-linear interaction patterns that a simple
    dot-product (as in GMF) cannot capture.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int,
        hidden_layers: List[int],
        dropout: float,
    ) -> None:
        """
        Initialize MLP user/item embeddings and the fully-connected stack.

        Args:
            num_users: total number of unique encoded users.
            num_movies: total number of unique encoded movies.
            embedding_dim: dimensionality of the latent embedding space
                (kept separate from GMF's embeddings, per the NeuMF paper,
                so each branch can learn representations suited to its
                own interaction function).
            hidden_layers: sizes of the fully-connected layers, e.g.
                [128, 64, 32].
            dropout: dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)

        # WHY build layers dynamically from a list: this lets `config.py`
        # define the architecture (e.g. [128, 64, 32]) without hardcoding
        # layer definitions here, keeping the network shape configurable.
        layers = []
        input_dim = embedding_dim * 2  # concatenated user + movie embeddings
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.mlp_stack = nn.Sequential(*layers)
        self.output_dim = input_dim  # dimensionality of the final MLP layer output

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the MLP interaction vector for a batch of (user, movie) pairs.

        Args:
            user_ids: LongTensor of shape (batch_size,).
            movie_ids: LongTensor of shape (batch_size,).

        Returns:
            torch.Tensor: output of the final MLP hidden layer, shape
            (batch_size, hidden_layers[-1]).
        """
        user_vec = self.user_embedding(user_ids)
        movie_vec = self.movie_embedding(movie_ids)
        concatenated = torch.cat([user_vec, movie_vec], dim=-1)
        return self.mlp_stack(concatenated)


class NeuMF(nn.Module):
    """
    Neural Collaborative Filtering model: fusion of GMF and MLP branches.

    Architecture:
        GMF branch  -> (batch, embedding_dim)
        MLP branch  -> (batch, mlp_layers[-1])
        Concatenate -> (batch, embedding_dim + mlp_layers[-1])
        Dense       -> (batch, 1)
        Sigmoid     -> (batch, 1)  probability of positive interaction

    WHY fuse at the final layer rather than sharing embeddings between
    branches: the original NeuMF paper found that letting GMF and MLP
    learn separate embedding spaces (rather than sharing one) gives each
    branch the freedom to optimize its embeddings for its own interaction
    function (linear product vs. non-linear MLP), improving overall
    predictive performance.
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int = None,
        mlp_layers: List[int] = None,
        dropout: float = None,
    ) -> None:
        """
        Initialize the full NeuMF model.

        Args:
            num_users: total number of unique encoded users (from
                preprocessing.py's LabelEncoder).
            num_movies: total number of unique encoded movies.
            embedding_dim: GMF and MLP embedding dimensionality. Defaults
                to `config.model_cfg.embedding_dim` (64).
            mlp_layers: hidden layer sizes for the MLP branch. Defaults to
                `config.model_cfg.mlp_layers` ([128, 64, 32]).
            dropout: dropout probability in the MLP branch. Defaults to
                `config.model_cfg.dropout` (0.2).
        """
        super().__init__()

        embedding_dim = embedding_dim if embedding_dim is not None else model_cfg.embedding_dim
        mlp_layers = mlp_layers if mlp_layers is not None else model_cfg.mlp_layers
        dropout = dropout if dropout is not None else model_cfg.dropout

        self.gmf = GMF(num_users, num_movies, embedding_dim)
        self.mlp = MLP(num_users, num_movies, embedding_dim, mlp_layers, dropout)

        # WHY this input size: concatenation of GMF's embedding_dim-sized
        # output and MLP's final hidden layer output.
        fusion_input_dim = embedding_dim + self.mlp.output_dim
        self.output_layer = nn.Linear(fusion_input_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # Store architecture metadata for checkpointing/reconstruction.
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers
        self.dropout = dropout

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict the probability of a positive interaction.

        Args:
            user_ids: LongTensor of shape (batch_size,).
            movie_ids: LongTensor of shape (batch_size,).

        Returns:
            torch.Tensor: predicted probabilities, shape (batch_size,),
            each value in [0, 1].
        """
        gmf_output = self.gmf(user_ids, movie_ids)
        mlp_output = self.mlp(user_ids, movie_ids)

        # WHY concatenate rather than average/sum: concatenation preserves
        # both branches' full representational content, letting the final
        # dense layer learn how to weigh linear vs. non-linear signals
        # rather than forcing an equal, fixed blend.
        fused = torch.cat([gmf_output, mlp_output], dim=-1)
        logits = self.output_layer(fused)
        probabilities = self.sigmoid(logits)

        return probabilities.squeeze(-1)

    def get_config(self) -> dict:
        """
        Return the architecture hyperparameters needed to reconstruct
        this model from a checkpoint.

        WHY: `train.py` saves this alongside model weights so that
        `evaluate.py` and `recommend.py` can rebuild an identical
        architecture before loading the state dict, without depending on
        `config.py` still holding the exact same values used at train time.

        Returns:
            dict: architecture hyperparameters.
        """
        return {
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "embedding_dim": self.embedding_dim,
            "mlp_layers": self.mlp_layers,
            "dropout": self.dropout,
        }


if __name__ == "__main__":
    # Quick sanity check: build a tiny NeuMF and run a forward pass.
    model = NeuMF(num_users=100, num_movies=200)
    dummy_users = torch.randint(0, 100, (16,))
    dummy_movies = torch.randint(0, 200, (16,))
    predictions = model(dummy_users, dummy_movies)
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5].detach().numpy()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
