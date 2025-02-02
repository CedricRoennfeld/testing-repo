import math

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict


def initialize_parameter(
    shape: tuple,
    pretrained_parameter: Tensor,
    initializer: Callable[[Tensor], Tensor]
) -> Tensor:
    r"""
    Small function to shorten parameter initialization

    Parameter:
        shape: shape of tensor, that should be created
        pretrained_parameter: pretrained parameter if it exists
        initializer: initialization function for the parameter
    """
    if pretrained_parameter is None:
        return initializer(torch.empty(shape))
    else:
        assert pretrained_parameter.shape == shape, "Shape mismatch"
        return pretrained_parameter


class Embedding(nn.Module):
    """
    Lookup table for fixed number of learnable embeddings of fixed
    dimension.

    Attributes:
        length (int): amount of embeddings
        dimension (int): dimension of each embedding
    """
    def __init__(
        self,
        length: int,
        dimension: int,
        pretrained_embedding: Tensor = None,
    ) -> None:
        super().__init__()
        self.length = length
        self.dimension = dimension
        # initialize embedding as learnable parameter with
        # standard normal distribution as start values
        self.embedding = nn.Parameter(initialize_parameter(
            (length, dimension),
            pretrained_embedding,
            lambda x: nn.init.normal_(x,std=1, mean=0))
        )

    # return scaled embedding
    def forward(self, x:Tensor) -> Tensor:
        r"""
        Maps an integer tensor with entries between 0 and length-1 to
        its corresponding embedding vector of dimension D
        Parameter:
            x (Tensor): integer tensor of shape (*) with values between
                0 and length-1
        Return:
            Tensor of shape (*, D) where D = dimension
        """
        return self.embedding[x] * math.sqrt(self.dimension)


class LayerNorm(nn.Module):
    """
    Normalize input from previous layer with addition learnable
    linear transformation on each value

    Attributes:
        eps (float): small increase of variance to avoid division
            by zero
    """
    def __init__(
        self,
        eps: float = 1e-05,
        weight: Tensor = None,
        bias: Tensor = None
    ) -> None:
        super().__init__()
        self.eps = eps
        # initialize linear transformation as identity transformation
        self.weight = nn.Parameter(
            initialize_parameter((1,), weight, lambda x: torch.ones((1,)))
        )
        self.bias = nn.Parameter(
            initialize_parameter((1,), bias, lambda x: torch.zeros((1,)))
        )

    # applies layer normalization on last dimension
    def forward(self, x:Tensor) -> Tensor:
        r"""
        Parameter:
            x (Tensor): tensor of shape (*)
        Return:
            Tensor of shape (*)
        """
        # calculate mean and variance of last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        # layer norm: (x-mean)/std
        # add small eps to prevent division by zero
        return self.weight * ((x - mean) / torch.sqrt(var + self.eps)
            + self.bias)


# randomly puts elements of x to zero with probability p
def dropout(x:Tensor, p:float = .5) -> Tensor:
    r"""
    Randomly puts elements to zero
    Parameter:
        x (Tensor): tensor of shape (*)
        p (float): probability of dropping a value
    Return:
        Tensor of shape (*)
    """
    # applies bernoulli distribution (0 or 1) on a tensor filled with
    # p values to create a mask
    mask = torch.bernoulli(p * torch.ones_like(x))
    # elementwise multiplication to drop values
    return torch.mul(x,mask)


class FeedForward(nn.Module):
    """
    Fully connected Feed-Forward network with two linear
    transformations and a ReLU activation layer

    Parameter:
        outer_dimension (int): dimension of vectors
            going in and coming out
        inner_dimension (int): dimension after first transformation
    """
    def __init__(
        self,
        outer_dimension: int,
        inner_dimension: int,
        weight1: Tensor = None,
        weight2: Tensor = None,
        bias1: Tensor = None,
        bias2: Tensor = None
    ) -> None:
        super().__init__()
        # initialize both linear layer using a uniform distribution
        self.weight1 = nn.Parameter(initialize_parameter(
            (inner_dimension, outer_dimension),
            weight1,
            lambda x: nn.init.uniform_(
                x,
                -1/math.sqrt(outer_dimension),
                1/math.sqrt(outer_dimension)
            )
        ))
        self.bias1 = nn.Parameter(initialize_parameter(
            (inner_dimension,1),
            bias1,
            lambda x: nn.init.uniform_(
                x,
                -1/math.sqrt(outer_dimension),
                1/math.sqrt(outer_dimension)
            )
        ))
        self.weight2 = nn.Parameter(initialize_parameter(
            (outer_dimension, inner_dimension),
            weight2,
            lambda x: nn.init.uniform_(
                x,
                -1/math.sqrt(inner_dimension),
                1/math.sqrt(inner_dimension)
            )
        ))
        self.bias2 = nn.Parameter(initialize_parameter(
            (outer_dimension,1),
            bias2,
            lambda x: nn.init.uniform_(
                x,
                -1/math.sqrt(inner_dimension),
                1/math.sqrt(inner_dimension)
            )
        ))

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Applies the linear transformations with an ReLU between

        Parameter:
            x (Tensor): tensor of shape (*)
        Return:
            Tensor of shape (*)
        """
        # (bs, sl, d) -> (bs,d,sl)
        # transpose at start and end, because excepted input and output
        # are of shape (*, S, D) but the transformation should be
        # applied to each embedding individually
        x = x.transpose(-1, -2)
        x = self.weight2 @ FeedForward.relu(
            self.weight1 @ x + self.bias1
        ) + self.bias2
        return x.transpose(-1, -2)

    @staticmethod
    def relu(x: Tensor) -> Tensor:
        r"""
        applies Rectifier Linear Unit (maps negative values to zero)
        Parameter:
            x (Tensor): tensor of shape (*)
        Return:
            Tensor of shape (*)
        """
        return torch.max(x, torch.zeros_like(x))

class MaskedMultiHeadAttention(nn.Module):
    """

    """
    def __init__(
        self,
        dimension: int,
        heads_amount: int,
        weight: Dict[str,Tensor] = None,
        bias: Dict[str,Tensor] = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.heads_amount = heads_amount
        assert self.dimension % self.heads_amount == 0, \
            "Dimension must be divisible by number of heads"
        self.sub_dimension = self.dimension // self.heads_amount

        self.weight = nn.ParameterDict({
            key: nn.Parameter(initialize_parameter(
                (self.dimension, self.dimension),
                weight[key] if weight is not None else None,
                lambda x:nn.init.uniform_(
                    x,
                    -1/math.sqrt(dimension),
                    1/math.sqrt(dimension)
                )
            )) for key in ["query_", "keys_", "values_", "output_"]
        })
        self.bias = nn.ParameterDict({
            key: nn.Parameter(initialize_parameter(
                (self.dimension,1),
                bias[key] if bias is not None else None,
                lambda x: nn.init.uniform_(
                    x,
                    -1 / math.sqrt(dimension),
                    1 / math.sqrt(dimension)
                )
            )) for key in ["query_", "keys_", "values_", "output_"]
        })

    @staticmethod
    def sdp_attention(x: Dict[str,Tensor], mask:Tensor) -> Tensor:
        # input:
        #   Q:      (bs, h, sl2, d/h)
        #   K:      (bs, h, sl1, d/h)
        #   V:      (bs, h, sl1, d/h)
        #   mask:   (bs, 1, sl1,sl1) or (bs,1,1,sl1)
        # mask shape, because for each batch its different, but same for each head
        sub_dimension = x["query_"].shape[-2]

        # score: (bs, h, sl2, sl1)
        score = ((x["query_"] @ x["keys_"].transpose(-2, -1))
                 / math.sqrt(sub_dimension))
        if mask is None:
            mask = torch.ones_like(score).type(torch.bool)
        else:
            mask = mask.unsqueeze(1)

        score = score.masked_fill(~mask, float("-inf"))

        # score: (bs, h, sl2, sl2) -> (bs, h, sl2, d/h) -> (bs, h, d/h, sl2)
        score = score.softmax(dim=-1)
        score = score @ x["values_"]
        return score.transpose(-1, -2)

    def forward(self, x: Dict[str,Tensor], mask: Tensor = None) -> Tensor:
        # input:
        #   Q:      (bs, sl2, d)
        #   K:      (bs, sl1, d)
        #   V:      (bs, sl1, d)
        #   mask:   (bs, sl1,sl1) or (bs,1,sl1)
        # mask is a bool tensor where mask[i,j,l] = True if Attention in batch i
        # from i -> j should be calculated
        batch_size = x["query_"].shape[0]
        seq_length = {key: x[key].shape[1] for key in ["query_", "keys_", "values_"]}

        for key in ["query_", "keys_", "values_"]:
            # can't do: (d,d) @ (bs,sl,d)
            # therefore transpose: (bs,sl,d) -> (bs,d,sl)
            x[key] = self.weight[key] @ x[key].transpose(-1,-2) + self.bias[key]
            # split (bs,d,sl) -> (bs,h,d/h,sl) -> (bs, h, sl, d/h) for head analysis
            x[key] = x[key].view(
                batch_size,
                self.heads_amount,
                self.sub_dimension,
                seq_length[key]
            ).transpose(-1,-2)

        # [(bs,h,d/h,sl2), (bs,h,d/h,sl1), (bs,h,d/h,sl1)] -> (bs, h, d/h, sl2)
        sdp = MaskedMultiHeadAttention.sdp_attention(x, mask)
        # (bs,h,d/h,sl2) -> (bs,d,sl2)
        sdp = torch.flatten(sdp,-3,-2) # merge h and d/h back to d, using flatten incase x is not contiguous

        # again transpose before linear, same as above
        # (bs, d, sl2) -> (bs, sl2, d) -> (bs, sl2, d) -> (bs, d, sl2)
        sdp = self.weight["output_"] @ sdp + self.bias["output_"]
        return sdp.transpose(-1,-2)


class PredictionLayer(nn.Module):
    def __init__(
            self,
            dimension:int,
            length:int,
            weight:Tensor = None,
            bias:Tensor = None
        ):
        super().__init__()
        self.weight = nn.Parameter(initialize_parameter(
            (length,dimension),
            weight,
            lambda x: nn.init.uniform_(
                x,
                -1/math.sqrt(dimension),
                1/math.sqrt(dimension)
            )
        ))
        self.bias = nn.Parameter(initialize_parameter(
            (length,1),
            bias,
            lambda x: nn.init.uniform_(
                x,
                -1 / math.sqrt(dimension),
                1 / math.sqrt(dimension)
            )
        ))
    def forward(self, x:Tensor) -> Tensor:
        x = x.transpose(-1, -2)
        x = self.weight @ x + self.bias
        return nn.functional.softmax(x.transpose(-1,-2), dim=-1)


