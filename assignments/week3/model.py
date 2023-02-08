import xxlimited
from scipy import datasets
import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    A deep learning model to classify the images
    ...

    Attributes
    ----------
    input size: int
        the input shape (features) of the images
    hidden_size: int
        the The number of neurons H in the hidden layer
    num_classes: int
        The number of classes C
    activation (torch.nn.func):
        The activation function to use in the hidden layer.
    initializer (torch.nn.init):
        The initializer to use for the weights.

    Methods
    -------
    _init_: initialization process
    foward: predict the output of model
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size (int): The dimension D of the input data.
            hidden_size (int): The number of neurons H in the hidden layer.
            num_classes (int): The number of classes C.
            activation (torch.nn.func): The activation function to use in the hidden layer.
            initializer (torch.nn.init): The initializer to use for the weights.

        Returns:
            None
        """
        super(MLP, self).__init__()
        # save the activation and initializer
        self.actv = activation()
        self.init = initializer
        # create a series of layers with ModuelList
        self.layers = torch.nn.ModuleList()
        for i in range(hidden_count):
            # save the number of hidden neurons
            next_num_inputs = hidden_size
            # create each layer
            self.layers += [torch.nn.Linear(input_size, next_num_inputs)]
            # save the current output size as the next input size
            input_size = next_num_inputs
        # create the output linear layer
        self.out = torch.nn.Linear(input_size, num_classes)
        # initialize the weight using the first layer
        self.init(self.layers[0].weight)

    def forward(self, x: datasets) -> float:
        """
        Forward pass of the network.

        Arguments:
            x (datasets): The input data.

        Returns:
            x (float): The output of the network.
        """
        # flatten the tensor
        x = x.view(x.shape[0], -1)
        # for each layer, apply the activation function to it and return the prediction
        for layer in self.layers:
            x = self.actv(layer(x))
        x = self.out(x)
        return x
