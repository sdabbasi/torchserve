import os
import time

import torch
import numpy as np

from ts.torch_handler.base_handler import BaseHandler

import logging 

logger = logging.getLogger(__name__)


def extra_util_function():
    return None

class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.device = None


    def initialize(self, context):
        """
        Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model or
            tokenizer is missing
        """

        # might use input context to retrieve data from input command: 
        # properties = context.system_properties
        # self.manifest = context.manifest

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = None
        self.model = model 
        self.initialized = True



    def preprocess(self, requests):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """

        input = None
        
        # have to use input request to feed "data" variable

        return torch.as_tensor(input, device=self.device)


    def inference(self, data):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.

        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.

        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """

        output = self.model(data)

        # have to use input data to feed the "input" variable

        return torch.as_tensor(output, device=self.device)
    
    
    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.

        Returns:
            List: The post process function returns a list of the predicted output.
        """
        processed_output = None

        # should use data to feed the processed_output 

        return [processed_output.tolist()]