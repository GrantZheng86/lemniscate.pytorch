import torch
from torch.autograd import Function
from torch import nn
import math

class LinearAverageOp(Function):
    """
    A customary function with autograd implemented for back prop.

    Question: Should memory bank be updated every epoch?
    """
    @staticmethod
    def forward(self, x, y, memory, params):
        """
        x: latent dimension after pass through the model. Dimension = Batch size x latent dimension(128)
        y: labels for each images. Labels are image indices since each image is its own class
        memory: memory bank? Dimension= class numbers(number of images) x latent dimension(128)
        params: hyper-parameter "temperature" and "momentum"
        """
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N
        
        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        """
        Break point in this function does not stop the run. Some suggestions said that things here are achieved through
        the C side.
        """
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        """
        inputSize: Latent feature dimension for image, 128 in the paper;
        outputSize: number of total images in set. Each image is regarded as a single class
        T: Temperature hyper-parameter
        momentum: momentum hyper-parameter for SGD
        """
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)

        # Memory back for feature storing
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))


    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

