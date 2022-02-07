import torch
from torch.autograd import Function
from torch import nn
from .alias_multinomial import AliasMethod
import math

class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        """
        x: latent representation after pass through the initial network, 128 in paper
        y: label for each image, y in this case is the index of each image since each one is its own category
        memory: memory bank
        idx:
        params: Hyper parameter [negative samples, temperature, normalizing constant, momentum]

        This function finds P(i|V) as shown in equation 2 in the paper
        """
        K = int(params[0].item())   # Number of negative samples
        T = params[1].item()        # Hyper parameter -- Temperature
        Z = params[2].item()        # Normalizing Constant

        momentum = params[3].item()
        batchSize = x.size(0)
        outputSize = memory.size(0)     # Number of training images, since each image is categorized as its own cate
        inputSize = memory.size(1)      # Latent dimensions

        # sample positives & negatives
        idx.select(1,0).copy_(y.data)       # Changes idx[:, 0] into labels, idx[:, 1:end] remains randomly generated

        # sample correspoinding weights from memory
        # torch.index_select： selects samples along the 2nd argument with indices in 3rd argument
        weight = torch.index_select(memory, 0, idx.view(-1))
        weight.resize_(batchSize, K+1, inputSize)

        # inner product
        # torch.bmm : batch matrix multiplication. The non-parameterized equation in paper
        out = torch.bmm(weight, torch.reshape(x, (batchSize, inputSize, 1)))
        out.div_(T).exp_() # batchSize * self.K+1
        x.data.resize_(batchSize, inputSize)

        if Z < 0:   # if Z is not initialized, using the Monte-Carlo method to approximate one
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))

        out.div_(Z).resize_(batchSize, K+1)

        self.save_for_backward(x, memory, y, weight, out, params)
        # output in dimension of (k + 1) * batch_size

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        
        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.data.mul_(out.data)
        # add temperature
        gradOutput.data.div_(T)

        # gradOutput.data.resize_(batchSize, 1, K+1)
        torch.reshape(gradOutput, (batchSize, 1, K+1))
        
        # gradient of linear
        gradInput = torch.bmm(gradOutput.data, weight)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None, None

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params',torch.tensor([K, T, -1, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
 
    def forward(self, x, y):
        batchSize = x.size(0)
        idx = self.multinomial.draw(batchSize * (self.K+1)).view(batchSize, -1)
        out = NCEFunction.apply(x, y, self.memory, idx, self.params)
        return out

