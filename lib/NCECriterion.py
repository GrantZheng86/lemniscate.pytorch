import torch
from torch import nn

eps = 1e-7

class NCECriterion(nn.Module):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets):
        """
        x : p values from the non-parametric portion
        """
        batchSize = x.size(0)
        K = x.size(1)-1     # Number of negative samples
        Pnt = 1 / float(self.nLem)  # Probability of noise
        Pns = 1 / float(self.nLem)  # Probability of actual samples
        # NOTE: the noise and sample probability are the same because only 1 out of (number of training) is selected.
        # This applies to both noise and true sample

        # Equation references : RECURRENT NEURAL NETWORK LANGUAGE MODEL TRAINING WITH NOISE CONTRASTIVE ESTIMATION
        # FOR SPEECH RECOGNITION
        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt)
        Pmt = x.select(1,0)     # Equivalent to x[:, 0], the genuine samples
        Pmt_div = Pmt.add(K * Pnt + eps)
        lnPmt = torch.div(Pmt, Pmt_div)
        
        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        # Tensor.narrow => selects only a portion of the tensor. In this case, it selects the noise samples
        Pon_div = x.narrow(1,1,K).add(K * Pns + eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)
     
        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()
        
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)
        
        loss = - (lnPmtsum + lnPonsum) / batchSize
        
        return loss

