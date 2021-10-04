import torch
from torch.autograd import Variable
from probability import Sparsegen_lin


class Sparsegen_linear1(torch.nn.Module):
    def __init__(self, lam, normalized=True):
        super(Sparsegen_linear1, self).__init__()
        self.lam = lam
        self.normalized = normalized

    def forward(self, input):
        d1 = input.data.size()[0]
        d2 = input.data.size()[1]
        bs = input.data.size()[2]
        dim = input.data.size()[3]
        # z = input.sub(torch.mean(input,dim=1).repeat(1,dim))
        dtype = torch.FloatTensor
        z = input.type(dtype)

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        # Calculate data-driven lambda for self.data_driven = True
        # sort z
        z_sorted = torch.sort(z, descending=True)[0]

        # calculate k(z)
        z_cumsum = torch.cumsum(z_sorted, dim=3)
        k = Variable(torch.arange(1, dim + 1).unsqueeze(0).repeat(bs, 1))
        z_check = torch.gt(1 - self.lam + k * z_sorted, z_cumsum)

        # because the z_check vector is always [1,1,...1,0,0,...0] finding the
        # (index + 1) of the last `1` is the same as just summing the number of 1.
        k_z = torch.sum(z_check.float(), 3)

        # calculate tau(z)
        tausum = torch.sum(z_check.float() * z_sorted, 3)
        tau_z = (tausum - 1 + self.lam) / k_z
        prob = z.sub(tau_z.view(d1, d2, bs, 1).repeat(1, 1, 1, dim)).clamp(min=0).type(dtype)
        if self.normalized:
            prob /= (1 - self.lam)
        return prob


class Sparsegen_linear2(torch.nn.Module):
    def __init__(self, lam, normalized=True):
        super(Sparsegen_linear2, self).__init__()
        self.lam = lam
        self.normalized = normalized

    def forward(self, input):
        sparsegen = Sparsegen_lin(lam=self.lam)
        list1 = []
        for i in range(0, input.shape[0]):
            list2 = []
            for j in range(0, input.shape[1]):
                list2.append(sparsegen(input[i, j]))
            list1.append(torch.stack(list2, 0))
        attention_probs = torch.stack(list1, 0)
        return attention_probs
