from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Network(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.mlp = nn.Linear(num_in, num_out)

    def forward(self, x):
        return F.softplus(self.mlp(x))


if __name__=="__main__":
    torch.manual_seed(0)
    net = Network(2, 3)

    B = 4
    sample = Variable(torch.randn(B,2), requires_grad=True)

    y = net(sample)
    jac_list = []
    for i in range(B):
        jacob = torch.autograd.functional.jacobian(net, sample[i],
                                               create_graph=True,
                                               vectorize=True)
        print(jacob)
        jac_list.append(jacob)
    