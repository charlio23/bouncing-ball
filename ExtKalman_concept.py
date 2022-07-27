from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.modules import MLPJacobian
import time


if __name__=="__main__":
    torch.manual_seed(0)
    net = MLPJacobian(32, 256, 32).cuda().float()

    B = 32
    sample = Variable(torch.randn(B,32), requires_grad=True).cuda().float()

    y = net(sample)
    jac_list = []
    start = time.time()
    for i in range(B):
        jacob = torch.autograd.functional.jacobian(net, sample[i],
                                               create_graph=True,
                                               vectorize=True)
        jac_list.append(jacob)
    print("Pytorch jacobian time: ", time.time() - start)
    start = time.time()
    y, jacob = net.jacobian(sample)
    print("Explicit jacobian time: ", time.time() - start)
    print(y.size())
    print(jacob.size())