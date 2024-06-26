import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from argparse import Namespace

target_param_shape = [
    [2, 2],
    [3, 3]
]


task_emb_dim = 8
num_tasks = 2


def get_hnet_model(args:Namespace, mnet:nn.Module):
    # device = args.device
    # args.logger.info('Creating hypernetwork ...')


    # hnet_arch, temb_size, hnet_act, hnet_dropput_rate, hnet_noise_dim, temb_std = args.hnet
    # hnet = HyperNetwork(mnet.param_shapes, args.num_tasks, layers=hnet_arch,
    #         te_dim=temb_size, activation_fn=hnet_act,
    #         dropout_rate=hnet_dropput_rate,
    #         noise_dim=hnet_noise_dim,
    #         temb_std=temb_std).to(device)
    
    model = MLP()
    # .to(device)

    return model


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = torch.nn.Linear(task_emb_dim, 50)
        self.fc2 = torch.nn.Linear(50,100)

        self.output_layers = [
            torch.nn.Linear(100, np.prod(dims))
            for dims in target_param_shape
        ]

        # self._task_embs = nn.ParameterList()
        # for _ in range(num_tasks):
        #     self._task_embs.append(nn.Parameter(data=torch.Tensor(te_dim),
        #                                         requires_grad=True))
        #     torch.nn.init.normal_(self._task_embs[-1], mean=0., std=1.)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        outputs = []
        for layer, dims in zip(self.output_layers, target_param_shape):
            res = layer(x).view(-1, *dims)
            outputs.append(res)

        return outputs