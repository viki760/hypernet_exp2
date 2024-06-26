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
    
    # mnet_shape = [param.shape for param in mnet.parameters()]
    mnet_shape = [
        param.shape
        for name, param in mnet.named_parameters()
    ]
    # state_dict()
    # https://stackoverflow.com/questions/54746829/pytorch-whats-the-difference-between-state-dict-and-parameters
    # state_dict  contains not just the call to parameters but also buffers, etc.

    model = MLP(output_shape=mnet_shape, task_emb_dim=args.cfg.task_emb_dim, layers=[2, 3])
    # .to(device)

    return model


class MLP(torch.nn.Module):
    def __init__(self, output_shape, task_emb_dim, layers=[50, 100]):
        super(MLP,self).__init__()

        self.output_shape = output_shape

        _layers = [task_emb_dim] + layers
        self.fc_list = nn.ModuleList(
            [
                nn.Linear(_layers[i], _layers[i+1]) 
             for i in range(len(_layers)-1)
            ]
        )

        self.output_layers = [
            torch.nn.Linear(100, np.prod(dims))
            for dims in output_shape
        ]



    def forward(self, task_emb):
        x = F.relu(self.fc_list(task_emb))

        outputs = []
        for layer, dims in zip(self.output_layers, target_param_shape):
            res = layer(x).view(-1, *dims)
            outputs.append(res)

        return outputs