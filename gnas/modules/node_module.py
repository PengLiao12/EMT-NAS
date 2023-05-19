import torch.nn as nn
from gnas.modules.module_generator import generate_op
from modules.drop_module import DropModule


class ConvNodeModule(nn.Module):
    def __init__(self, node_config, config_dict):
        super(ConvNodeModule, self).__init__()
        self.nc = node_config

        self.n_channels = config_dict.get('n_channels')
        self.conv_module = []
        for j in range(node_config.get_n_inputs()):
            op_list = [DropModule(op, node_config.drop_path_control) for op in
                       generate_op(self.nc.op_list, self.n_channels, self.n_channels)]
            self.conv_module.append(op_list)
            [self.add_module('conv_op_' + str(i) + '_in_' + str(j), m) for i, m in enumerate(self.conv_module[-1])]

        self.non_linear_a = None
        self.non_linear_b = None
        self.input_a = None
        self.input_b = None
        self.cc = None
        self.op_a = None
        self.op_b = None

    def forward(self, inputs):
        net_a = inputs[self.input_a]
        net_b = inputs[self.input_b]
        return self.op_a(net_a) + self.op_b(net_b)

    def set_current_node_config(self, current_config):
        input_a, input_b, input_index_a, input_index_b, op_a, op_b = current_config
        self.select_index = [input_a, input_b]
        self.cc = current_config
        self.input_a = input_a
        self.input_b = input_b
        self.op_a = self.conv_module[input_index_a][op_a]
        self.op_b = self.conv_module[input_index_b][op_b]
        #### set grad false
        for p in self.parameters():
            p.requires_grad = False
        #### set grad true
        for p in self.op_b.parameters():
            p.requires_grad = True
        for p in self.op_a.parameters():
            p.requires_grad = True
