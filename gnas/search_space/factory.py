import numpy as np
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.operation_space import CnnNodeConfig
from enum import Enum

CNN_OP = ['Dw3x3', 'Dw5x5', 'Dw7x7', 'Dc3x3', 'Dc5x5', 'Dc7x7', 'Identity', 'Max3x3', 'Avg3x3']

class SearchSpaceType(Enum):
    CNNSingleCell = 0
    CNNDualCell = 1


def _two_input_cell(n_nodes, drop_path_control):
    node_config_list = [CnnNodeConfig(2, [0, 1], CNN_OP, drop_path_control=drop_path_control)]
    for i in range(n_nodes - 1):
        node_config_list.append(
            CnnNodeConfig(3 + i, list(np.linspace(0, 2 + i, 3 + i).astype('int')), CNN_OP,
                          drop_path_control=drop_path_control))
    return node_config_list


def _one_input_cell(n_nodes, drop_path_control):
    node_config_list = [CnnNodeConfig(1, [0], CNN_OP, drop_path_control=drop_path_control)]
    for i in range(n_nodes - 1):
        node_config_list.append(
            CnnNodeConfig(2 + i, list(np.linspace(0, 1 + i, 2 + i).astype('int')), CNN_OP,
                          drop_path_control=drop_path_control))
    return node_config_list


def get_gnas_cnn_search_space(n_nodes, drop_path_control, n_cell_type: SearchSpaceType) -> SearchSpace:
    node_config_list_a = _two_input_cell(n_nodes, drop_path_control)
    if n_cell_type == SearchSpaceType.CNNSingleCell:
        return SearchSpace(node_config_list_a)
    elif n_cell_type == SearchSpaceType.CNNDualCell:
        node_config_list_b = _two_input_cell(n_nodes, drop_path_control)
        return SearchSpace([node_config_list_a, node_config_list_b], single_block=False)


