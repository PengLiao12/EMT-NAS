from gnas.search_space.operation_space import CnnNodeConfig
from gnas.modules.node_module import ConvNodeModule

__module_dict__ = {
                   CnnNodeConfig: ConvNodeModule}


def get_module(node_config, config_dict):
    m = __module_dict__.get(type(node_config))
    if m is None:
        raise Exception('Can\'t find module named:' + node_config)
    return m(node_config, config_dict)
