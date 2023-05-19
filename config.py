import json
import os


def save_config(path_dir, config):
    with open(os.path.join(path_dir, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)


def load_config(path_dir):
    with open(path_dir, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_config():
    return default_config_cnn()


def default_config_cnn():
    return {'batch_size': 128,
            'batch_size_val': 1000,
            'n_generations': 3,
            'n_blocks': 1,
            'n_block_type': 2,
            'n_nodes': 5,
            'n_channels_1': 20,
            'n_channels_2': 20,
            'generation_size': 10,
            'population_size': 10,
            'mutation_p': 0.02,
            'p_cross_over': 1.0,
            'RMP': 0.3,
            'learning_rate': 0.1,
            'lr_min': 0.0001,
            'weight_decay': 0.0001,
            'dropout': 0.10,
            'drop_path_keep_prob': 0.90,
            'drop_path_start_epoch': 50,
            'momentum': 0.9,
            }



