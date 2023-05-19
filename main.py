import time
import torch.nn as nn

import torch
import torch.optim as optim
import os
import pickle
import argparse

import gnas
from models import model_cnn
from cnn_utils import evaluate_single, evaluate_individual_list,evaluate_parents_individual_list, uptate_parents_individual_list,  uptate_children_individual_list
from data import get_dataset
from common import make_log_dir
from config import get_config, load_config, save_config
from modules.drop_module import DropModuleControl
from modules.cosine_annealing import CosineAnnealingLR

def main():

    parser = argparse.ArgumentParser(description='PyTorch EMT-NAS')
    parser.add_argument('--Task_1', type=str, choices=['CIFAR10'], help='the working data',
                        default='CIFAR10')
    parser.add_argument('--Task_2', type=str, choices=['CIFAR100'], help='the working data',
                        default='CIFAR100')
    parser.add_argument('--config_file', type=str, help='location of the config file')
    parser.add_argument('--data_path', type=str, default=r".\dataset", help='location of the dataset')
    args = parser.parse_args()
    #######################################
    # Search Working Device
    #######################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(working_device)
    #######################################
    # Parameters
    #######################################
    config = get_config()
    if args.config_file is not None:
        print("Loading config file:" + args.config_file)
        config.update(load_config(args.config_file))
    config.update({'data_path': args.data_path, 'Task_1': args.Task_1, 'Task_2': args.Task_2, 'working_device': str(working_device)})
    print(config)
    ######################################
    # dataset
    ######################################
    trainloader_1, valloader_1, n_param_1 = get_dataset(config, args.Task_1)
    trainloader_2, valloader_2, n_param_2 = get_dataset(config, args.Task_2)
    ######################################
    # Config model and search space
    ######################################
    n_cell_type = gnas.SearchSpaceType(config.get('n_block_type') - 1)
    dp_control = DropModuleControl(config.get('drop_path_keep_prob'))
    ss = gnas.get_gnas_cnn_search_space(config.get('n_nodes'), dp_control, n_cell_type)

    net_1 = model_cnn.Net(config.get('n_blocks'), config.get('n_channels_1'), n_param_1,
                          config.get('dropout'), ss).to(working_device)
    net_2 = model_cnn.Net(config.get('n_blocks'), config.get('n_channels_2'), n_param_2,
                          config.get('dropout'), ss).to(working_device)
    ######################################
    # Build Optimizer
    #####################################
    optimizer_1 = optim.SGD(net_1.parameters(), lr=config.get('learning_rate'), momentum=config.get('momentum'),
                              nesterov=True,
                              weight_decay=config.get('weight_decay'))

    optimizer_2 = optim.SGD(net_2.parameters(), lr=config.get('learning_rate'), momentum=config.get('momentum'),
                               nesterov=True,
                               weight_decay=config.get('weight_decay'))
    ######################################
    # Build genetic_algorithm_searcher
    #####################################
    ga = gnas.genetic_algorithm_searcher(ss, generation_size=config.get('generation_size'),
                                         population_size=config.get('population_size'),
                                         keep_size=config.get('keep_size'),
                                         mutation_p=config.get('mutation_p'),
                                         p_cross_over=config.get('p_cross_over'),
                                         n_epochs=config.get('n_generations'),
                                         RMP=config.get('RMP'),)
    ######################################
    # Loss function
    ######################################
    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()
    ######################################
    # Select Learning schedule
    #####################################
    scheduler_1 = CosineAnnealingLR(optimizer_1, config.get('n_generations'), 1)
    scheduler_2 = CosineAnnealingLR(optimizer_2, config.get('n_generations'), 1)
    ##################################################
    # Generate log dir and Save Params
    ##################################################
    log_dir = make_log_dir(config)
    save_config(log_dir, config)
    ##################################################
    # Start
    ##################################################
    ra = gnas.ResultAppender()

    best_1 = 0
    best_2 = 0

    s = time.time()
    for generation in range(config.get('n_generations')):  # loop over the dataset multiple times
        print(generation)
        if generation == 0:
            uptate_parents_individual_list(ga.get_current_generation(1), ga, 1)
            uptate_parents_individual_list(ga.get_current_generation(2), ga, 2)

            # Taks 1
            running_loss_1 = 0.0
            correct_1 = 0
            total_1 = 0

            scheduler_1.step()
            s_1 = time.time()
            net_1 = net_1.train()

            if generation == config.get('drop_path_start_epoch'):
                dp_control.enable()

            for i, (inputs, labels) in enumerate(trainloader_1, 0):  # Loop over batchs

                # sample child from population
                net_1.set_individual(ga.sample_child(1,0))

                inputs = inputs.to(working_device)
                labels = labels.to(working_device)

                optimizer_1.zero_grad()
                outputs = net_1(inputs)

                _, predicted = torch.max(outputs[0], 1)
                total_1 += labels.size(0)
                correct_1 += (predicted == labels).sum().item()

                loss1 = criterion_1(outputs[0], labels)
                loss1.backward()
                optimizer_1.step()
                running_loss_1 += loss1.item()

            #  Task 2
            running_loss_2 = 0.0
            correct_2 = 0
            total_2 = 0

            scheduler_2.step()
            net_2 = net_2.train()

            for i, (inputs, labels) in enumerate(trainloader_2, 0):  # Loop over batchs
                 # sample child from population
                 net_2.set_individual(ga.sample_child(2,0))

                 inputs = inputs.to(working_device)
                 labels = labels.to(working_device)

                 optimizer_2.zero_grad()
                 outputs = net_2(inputs)

                 _, predicted = torch.max(outputs[0], 1)
                 total_2 += labels.size(0)
                 correct_2 += (predicted == labels).sum().item()

                 loss2 = criterion_2(outputs[0], labels)
                 loss2.backward()
                 optimizer_2.step()
                 running_loss_2 += loss2.item()

            # Evaluate parents Task 1 on the validation set
            evaluate_parents_individual_list(ga.get_current_generation(1), ga, net_1, valloader_1,
                                         working_device, 1)
            # Evaluate parents Task 2 on the validation set
            evaluate_parents_individual_list(ga.get_current_generation(2), ga, net_2, valloader_2,
                                         working_device, 2)

            _, _, _, _,  n_diff_1, _, _, _, _,  n_diff_2 = ga.first_population()  # replacement
        else:
            # Update population
            ga._create_new_generation()
            uptate_children_individual_list(ga.get_current_generation(1), ga, 1)
            uptate_children_individual_list(ga.get_current_generation(2), ga, 2)

            #  Task 1
            running_loss_1 = 0.0
            correct_1 = 0
            total_1 = 0


            scheduler_1.step()
            s_1 = time.time()
            net_1 = net_1.train()

            if generation == config.get('drop_path_start_epoch'):
                dp_control.enable()

            for i, (inputs, labels) in enumerate(trainloader_1, 0):  # Loop over batchs
                # sample child from population
                net_1.set_individual(ga.sample_child(1, 1))

                inputs = inputs.to(working_device)
                labels = labels.to(working_device)

                optimizer_1.zero_grad()  # zero the parameter gradients
                outputs = net_1(inputs)  # forward

                _, predicted = torch.max(outputs[0], 1)
                total_1 += labels.size(0)
                correct_1 += (predicted == labels).sum().item()

                loss1 = criterion_1(outputs[0], labels)
                loss1.backward()
                optimizer_1.step()
                running_loss_1 += loss1.item()

            #  Task 2
            running_loss_2 = 0.0
            correct_2 = 0
            total_2 = 0

            scheduler_2.step()
            net_2 = net_2.train()
            for i, (inputs, labels) in enumerate(trainloader_2, 0):  # Loop over batchs
                # sample child from population
                net_2.set_individual(ga.sample_child(2,1))

                inputs = inputs.to(working_device)
                labels = labels.to(working_device)

                optimizer_2.zero_grad()
                outputs = net_2(inputs)

                _, predicted = torch.max(outputs[0], 1)
                total_2 += labels.size(0)
                correct_2 += (predicted == labels).sum().item()

                loss2 = criterion_2(outputs[0], labels)
                loss2.backward()
                optimizer_2.step()
                running_loss_2 += loss2.item()


            # Evaluate child Task 1 on the validation set
            evaluate_individual_list(ga.get_current_generation(1), ga, net_1, valloader_1,
                                             working_device, 1)
            # Evaluate child Task 2 on the validation set
            evaluate_individual_list(ga.get_current_generation(2), ga, net_2, valloader_2,
                                             working_device, 2)
            # Generate offspring
            _, _, _, _, n_diff_1, _, _, _, _, n_diff_2 = ga.second_population()


        # evalute best
        f_max_1 = evaluate_single(ga.best_individual_1, net_1, valloader_1, working_device)
        f_max_2 = evaluate_single(ga.best_individual_2, net_2, valloader_2, working_device)
        total_params_1 = sum(p.numel() for p in net_1.parameters() if p.requires_grad)
        total_params_2 = sum(p.numel() for p in net_2.parameters() if p.requires_grad)

        # results
        if f_max_1 > best_1:
            print("Update Best 1")
            best_1 = f_max_1
            torch.save(net_1.state_dict(), os.path.join(log_dir, 'best_model_1.pt'))
            torch.save(net_1, os.path.join(log_dir, 'ALL_best_model_1.pt'))
            gnas.draw_network(ss, ga.best_individual_1, os.path.join(log_dir, 'best_graph_1_' + str(generation) + '_'))
            pickle.dump(ga.best_individual_1, open(os.path.join(log_dir, 'best_individual_1.pickle'), "wb"))

        print(
            '1 |Generation: {:2d}|Time: {:2.3f}|Loss:{:2.3f}|Accuracy: {:2.3f}%|Validation Accuracy: {:2.3f}%|LR: {:2.5f}|N Change : {:2d}|total_params: {:2.3f}|'.format(
                generation, (
                               time.time() - s_1) / 60,
                       running_loss_1 / i,
                       100 * correct_1 / total_1, best_1,
                scheduler_1.get_lr()[
                    -1],
                n_diff_1, total_params_1))
        ra.add_epoch_result('Generation', generation)
        ra.add_epoch_result('N 1', n_diff_1)
        ra.add_epoch_result('Best 1', best_1)
        ra.add_epoch_result('Validation Accuracy 1', f_max_1)
        ra.add_epoch_result('LR 1', scheduler_1.get_lr()[-1])
        ra.add_epoch_result('Training Loss 1', running_loss_1 / i)
        ra.add_epoch_result('Training Accuracy 1', 100 * correct_1 / total_1)
        ra.add_epoch_result('Time', (time.time() - s_1) / 60)
        ra.add_result('Fitness 1', ga.ga_result_1.fitness_list)
        ra.add_result('Fitness-Population 1', ga.ga_result_1.fitness_full_list)

        if f_max_2 > best_2:
            print("Update Best 2")
            best_2 = f_max_2
            torch.save(net_2.state_dict(), os.path.join(log_dir, 'best_model_2.pt'))
            torch.save(net_2, os.path.join(log_dir, 'ALL_best_model_2.pt'))
            gnas.draw_network(ss, ga.best_individual_2, os.path.join(log_dir, 'best_graph_2_' + str(generation) + '_'))
            pickle.dump(ga.best_individual_2, open(os.path.join(log_dir, 'best_individual_2.pickle'), "wb"))

        print(
            '2 |Generation: {:2d}|Time: {:2.3f}|Loss:{:2.3f}|Accuracy: {:2.3f}|Validation Accuracy: {:2.3f}|LR: {:2.5f}|N Change : {:2d}|total__params : {:2.3f}|'.format(
                generation, (
                               time.time() - s) / 60,
                       running_loss_2 / i,
                       100 * correct_2 / total_2, best_2,
                scheduler_2.get_lr()[
                    -1],
                n_diff_2, total_params_2))
        ra.add_epoch_result('N 2', n_diff_2)
        ra.add_epoch_result('Best 2', best_2)
        ra.add_epoch_result('Validation Accuracy 2', f_max_2)
        ra.add_epoch_result('LR 2', scheduler_2.get_lr()[-1])
        ra.add_epoch_result('Training Loss 2', running_loss_2 / i)
        ra.add_epoch_result('Training Accuracy 2', 100 * correct_2 / total_2)
        ra.add_epoch_result('Time Sum', (time.time() - s) / 60)
        ra.add_result('Fitness 2', ga.ga_result_2.fitness_list)
        ra.add_result('Fitness-Population 2', ga.ga_result_2.fitness_full_list)
        ra.save_result(log_dir)

        
if __name__ == '__main__':
    main()
