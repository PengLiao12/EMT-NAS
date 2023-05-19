import numpy as np
from random import choices
from gnas.search_space.search_space import SearchSpace
from gnas.search_space.cross_over import individual_uniform_crossover, individual_block_crossover
from gnas.search_space.mutation import individual_flip_mutation
from gnas.genetic_algorithm.ga_results import GenetricResult
from gnas.genetic_algorithm.population_dict import PopulationDict
import random
import math


def genetic_algorithm_searcher(search_space: SearchSpace, generation_size=20, population_size=20, keep_size=0,
                               min_objective=False, mutation_p=None, p_cross_over=None, n_epochs=300, RMP=0.3):

    def population_initializer(p_size):
        return search_space.generate_population(p_size)

    def mutation_function(x):
        return individual_flip_mutation(x, mutation_p)

    def cross_over_function(x0, x1):
        return individual_block_crossover(x0, x1, p_cross_over)

    def selection_function(p):
        couples = choices(population=list(range(len(p))), weights=p,
                          k=generation_size*2*100)
        return np.reshape(np.asarray(couples), [-1, 2])

    return GeneticAlgorithms(population_initializer, mutation_function, cross_over_function, selection_function,
                             min_objective=min_objective, generation_size=generation_size,
                             population_size=population_size, keep_size=keep_size,n_epochs=n_epochs, RMP=RMP)


class GeneticAlgorithms(object):
    def __init__(self, population_initializer, mutation_function, cross_over_function, selection_function,
                 population_size=300, generation_size=20, keep_size=20, min_objective=False, n_epochs=300, RMP=0.3):
        ####################################################################
        # Functions
        ####################################################################
        self.population_initializer = population_initializer
        self.mutation_function = mutation_function
        self.cross_over_function = cross_over_function
        self.selection_function = selection_function
        self.n_epochs = n_epochs
        ####################################################################
        # parameters
        ####################################################################
        self.population_size = population_size
        self.generation_size = generation_size
        self.keep_size = keep_size
        self.min_objective = min_objective
        self.RMP = RMP
        ####################################################################
        # status
        ####################################################################
        self.max_dict_1 = PopulationDict()
        self.ga_result_1 = GenetricResult()
        self.current_dict_1 = dict()
        self.new_current_dict_1 = dict()
        self.generation_1 = self._create_random_generation()
        self.i = 0
        self.best_individual_1 = None
        self.avg_individual_1_fitness = None

        self.max_dict_2 = PopulationDict()
        self.ga_result_2 = GenetricResult()
        self.current_dict_2 = dict()
        self.new_current_dict_2 = dict()
        self.generation_2 = self._create_random_generation()
        self.best_individual_2 = None
        self.avg_individual_2_fitness = None

        self.old_current_dict = dict()
        self.mapping = dict()

    def _create_random_generation(self):
        return self.population_initializer(self.generation_size)


    def _create_new_generation(self):
        self.mapping= dict()
        population_fitness_1 = np.asarray(list(self.max_dict_1.values())).flatten()
        population_1 = np.asarray(list(self.max_dict_1.keys())).flatten()
        population_fitness_2 = np.asarray(list(self.max_dict_2.values())).flatten()
        population_2 = np.asarray(list(self.max_dict_2.keys())).flatten()

        p_1 = population_fitness_1 / np.nansum(population_fitness_1)
        p_2 = population_fitness_2 / np.nansum(population_fitness_2)
        p = np.hstack((p_1, p_2))

        population = np.hstack((population_1, population_2))

        if self.min_objective: p = 1 - p
        couples = self.selection_function(p)  # selection

        child_1 = []
        child_2 = []

        for c in couples:
            if (len(child_1) >= self.population_size) & (len(child_2) >= self.population_size):
                break
            elif (c[0] < self.population_size) & (c[1] < self.population_size) & (len(child_1)+2 <= self.population_size):
                child_p_1, child_p_2 = self.cross_over_function(population[c[0]], population[c[1]])  # cross-over
                new_child_p_1 = self.mutation_function(child_p_1)
                new_child_p_2 = self.mutation_function(child_p_2)

                child_1 = np.hstack((child_1, new_child_p_1))
                child_1 = np.hstack((child_1, new_child_p_2))
                self.mapping.update({new_child_p_1: population[c[0]]})
                self.mapping.update({new_child_p_2: population[c[1]]})
            elif (c[0] >= self.population_size) & (c[1] >= self.population_size) & (len(child_2)+2 <= self.population_size):
                child_p_1, child_p_2 = self.cross_over_function(population[c[0]], population[c[1]])  # cross-over
                new_child_p_1 = self.mutation_function(child_p_1)
                new_child_p_2 = self.mutation_function(child_p_2)

                child_2 = np.hstack((child_2, new_child_p_1))
                child_2 = np.hstack((child_2, new_child_p_2))
                self.mapping.update({new_child_p_1: population[c[0]]})
                self.mapping.update({new_child_p_2: population[c[1]]})
            elif ((c[0] < self.population_size) & (c[1] >= self.population_size)) | ((c[0] >= self.population_size) & (c[1] < self.population_size)):
                if random.random() < self.RMP:
                    child_p_1, child_p_2 = self.cross_over_function(population[c[0]], population[c[1]])  # cross-over
                    new_child_p_1 = self.mutation_function(child_p_1)
                    new_child_p_2 = self.mutation_function(child_p_2)
                else:
                    new_child_p_1 = self.mutation_function(population[c[0]])
                    new_child_p_2 = self.mutation_function(population[c[1]])

                if random.random() < 0.5:
                    if len(child_1) + 1 <= self.population_size:
                        child_1 = np.hstack((child_1, new_child_p_1))
                        if c[0] < self.population_size:
                            self.mapping.update({new_child_p_1: population[c[0]]})
                        if c[1] < self.population_size:
                            self.mapping.update({new_child_p_1: population[c[1]]})

                    if len(child_2) + 1 <= self.population_size:
                        child_2 = np.hstack((child_2, new_child_p_2))
                        if c[0] >= self.population_size:
                            self.mapping.update({new_child_p_2: population[c[0]]})
                        if c[1] >= self.population_size:
                            self.mapping.update({new_child_p_2: population[c[1]]})
                else:
                    if len(child_1) + 1 <= self.population_size:
                        child_1 = np.hstack((child_1, new_child_p_2))
                        if c[0] < self.population_size:
                            self.mapping.update({new_child_p_2: population[c[0]]})
                        if c[1] < self.population_size:
                            self.mapping.update({new_child_p_2: population[c[1]]})

                    if len(child_2) + 1 <= self.population_size:
                        child_2 = np.hstack((child_2, new_child_p_1))
                        if c[0] >= self.population_size:
                            self.mapping.update({new_child_p_1: population[c[0]]})
                        if c[1] >= self.population_size:
                            self.mapping.update({new_child_p_1: population[c[1]]})

        self.generation_1 = np.asarray(child_1)
        self.generation_2 = np.asarray(child_2)


    def first_population(self):
        self.i += 1
        # Task 1
        n_diff_1 = self.population_size

        self.current_dict_1 = dict()
        population_fitness_1 = np.asarray(list(self.max_dict_1.values())).flatten()
        population_1 = np.asarray(list(self.max_dict_1.keys())).flatten()

        self.best_individual_1 = population_1[np.argmax(population_fitness_1)]
        
        self.old_current_dict = dict()
        for key, value in self.max_dict_1.items():
            self.old_current_dict.update({key: value})

        fp_mean_1 = np.mean(population_fitness_1)
        fp_var_1 = np.var(population_fitness_1)
        fp_max_1 = np.max(population_fitness_1)
        fp_min_1 = np.min(population_fitness_1)
        self.ga_result_1.add_population_result(population_fitness_1, population_1)
        print(
           "population results 1 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
               fp_mean_1, fp_var_1, fp_max_1, fp_min_1))

        # Task 2
        n_diff_2 = self.population_size

        population_fitness_2 = np.asarray(list(self.max_dict_2.values())).flatten()
        population_2 = np.asarray(list(self.max_dict_2.keys())).flatten()
        self.best_individual_2 = population_2[np.argmax(population_fitness_2)]

        fp_mean_2 = np.mean(population_fitness_2)
        fp_var_2 = np.var(population_fitness_2)
        fp_max_2 = np.max(population_fitness_2)
        fp_min_2 = np.min(population_fitness_2)
        self.ga_result_2.add_population_result(population_fitness_2, population_2)
        
        for key, value in self.max_dict_2.items():
            self.old_current_dict.update({key: value})

        print(
            "population results 2 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_2, fp_var_2, fp_max_2, fp_min_2))

        return fp_mean_1, fp_var_1, fp_max_1, fp_min_1, n_diff_1, fp_mean_2, fp_var_2, fp_max_2, fp_min_2, n_diff_2

    def second_population(self):
        self.i += 1
        # Task 1
        generation_fitness_1 = np.asarray(list(self.current_dict_1.values()))
        generation_1 = list(self.current_dict_1.keys())
        self.ga_result_1.add_generation_result(generation_fitness_1, generation_1)

        f_mean_1 = np.mean(generation_fitness_1)
        f_var_1 = np.var(generation_fitness_1)
        f_max_1 = np.max(generation_fitness_1)
        f_min_1 = np.min(generation_fitness_1)
        total_dict_1 = self.max_dict_1.copy()
        total_dict_1.update(self.current_dict_1)

        best_max_dict_1 = total_dict_1.filter_top_n(self.population_size, min_max=not self.min_objective)
        n_diff_1 = self.max_dict_1.get_n_diff(best_max_dict_1)
        self.max_dict_1 = best_max_dict_1

        self.old_current_dict = dict()
        for key, value in self.max_dict_1.items():
            self.old_current_dict.update({key: value})

        self.current_dict_1 = dict()
        population_fitness_1 = np.asarray(list(self.max_dict_1.values())).flatten()
        population_1 = np.asarray(list(self.max_dict_1.keys())).flatten()
        self.best_individual_1 = population_1[np.argmax(population_fitness_1)]

        fp_mean_1 = np.mean(population_fitness_1)
        fp_var_1 = np.var(population_fitness_1)
        fp_max_1 = np.max(population_fitness_1)
        fp_min_1 = np.min(population_fitness_1)
        self.ga_result_1.add_population_result(population_fitness_1, population_1)

        print(
            "Update generation 1 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |population size {:d}|".format(
                f_mean_1, f_var_1, f_max_1, f_min_1, len(population_1)))
        print(
            "population results 1 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_1, fp_var_1, fp_max_1, fp_min_1))

        # Task 2
        generation_fitness_2 = np.asarray(list(self.current_dict_2.values()))
        generation_2 = list(self.current_dict_2.keys())
        self.ga_result_2.add_generation_result(generation_fitness_2, generation_2)

        f_mean_2 = np.mean(generation_fitness_2)
        f_var_2 = np.var(generation_fitness_2)
        f_max_2 = np.max(generation_fitness_2)
        f_min_2 = np.min(generation_fitness_2)
        total_dict_2 = self.max_dict_2.copy()
        total_dict_2.update(self.current_dict_2)

        best_max_dict_2 = total_dict_2.filter_top_n(self.population_size, min_max=not self.min_objective)
        n_diff_2 = self.max_dict_2.get_n_diff(best_max_dict_2)
        self.max_dict_2 = best_max_dict_2

        for key, value in self.max_dict_2.items():
            self.old_current_dict.update({key: value})

        self.current_dict_2 = dict()
        population_fitness_2 = np.asarray(list(self.max_dict_2.values())).flatten()
        population_2 = np.asarray(list(self.max_dict_2.keys())).flatten()
        self.best_individual_2 = population_2[np.argmax(population_fitness_2)]
        fp_mean_2 = np.mean(population_fitness_2)
        fp_var_2 = np.var(population_fitness_2)
        fp_max_2 = np.max(population_fitness_2)
        fp_min_2 = np.min(population_fitness_2)
        self.ga_result_2.add_population_result(population_fitness_2, population_2)

        print(
            "Update generation 2 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |population size {:d}|".format(
                f_mean_2, f_var_2, f_max_2, f_min_2, len(population_2)))
        print(
            "population results 2 | mean fitness: {:5.2f} | var fitness {:5.2f} | max fitness: {:5.2f} | min fitness {:5.2f} |".format(
                fp_mean_2, fp_var_2, fp_max_2, fp_min_2))

        return f_mean_1, f_var_1, f_max_1, f_min_1, n_diff_1, f_mean_2, f_var_2, f_max_2, f_min_2, n_diff_2

    def get_current_generation(self, num):
        if num == 1:
            return self.generation_1
        elif num == 2:
            return self.generation_2

    def get_max_generation(self, num):
        if num == 1:
            return np.asarray(list(self.max_dict_1.keys())).flatten()
        elif num == 2:
            return np.asarray(list(self.max_dict_2.keys())).flatten()

    def update_current_individual_fitness(self, individual, individual_fitness, num):
        if num == 1:
           if self.i < 1:
                new_individual_fitness = individual_fitness
           else:
                old_individual_fitness = self.old_current_dict[self.mapping[individual]]
                new_individual_fitness = old_individual_fitness + (individual_fitness - old_individual_fitness) * math.log((self.i+1), self.n_epochs)
           self.current_dict_1.update({individual: new_individual_fitness})
        elif num == 2:
           if self.i < 1:
               new_individual_fitness = individual_fitness
           else:
               old_individual_fitness = self.old_current_dict[self.mapping[individual]]
               new_individual_fitness = old_individual_fitness + (individual_fitness - old_individual_fitness) * math.log((self.i+1), self.n_epochs)
           self.current_dict_2.update({individual: new_individual_fitness})


    def update_max_individual_fitness(self, individual, individual_fitness, num):
        if num == 1:
           if self.i < 1:
               new_individual_fitness = individual_fitness
           else:
               old_individual_fitness = self.max_dict_1.values_dict[individual]
               new_individual_fitness = old_individual_fitness + (individual_fitness - old_individual_fitness) * math.log((self.i+1), self.n_epochs)
           self.max_dict_1.update_2({individual: new_individual_fitness})
        elif num == 2:
           if self.i < 1:
                new_individual_fitness = individual_fitness
           else:
               old_individual_fitness = self.max_dict_2.values_dict[individual]
               new_individual_fitness = old_individual_fitness + (individual_fitness - old_individual_fitness) * math.log((self.i+1), self.n_epochs)
           self.max_dict_2.update_2({individual: new_individual_fitness})

    def sample_child(self, num, flag):    #采集个体用于epoch下的batch训练
        if flag == 0:
            if num == 1:
                couples = choices(list(self.max_dict_1.keys()), k=2)  # random select two indivuals from parents Task1
            elif num == 2:
                couples = choices(list(self.max_dict_2.keys()), k=2)  # random select two indivuals from parents Task2
            return couples[0]  # select the first then mutation
        else:
            if random.random() < 0.5:
                if num == 1:
                    couples = choices(list(self.max_dict_1.keys()), k=2)  # random select two indivuals from parents Task1
                elif num == 2:
                    couples = choices(list(self.max_dict_2.keys()), k=2)  # random select two indivuals from parents Task2
            else:
            	if num == 1:
                    couples = choices(list(self.current_dict_1.keys()), k=2)  # random select two indivuals from offspring Task1
            	elif num == 2:
                    couples = choices(list(self.current_dict_2.keys()), k=2)  # random select two indivuals from offspring Task2
            return couples[0]



