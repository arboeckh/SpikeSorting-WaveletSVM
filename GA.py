"""Genetic algorithm for spike sorting"""
from random import randint , random, choices, choice
from operator import add
from functools import reduce
import numpy as np
import matplotlib . pyplot as plt
import copy

import ss

class GA_SpikeSorting:
    """Custom genetic algorithm algorithm class for optimising a spike sorting algorithm"""

    def __init__(self, pop_count, params, data, survival_ratio, random_select, elitism_count, mutation_rate):
        #init of params and make value checks
        assert pop_count > 0
        self.pop_count = pop_count
        self.params = params
        self.data = data
        assert survival_ratio > 0 and survival_ratio <= 1
        self.survivor_count = int(self.pop_count * survival_ratio)
        assert random_select >= 0 and random_select <1
        self.random_select = random_select
        assert elitism_count >= 0 and elitism_count < self.pop_count
        self.elitism_count = elitism_count
        assert mutation_rate >= 0 and mutation_rate <=1 
        self.mutation_rate = mutation_rate

        #init the population and rank
        self.population = self.generate_population()
        self.rank_fitness()

        #variables to track progress
        self.generation_fitness_mean = [self.grade_pop()] #fitness initial population
        self.gen_best = [self.population[0]]  #individual with best fitness
        self.best = self.population[0]

    class individual:
        """Generic individual template that is filled out in child classes of 
        the generic GA class."""

        def __init__(self,  params, data, genome=None):
            self.params = params
            self.data = data
            self.crossover_locs = np.arange(0,len(params))
            #if initialized at the start, assign a genome and evaluate fitness
            if genome is None:
                self.init_genome()
                self.fitness_eval()   
            #if given a genome (i.e. from breeding) then wait for mutations
            #evaluate the fitness
            else:
                self.genome = genome
                self.fitness = 0        
            
        
        def init_genome(self):
            """initializes the individual's genome"""
            self.genome = {}
            for key in self.params:
                self.genome[key] = choice(self.params[key])
            pass

        def fitness_eval(self):
            """Evaluates and updates the individual's fintess"""
            #create spike sorter
            spike_sorter_params = copy.copy(self.genome)
            spike_sorter_params.update(self.data)
            spike_sorter = ss.SpikeSorter(**spike_sorter_params)
            spike_sorter.train()
            stabilities = spike_sorter.self_blur_stability(1, gamma=0.15, plotMeanWaveforms=False)
            stability = np.mean(stabilities)        #use mean stability
            cross_validation = spike_sorter.cross_validate(plot=False)
            cross_val_accuracy = cross_validation['total_accuracy'] #use total accuracy

            self.fitness = (stability**2 + cross_val_accuracy**2)/2
            print(f"fitness: {self.fitness}, acc: {cross_val_accuracy}, stab: {stability}")

        def mutate(self):
            """Cause mutation in genome based on params"""
            #choose a random parameter to mutate
            mut_key_index = randint(0,len(self.params)-1)
            for i,key in enumerate(self.params):
                if i == mut_key_index:
                    mut_key = key
                    break
            #choose new value form list available for that parameter
            self.genome[mut_key] = choice(self.params[mut_key]) 
            pass


    def generate_population(self):
        """Creates a population of individuals
        count: number of individuals in the population
        length: number of locii per individual
        min, max: minimum and maximum initialization values for each allele"""
        return [self.individual(self.params, self.data) for _ in range(self.pop_count)]

    def grade_pop(self):
        """Finds the average fitness of a population
        pop: population to be evaluated
        target: desired result"""
        summed = reduce(add, (x.fitness for x in self.population))
        return summed / (self.pop_count * 1.0)

    def rank_fitness(self):
        """rank population based on fitness"""
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

    def roulette_select(self, parents, roulette_weights):
        """Uses 'roulette' type selection to select parents. Parents with higher
        fitness will be more likely to be selected for reproduction"""
        while(1):
            sel_parents = choices(parents, weights=roulette_weights, k=2)
            if sel_parents[0] != sel_parents[1]:
                return sel_parents

    def cross_over_breed(self, parents):
        """Breed a child from 2 parents using cross-over 
        followed by a chance mutation"""
        #get cross-over location from parent1
        cross_loc = choice(parents[0].crossover_locs)
        genome = {}
        for i,key in enumerate(self.params):
            if i<cross_loc:
                genome[key] = parents[0].genome[key]
            else:
                genome[key] = parents[1].genome[key]
        new_child = self.individual(self.params, self.data, genome=genome)
        #check for mutation
        if self.mutation_rate > random():
            new_child.mutate()   #mutate within initialisation range
        new_child.fitness_eval()
        return new_child
        pass

    def breed(self, parents, offspring_no):
        """Uses the list of parents to generate a specific number of offspring"""
        # roulette_distribution = self.update_roulette_distribution(parents)
        roulette_weights = np.array([x.fitness for x in parents])
        #make all weights positive and add small number to ensure 
        # non-zero if all identical
        roulette_weights += np.abs(roulette_weights.min()) + 0.1  
        if sum(roulette_weights) == 0:
            a=1
        children = []
        while len(children) < offspring_no:
            selected_parents = self.roulette_select(parents, roulette_weights)
            children.append(self.cross_over_breed(selected_parents))
        return children

    def evolve(self):
        """One step of evolution applied to a population"""
        #1. only most successful live to reproduce based on survival ratio
        parents = self.population[:self.survivor_count]
        #2. append some individuals randomly from those non-selected
        for ind in self.population[self.survivor_count:]:
            if self.random_select > random():
                parents.append(ind)
        #3. select elite individuals and append to nextGen list 
        nextGen = parents[:self.elitism_count]    #return without their success
        #4. perform breeding according to how many children are needed
        children_no = self.pop_count - len(nextGen)
        children = self.breed(parents, children_no)
        #5. assign next generation
        nextGen.extend(children)
        self.population = nextGen
        self.rank_fitness() #rank new population based on their fitnesses

        #fill in performance tracking variables
        self.generation_fitness_mean.append(self.grade_pop()) 
        #append to the best fitness of generation
        self.gen_best.append(self.population[0])
        #check if new is all time best
        if self.population[0].fitness > self.best.fitness:
            self.best = self.population[0]
            print(f"best changed to {self.best.fitness}")


