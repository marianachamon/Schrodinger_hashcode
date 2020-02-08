# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:44:46 2020

@author: maria
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 19:00:26 2020

@author: maria
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:19:37 2020

@author: mariana
"""

import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange
import random
from item import Item
from datetime import datetime
import copy
  
def fitness_calculation(individual):
    fitness_value = np.dot(individual,slicesVec)
    if fitness_value > maxSlices:
        fitness_value = maxSlices - fitness_value
#        fitness_value = fitness_value - (maxSlices - fitness_value)
    return fitness_value

def selection(generation, method='Fittest Half'):
#    generation['Normalized Fitness'] = \
#        sorted([generation['Fitness'][x]/sum(generation['Fitness']) 
#        for x in range(len(generation['Fitness']))], reverse = True)
#    generation['Cumulative Sum'] = np.array(
#        generation['Normalized Fitness']).cumsum()
    if method == 'Fittest Half':
        selected_individuals = [generation['Individuals'][-x-1]
            for x in range(int(len(generation['Individuals'])//2))]
        selected_fitnesses = [generation['Fitness'][-x-1]
            for x in range(int(len(generation['Individuals'])//2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}

    return selected

def pairing(elit1, selected1, method = 'Fittest'):
    elit = elit1
    selected = selected1
    individuals = [elit['Individuals']]+selected['Individuals']
    fitness = [elit['Fitness']]+selected['Fitness']
    if method == 'Fittest':
        parents = [[individuals[x],individuals[x+1]] 
                   for x in range(len(individuals)//2)]

    return parents


def mating(parents1, method='Single Point'):
    parents = parents1
    if method == 'Single Point':
        pivot_point = randint(1, len(parents[0]))
        offsprings = [parents[0] \
            [0:pivot_point]+parents[1][pivot_point:]]
        offsprings.append(parents[1]
            [0:pivot_point]+parents[0][pivot_point:])

    return offsprings

def existed(individual):
    if individual in existedInds:
        return True
    else:
        return False

def change_gen(individual,x):
    if individual[x] == 1:
        individual[x] = 0
    else:
        individual[x]=1 

        
def mutation(individual, upper_limit=0, lower_limit=1, muatation_rate=3, 
    method='Reset', standard_deviation = 0.001, choose_mut=1):

    if method == 'Reset':
#        i=0
#        mutated_individual = individual.copy()
#        r = random.sample( range(0,len(individual)), len(individual) )
#        change_gen(mutated_individual, r[0])
#        for i in range(1, len(individual)):
#            if existed(mutated_individual):                
#                change_gen(mutated_individual, r[i])
#            else:
#                break
        i=0
        mutated_individual = individual.copy()
        r = random.sample( range(0,len(individual)), len(individual) )
        for i in range(0, len(individual)):
            change_gen(mutated_individual, r[i])
            if not existed(mutated_individual):                
                break

    #swap
#        x=random.randint(0,1)
#        if x==1:
#            r1,r2 = random.sample(range(0,len(individual)-1), 2)
#            aux = mutated_individual[r1];
#            mutated_individual[r1] = mutated_individual[r2]
#            mutated_individual[r2] = aux
#            mutated_individual[x] = round(rnd()* \
#                (upper_limit-lower_limit)+lower_limit,1)

    return mutated_individual


        
def next_generation(gen, upper_limit, lower_limit):
    elit = {}
    next_gen = {}
    elit['Individuals'] = gen['Individuals'].pop(-1)
    elit['Fitness'] = gen['Fitness'].pop(-1)
    selected = selection(gen)
    parents = pairing(elit, selected)
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                    [y][z] for z in range(2)] 
                    for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]
    unmutated = selected['Individuals']+offsprings1+offsprings2
    mutated = [mutation(unmutated[x], upper_limit, lower_limit) 
        for x in range(len(gen['Individuals']))]
    unsorted_individuals = mutated + [elit['Individuals']]
    unsorted_next_gen = \
        [fitness_calculation(mutated[x]) 
         for x in range(len(mutated))]
    unsorted_fitness = [unsorted_next_gen[x]
        for x in range(len(gen['Fitness']))] + [elit['Fitness']]
    sorted_next_gen = \
        sorted([[unsorted_individuals[x], unsorted_fitness[x]]
            for x in range(len(unsorted_individuals))], 
                key=lambda x: x[1])
    next_gen['Individuals'] = [sorted_next_gen[x][0]
        for x in range(len(sorted_next_gen))]
    next_gen['Fitness'] = [sorted_next_gen[x][1]
        for x in range(len(sorted_next_gen))]
    gen['Individuals'].append(elit['Individuals'])
    gen['Fitness'].append(elit['Fitness'])
    return next_gen

def first_generation(pop):
    fitness = [fitness_calculation(pop[x]) 
        for x in range(len(pop))]
    sorted_fitness = sorted([[pop[x], fitness[x]]
        for x in range(len(pop))], key=lambda x: x[1])
    population = [sorted_fitness[x][0] 
        for x in range(len(sorted_fitness))]
    fitness = [sorted_fitness[x][1] 
        for x in range(len(sorted_fitness))]
    return {'Individuals': population, 'Fitness': sorted(fitness)}
  
    
def spawn_starting_population(amount,sizeInd, goodInd = "none"):
    if goodInd == "none":
        return [ [1 for i in range(0,sizeInd)] ] + [spawn_individual(sizeInd) for x in range (1,amount)]
    else:
        return [ [1 for i in range(0,sizeInd)] ] + [goodInd] + [spawn_individual(sizeInd) for x in range (2,amount)]

        
def spawn_individual(sizeInd):    
        return [random.randint(0,1) for x in range (0,sizeInd)]


path = "./a_example"
#path = "./b_small"
#path = "./c_medium"
#path = "./d_quite_big";
#path = "./e_also_big";

with open(path+".in") as f:
    maxSlices, types = [int(x) for x in next(f).split()];
    slicesVec = [int(x) for x in next(f).split()];
    
SLICES_SIZE = len(slicesVec) 

def getBestIndFromAnotherRunning(filename):    
    with open('./'+ filename) as f:
        ignore=[int(x) for x in next(f).split()]
        goodIndIndex = [int(x) for x in next(f).split()];
    goodInd = list(np.zeros(SLICES_SIZE , dtype=int))
    for x in goodIndIndex:
        goodInd[x] = 1 
    return goodInd


#ITEMS = [Item(slicesVec[x], slicesVec[x]) for x in range(0,SLICES_SIZE ) ]

POPSIZE = 100;
GEN_MAX = 2000;
g = 0;
maxGen = 0;
EXISTED_MAX = 5000*2000//SLICES_SIZE # existed_max = 10000*2000/len(individual)
#ok tune: EXISTED_MAX = 10000, para D (len(individual)=2000)
#Com arquivo D, chegou a mais de 100.000 em 200 gerações, POPSIZE=500, mas ficou muito lento

#goodStart = 'd_quite_big_ - 26-01-2020 04_44 - _out.in'
 
#goodInd = getBestIndFromAnotherRunning(goodStart)
    
pop = spawn_starting_population(POPSIZE-1,SLICES_SIZE )
#pop = population(20,8,1,0)

gen = first_generation(pop)

fitness_max = np.array([max(gen['Fitness'])])

finish = False
existedInds = []

startTime = datetime. now();

while finish == False:
#    if max(fitness_max) > 6: 
#        break
#    if max(fitness_avg) > 5:
#        break
    
    #gen.append(next_generation(gen[-1],1,0))
    gen = next_generation(gen,1,0);
    existedInds += gen['Individuals'].copy()
    existedInds = np.ndarray.tolist(np.unique(existedInds, axis=0))
    if len(existedInds) > EXISTED_MAX:
        existedInds = existedInds[0:len(existedInds):round(len(existedInds)/EXISTED_MAX)]
        
    
    aux = max(gen['Fitness'])
    fitness_max = np.append(fitness_max, aux )  
    if aux > maxGen:        
        maxGen = aux;
        bI = gen['Individuals'][np.argmax(gen['Fitness'])]
        bestInd =  list( np.nonzero( bI)[0] )
#    fitness_avg = np.append(fitness_avg, sum(
#        gen[-1]['Fitness'])/len(gen[-1]['Fitness']))

    g += 1;
    print(g, maxGen)
#    res = open(Result_file, 'a')
#    res.write('\n'+str(gen[-1])+'\n')
#    res.close()
#    if g == GEN_MAX or maxGen == maxSlices:
    if maxGen == maxSlices or g == GEN_MAX:
        break

endTime = datetime.now()
endTimeFmt = endTime.strftime(" - %d-%m-%Y %H_%M - ");  
  
with open(path + "_" + endTimeFmt + "_out.in","w") as f:
    f.write(str(len(bestInd))+"\n" + str(' '.join(map(str,bestInd))));  


with open(path + endTimeFmt + "statics.in","w") as f:
    f.write('Time lapsed: ' + str(endTime - startTime) + 'g: ' + str(g))
    f.write('\n# gen: ' + str(GEN_MAX) + '\nPOPSIZE: ' + str(POPSIZE))
    f.write('\nBest fitness: '+ str(max(fitness_max)))
    f.write('\nPoulation: '+ str(gen))