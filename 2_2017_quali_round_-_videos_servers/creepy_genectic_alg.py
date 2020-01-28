# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:58:03 2020

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
from datetime import datetime


def overflowFix(individual):
    fixedInd = individual.copy()
    for i in range(0,len(fixedInd)): 
        x = fixedInd[i]
        si = np.sum(x)
        while si > X:
           r = random.randint(0,len(x)-1)               
           fixedInd[i].pop(r)
           si = np.sum(x)
    return fixedInd  
        
def fitness_calculation(individual):
    fitness_value = 0
    individual = overflowFix(individual)
    for i in range(0,len(req)):  
        #req[i,1]: index of the endPoint in req i
        cachesEndp = endPoints['svconn'][ req[i,1] ] # Get caches connect to endPoint of req i
        lci = 0
        Ldi=0
        for ci,lc in cachesEndp:
            if  req[i,0] in individual[ci]: # if the video from req i is in one or more of the caches
                aux = lc # get lattency of cache_x-endPoint_i
                if aux < lci  or lci==0:
                    lci = aux
                    Ldi = endPoints['dtc'][ req[i,1] ] 
        fitness_value += (Ldi - lci)*req[i,2]
    fitness_value *= 1000/np.sum(req[:,2])
    
#        if notFeasible(indivi)  ...          

    return fitness_value

def roulette(cum_sum, chance):
    veriable = list(cum_sum.copy())
    veriable.append(chance)
    veriable = sorted(veriable)
    return veriable.index(chance)

def selection(generation, method='Fittest Half'):
#    generation['Normalized Fitness'] = \
#        sorted([generation['Fitness'][x]/sum(generation['Fitness']) 
#        for x in range(len(generation['Fitness']))], reverse = True)
#    generation['Cumulative Sum'] = np.array(
#        generation['Normalized Fitness']).cumsum()
    if method == 'Roulette Wheel':
        selected = []
        for x in range(len(generation['Individuals'])//2):
            selected.append(roulette(generation
                ['Cumulative Sum'], rnd()))
            while len(set(selected)) != len(selected):
                selected[x] = \
                    (roulette(generation['Cumulative Sum'], rnd()))
        selected = {'Individuals': 
            [generation['Individuals'][int(selected[x])]
                for x in range(len(generation['Individuals'])//2)]
                ,'Fitness': [generation['Fitness'][int(selected[x])]
                for x in range(
                    len(generation['Individuals'])//2)]}
    elif method == 'Fittest Half':
        selected_individuals = [generation['Individuals'][-x-1]
            for x in range(int(len(generation['Individuals'])//2))]
        selected_fitnesses = [generation['Fitness'][-x-1]
            for x in range(int(len(generation['Individuals'])//2))]
        selected = {'Individuals': selected_individuals,
                    'Fitness': selected_fitnesses}
#    elif method == 'Random':
#        selected_individuals = \
#            [generation['Individuals']
#                [randint(1,len(generation['Fitness']))]
#            for x in range(int(len(generation['Individuals'])//2))]
#        selected_fitnesses = [generation['Fitness'][-x-1]
#            for x in range(int(len(generation['Individuals'])//2))]
#        selected = {'Individuals': selected_individuals,
#                    'Fitness': selected_fitnesses}
    return selected

def pairing(elit, selected, method = 'Fittest'):
    individuals = [elit['Individuals']]+selected['Individuals']
    fitness = [elit['Fitness']]+selected['Fitness']
    if method == 'Fittest':
        parents = [[individuals[x],individuals[x+1]] 
                   for x in range(len(individuals)//2)]
#    if method == 'Random':
#        parents = []
#        for x in range(len(individuals)//2):
#            parents.append(
#                [individuals[randint(0,(len(individuals)-1))],
#                 individuals[randint(0,(len(individuals)-1))]])
#            while parents[x][0] == parents[x][1]:
#                parents[x][1] = individuals[
#                    randint(0,(len(individuals)-1))]
#    if method == 'Weighted Random':
#        normalized_fitness = sorted(
#            [fitness[x] /sum(fitness) 
#             for x in range(len(individuals)//2)], reverse = True)
#        cummulitive_sum = np.array(normalized_fitness).cumsum()
#        parents = []
#        for x in range(len(individuals)//2):
#            parents.append(
#                [individuals[roulette(cummulitive_sum,rnd())],
#                 individuals[roulette(cummulitive_sum,rnd())]])
#            while parents[x][0] == parents[x][1]:
#                parents[x][1] = individuals[
#                    roulette(cummulitive_sum,rnd())]
    return parents


def mating(parents, method='Single Point'):
    if method == 'Single Point':
        pivot_point = randint(1, len(parents[0]))
        offsprings = [parents[0] \
            [0:pivot_point]+parents[1][pivot_point:]]
        offsprings.append(parents[1]
            [0:pivot_point]+parents[0][pivot_point:])
    if method == 'Two Pionts':
        pivot_point_1 = randint(1, len(parents[0]-1))
        pivot_point_2 = randint(1, len(parents[0]))
        while pivot_point_2<pivot_point_1:
            pivot_point_2 = randint(1, len(parents[0]))
        offsprings = [parents[0][0:pivot_point_1]+
            parents[1][pivot_point_1:pivot_point_2]+
            [parents[0][pivot_point_2:]]]
        offsprings.append([parents[1][0:pivot_point_1]+
            parents[0][pivot_point_1:pivot_point_2]+
            [parents[1][pivot_point_2:]]])
    return offsprings

def mutation(individual, upper_limit, lower_limit, muatation_rate=3, 
    method='Reset', standard_deviation = 0.001):
#    gene = [randint(0, 7)]
#    for x in range(muatation_rate-1):
#        gene.append(randint(0, 7))
#        while len(set(gene)) < len(gene):
#            gene[x] = randint(0, 7)
    mutated_individual = individual.copy()
    if method == 'Gauss':
        for x in range(muatation_rate):
            mutated_individual[x] = \
            round(individual[x]+gauss(0, standard_deviation), 3)
    if method == 'Reset':
        for x in range(random.randint(0,muatation_rate)):
            r1 = random.randint(0,len(individual)-1)
            r2 = random.randint(0, len(individual[r1])-1)
            possibleGenes = list( set(range(0,V)) - set(individual[r1]) )
            r3 = random.randint(0,len(possibleGenes)-1)
            mutated_individual[r1][r2] = possibleGenes[r3]

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



# Generations and fitness values will be written to this file
Result_file = 'GA_Results.txt'
# Creating the First Generation
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

  
    
def spawn_starting_population(amount,sizeInd):
    return [spawn_individual(sizeInd) for x in range (0,amount)]

def spawn_individual(sizeInd): 
#    ind = []
#    [V,C] = sizeInd
#    space = 0;
#    for i in range (0, V*C):
#        g = random.randint(0,1)
#        if g==1:
#            if space + sVideos[np.mod(i,V)] > X:
#                g = 0;
#            else:
#                space += sVideos[np.mod(i,V)];
#        if np.mod(i,V) == 0:
#            space = 0;
#        ind.append(g)
    ind=[]
    for i in range(0,C):
        j=0;
        r = random.sample(range(0,V),V)
        space = r[0]
        aux = []

        while space < X :   
            aux.append( r[j] )
            space += sVideos[ r[j] ]
            j += 1;
        ind.append( aux )

#    fitness_value = 0
#    individual = overflowFix(individual)
#    for i in range(0,1):  
#        cachesEndp = endPoints['svconn'][ req[i,1] ] # Get caches connect to endPoint i
#        lci = 0
#        Ldi=0
#        for ci,lc in cachesEndp:
#            if  req[i,0] in individual[ci]:
#                aux = lc # get lattency of cache_x-endPoint_i
#                Ldi = endPoints['dtc'][ req[i,1] ] 
#                if lci < aux:
#                    lci = aux
#                    Ldi = endPoints['dtc'][ req[i,1] ] 
#        fitness_value += (Ldi - lci)*req[i,2]     
    return ind


filename = 'me_at_the_zoo';
filename = "videos_worth_spreading"
filename = 'example'
#filename = "trending_today";
#filename = "kittens.in";

endPoints = {}
endPoints["svconn"] = []
endPoints["dtc"] = []

with open('../'+ filename +".in") as f:
    [V,E,R,C,X] = [int(x) for x in next(f).split()];
    sVideos = [int(x) for x in next(f).split()];
    for i in range (0,E): 
        #Ld: latency to datacenter; K: # of cache servers connected
        [Ld,K] = [int(x) for x in next(f).split()];
        endPoints["dtc"].append(Ld);
        #c: id of the cache server; Lc: latency from server to the ith endpoint
        # cache server connection[i] = [[c Lc]k1, [c Lc]k2 ...]; 
        endPoints["svconn"].append(np.array( [[ int(x) for x in next(f).split() ] 
                                        for j in range (0,K)] ))

        # requisition[i] = [Rv,Re,Rn]
        #Rv: Id of the requested video; Re: id of the endpoint; Rn: # of requests
    req = np.array([[ int(x) for x in next(f).split() ] 
                for i in range(0,R)])        

    
sizeInd = [V,C];

popsize = 200;
GEN_MAX = 20;
g = 0;
maxGen = 0;

print('hi1')
pop = spawn_starting_population(popsize-1, sizeInd)
print('hi2')
#pop = population(20,8,1,0)
gen = []
gen.append(first_generation(pop))
#fitness_avg = np.array([sum(gen[0]['Fitness'])/
#                        len(gen[0]['Fitness'])])
fitness_max = np.array([max(gen[0]['Fitness'])])

res = open(Result_file, 'a')
res.write('\n'+str(gen)+'\n')
res.close()
finish = False



startTime = datetime. now();

while finish == False:
#    if max(fitness_max) > 6: 
#        break
#    if max(fitness_avg) > 5:
#        break
    
    #gen.append(next_generation(gen[-1],1,0))

    gen[-1] = next_generation(gen[-1],1,0);
    aux = max(gen[-1]['Fitness'])
    fitness_max = np.append(fitness_max, aux )  
    if aux > maxGen:        
        maxGen = aux;
        bestInd =  list(  gen[-1]['Individuals'][np.argmax(gen[-1]['Fitness'])])
#    fitness_avg = np.append(fitness_avg, sum(
#        gen[-1]['Fitness'])/len(gen[-1]['Fitness']))

    g += 1;
    print(g, maxGen)
#    res = open(Result_file, 'a')
#    res.write('\n'+str(gen[-1])+'\n')
#    res.close()
    if g == GEN_MAX :
        break

endTime = datetime.now()
endTimeFmt = endTime.strftime(" - %d-%m-%Y %H_%M - ");  

 
with open('./'+filename + "_" + endTimeFmt + "_out.in","w") as f:
    f.write(str(len(bestInd))+"\n" + str(' '.join(map(str,bestInd))));  


with open('./'+ filename + endTimeFmt + "statics.in","w") as f:
    f.write('Time lapsed: ' + str(endTime - startTime)); 
    f.write('\n# gen: ' + str(GEN_MAX) + '\npopsize: ' + str(popsize))
    f.write('\nBest fitness: '+ str(max(fitness_max)))
