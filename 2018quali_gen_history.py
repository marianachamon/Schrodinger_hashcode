#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy.random import randint
from random import random as rnd
from random import gauss, randrange
import random
from datetime import datetime
import copy
import tqdm


# In[55]:
def analyze_car(car):
    k=0
    score = 0
    pos = np.array([0,0])
    totalSteps = 0 #actual step of the car, it doesn't move until the number of steps
    # required to move pass
    if len(car) == 0:
        return 0
#        score=0
    for i in range(0,T): 
        step = xy[car[k]]-ab[car[k]]  
        future_pos = pos+step

        if future_pos[0] > R or future_pos[0]<0 or future_pos[1]>C or future_pos[1]<0:
            return 0
        if i >= totalSteps+sum(abs(step)):            
            if sf[car[k],0] > i or sf[car[k],1] < i+sum(abs(step)):
                return 0
#                break
            else:
                if sf[car[k],0] == totalSteps: #bonus!                
                    score += B
                pos += step
                totalSteps = totalSteps + sum(abs(step))
                k += 1
                if k==len(car):
                    break
                
    score += totalSteps
    return score      

# In[ ]:

    
def fitness_calculation(individual):
#    fitness_value = np.dot(individual,slicesVec)
    carRides = [[] for i in range(F)] # rides assigned to each car
    score = 0
    for i in range(F):
        for j in range(len(individual)):
            if individual[j] == i:
                carRides[i].append(j)
    for i in range(F):
        score += analyze_car(carRides[i])
#    maxSlices=1
#    fitness_value=1
#    if fitness_value > maxSlices:
#        fitness_value = maxSlices - fitness_value
##        fitness_value = fitness_value - (maxSlices - fitness_value)
    return score


# In[ ]:


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


# In[ ]:



def pairing(elit1, selected1, method = 'Fittest'):
    elit = elit1
    selected = selected1
    individuals = [elit['Individuals']]+selected['Individuals']
    fitness = [elit['Fitness']]+selected['Fitness']
    if method == 'Fittest':
        parents = [[individuals[x],individuals[x+1]] 
                   for x in range(len(individuals)//2)]

    return parents


# In[ ]:



def mating(parents1, method='Single Point'):
    parents = parents1
    if method == 'Single Point':
        pivot_point = randint(1, len(parents[0]))
        offsprings = [parents[0]             [0:pivot_point]+parents[1][pivot_point:]]
        offsprings.append(parents[1]
            [0:pivot_point]+parents[0][pivot_point:])

    return offsprings


# In[ ]:


def existed(individual):
    if individual in existedInds:
        return True
    else:
        return False

def change_gen(individual,x):
            # -1 means no car is assigned to the ride
    r2 = random.sample(list(range(0, F-1))+[-1],F) # new car to assign, ensure it's different from the current
    r2.pop(individual[x]) # new car to assign, ensure it's different from the current
    individual[x] = r2[0]


# In[ ]:

def mutation(individual, upper_limit=0, lower_limit=1, muatation_rate=3, 
    method='Reset', standard_deviation = 0.001, choose_mut=1):

    if method == 'Reset':
        i=0
        mutated_individual = individual.copy()
        r = random.sample( range(0,len(individual)), len(individual) )  

        for i in range(0, len(individual)):
            change_gen(mutated_individual, r[i])
            if not existed(mutated_individual):                
                break
    return mutated_individual

# In[ ]:


def next_generation(gen1, upper_limit, lower_limit):
    elitg = {}
    next_gen = {}
 
    elitg['Individuals'] = gen1['Individuals'].pop(-1)
    elitg['Fitness'] = gen1['Fitness'].pop(-1)
    selected = selection(gen1)
    parents = pairing(elitg, selected) 
    #half of population was selected; the other half will be generated
    # we create parents to generate two children each couple
    #the children will compose the other half of new pop
    offsprings = [[[mating(parents[x])
                    for x in range(len(parents))]
                    [y][z] for z in range(2)] 
                    for y in range(len(parents))]
    offsprings1 = [offsprings[x][0]
                   for x in range(len(parents))]
    offsprings2 = [offsprings[x][1]
                   for x in range(len(parents))]
    

    unmutated = selected['Individuals'].copy()+offsprings1 + offsprings2
    mutated = [mutation(unmutated[x], upper_limit, lower_limit) 
        for x in range(len(gen1['Individuals']))] 
    unsorted_individuals = mutated + [elitg['Individuals']]
    unsorted_next_gen =  [fitness_calculation(mutated[x]) for x in range(len(mutated))]
    unsorted_fitness = [unsorted_next_gen[x]
        for x in range(len(gen['Fitness']))] + [elitg['Fitness']]
#    unsorted_fitness.append(elitg['Fitness']) #altered
    sorted_next_gen =         sorted([[unsorted_individuals[x], unsorted_fitness[x]]
            for x in range(len(unsorted_individuals))], 
                key=lambda x: x[1])
    next_gen['Individuals'] = [sorted_next_gen[x][0]
        for x in range(len(sorted_next_gen))]
    next_gen['Fitness'] = [sorted_next_gen[x][1]
        for x in range(len(sorted_next_gen))]

    return next_gen


# In[ ]:


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


# In[ ]:


def spawn_starting_population(amount,sizeInd):
    return [spawn_individual(sizeInd) for x in range (0,amount)]


# In[42]:


def spawn_individual(sizeInd): 
    return  [random.randint(0,F-1) for k in range(sizeInd)]
#individual: a vector of size N (# of rides), assigning a car to each ride


# In[61]:
    

filename = 'a_example'
#filename = 'b_should_be_easy'
ab = []
xy=[]
sf=[]
with open('./'+ filename +".in") as file:
    [R,C,F,N,B,T] = [int(k) for k in next(file).split()]
    for i in range(0,N): 
        #Ld: latency to datacenter; K: # of cache servers connected
        [a,b,x,y,s,f] = [int(k) for k in next(file).split()]
        ab.append([a,b])
        xy.append([x,y])
        sf.append([s,f])
ab = np.array(ab)
xy = np.array(xy)
sf = np.array(sf)

#%%

POPSIZE = 32;
GEN_MAX = 20;
g = 0;
maxGen = 0;
EXISTED_MAX = 100 # existed_max = 10000*2000/len(individual)
#ok tune: EXISTED_MAX = 10000, para D (len(individual)=2000)
#Com arquivo D, chegou a mais de 100.000 em 200 gerações, POPSIZE=500, mas ficou muito lento

#goodStart = 'd_quite_big_ - 26-01-2020 04_44 - _out.in'
 
#goodInd = getBestIndFromAnotherRunning(goodStart)
    
pop = spawn_starting_population(POPSIZE-1,N )
#pop = population(20,8,1,0)

gen = first_generation(pop)

fitness_max = np.array([max(gen['Fitness'])])

finish = False
existedInds = []
score= 1
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
    if score == 100 or g == GEN_MAX:
        break

endTime = datetime.now()
endTimeFmt = endTime.strftime(" - %d-%m-%Y %H_%M - ");  
 
path = './solutions/' + filename
with open(path + "_" + endTimeFmt + "_out.in","w") as f:
    f.write(str(len(bestInd))+"\n" + str(' '.join(map(str,bestInd))));  


with open(path + endTimeFmt + "statics.in","w") as f:
    f.write('Time lapsed: ' + str(endTime - startTime) + 'g: ' + str(g))
    f.write('\n# gen: ' + str(GEN_MAX) + '\nPOPSIZE: ' + str(POPSIZE))
    f.write('\nBest fitness: '+ str(max(fitness_max)))
    f.write('\nPoulation: '+ str(gen))


# In[ ]:



