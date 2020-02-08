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
from itertools import combinations;

  


#eliminar números maiores que maxSlices, caso haja

#A naive recursive implementation of 0-1 Knapsack Problem 

# Returns the maximum value that can be put in a knapsack of 
# capacity W 
# A Dynamic Programming based Python Program for 0-1 Knapsack problem 
# Returns the maximum value that can be put in a knapsack of capacity W 

#OPTION OF APPROACH TO TEST
#   Sample the problem:
#   1- Define a number of steps to save - svSteps
#   2- run knapsack for the first svSteps knasack capacities ( range(0,svSteps))
#   3- feed the results to calculate next svSteps capacities and so one
def knapSack(W, wt, n): 

    K = [[0 for x in range(W+1)] for x in range(2)] 
	# Build table K[][] in bottom up manner 
    
    for i in range(n+1): 
#        print('\n--------- i=', i, 'wt[i-1]=',wt[i-1],'\n K:')
#        for x in K: print(x)
        for w in range(W+1): 
#            print('wt[i-1=',i-1,']: ', wt[i-1], 'w: ', w)
            if i==0 or w==0: 
                K[1][w] = 0
            elif wt[i-1] <= w: 
#                print('<= -- (w-wt[i-1]): ', (w-wt[i-1]), ' K[0][(w-wt[i-1])] ',  K[0][(w-wt[i-1])])
            
                K[1][w] = max(wt[i-1] + K[0][(w-wt[i-1])], K[0][w]) 
            else: 
                K[1][w] = K[0][w]             
#                print('> -- K[0][w]: ', K[0][w])
#            print('\nw= ',w)
#            for x in K: print(x)
        K[0] = K[1].copy()



    return K 


path = "./a_example"
path = "./b_small"
path = "./c_medium"
#path = "./d_quite_big";
#path = "./e_also_big";

with open(path+".in") as f:
    maxSlices, types = [int(x) for x in next(f).split()];
    slicesVec = [int(x) for x in next(f).split()];
    
n = len(slicesVec) 

# n amount of items = amount of pizza types
# W capacity of knapsack = quantitity of pizza slices to order

startTime = datetime. now();

score = 0; 

# To test above function 
val = slicesVec
wt = slicesVec
W = maxSlices
n = len(val) 
#
#val = [7, 8, 4] 
#wt = [3, 8, 6]  
#W = 10
#n = 3
#
#
K=knapSack(W , wt , n) 
print(K[1][-1])
# This code is contributed by Nikhil Kumar Singh 



endTime = datetime.now()
endTimeFmt = endTime.strftime(" - %d-%m-%Y %H_%M - ");  
  
#with open(path + "_" + endTimeFmt + "_out.in","w") as f:
#    f.write(str(len(bestInd))+"\n" + str(' '.join(map(str,bestInd))));  


with open(path + endTimeFmt + "statics.in","w") as f:
    f.write('Time lapsed: ' + str(endTime - startTime))
#    f.write('\n# gen: ' + str(GEN_MAX) + '\npopsize: ' + str(popsize))
#    f.write('\nBest fitness: '+ str(max(fitness_max)))
#    f.write('\nPoulation: '+ str(gen))# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:26:21 2020

@author: maria
"""

