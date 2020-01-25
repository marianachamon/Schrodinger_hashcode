# Python program of a space 
# optimized DP solution for 
# 0-1 knapsack problem. 

# val[] is for storing maximum 
# profit for each weight 
# wt[] is for storing weights 
# n number of item 
# W maximum capacity of bag 
# mat[2][W+1] to store final result 

def KnapSack(val, wt, n, W): 
	
	# matrix to store final result 
	mat = [[0 for i in range(W + 1)] 
			for i in range(2)] 
	# iterate through all items 
	i = 0
	while i < n: # one by one traverse 
				# each element 
		j = 0 # traverse all weights j <= W 
		
		# if i is odd that mean till 
		# now we have odd number of 
		# elements so we store result 
		# in 1th indexed row 
		if i % 2 == 0: 
			while j < W: # check for each value 
				j += 1
				if wt[i] <= j: # include element 
					mat[1][j] = max(val[i] + mat[0][j -
									wt[i]], mat[0][j]) 
				else: # exclude element 
					mat[1][j] = mat[0][j] 
					
		# if i is even that mean till 
		# now we have even number 
		# of elements so we store 
		# result in 0th indexed row 
		else: 
			while j < W: 
				j += 1
				if wt[i] <= j: 
					mat[0][j] = max(val[i] + mat[1][j -
									wt[i]], mat[1][j]) 
				else: 
					mat[0][j] = mat[1][j] 
		i += 1
	# Return mat[0][W] if n is 
	# odd, else mat[1][W] 
	if n % 2 == 0: 
		return mat[0][W] 
	else: 
		return mat[1][W] 

# Driver code 
val = [7, 8, 4] 
wt = [3, 8, 6] 
W = 10
n = 3
#print(KnapSack(val, wt, n, W)) 

# This code is contributed 
# by sahilshelangia 


#path = "./a_example";
path = "./b_small";
path = "./c_medium";
#path = "./d_quite_big";
#path = "./e_also_big";
with open(path+".in") as f:
    maxSlices, types = [int(x) for x in next(f).split()];
    slicesVec = [int(x) for x in next(f).split()];

n = len(slicesVec) 
#print( knapSack(maxSlices , slicesVec , slicesVec , n )) 
print(KnapSack(slicesVec, slicesVec, n, maxSlices)) 
