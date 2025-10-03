<<<<<<< HEAD:compressed_sensing/CS_brute_force.py
#!/usr/bin/env python
# coding: utf-8

# _Version log: TRUE AND ULTIMATE BRUTE FORCE_

# In[1]:


import sys, itertools, random, math
import numpy as np
import CS_functions as cs
from matplotlib import pyplot as plt
from scipy import fft as spfft
=======
import os, random, math, time
import numpy as np
import CS_functions as cs
from matplotlib import pyplot as plt
os.chdir(os.path.dirname(__file__))
>>>>>>> 591a1f782899360c32c411574d659d001448ff5f:CS_brute_force.py

plt.rcParams.update({'font.size':16})
#np.set_printoptions(threshold=sys.maxsize)

<<<<<<< HEAD:compressed_sensing/CS_brute_force.py

# In[2]:


=======
>>>>>>> 591a1f782899360c32c411574d659d001448ff5f:CS_brute_force.py
# made by Thomas Lux on Stack Overflow
# Return a randomized "range" using a Linear Congruential Generator
# to produce the number sequence. Parameters are the same as for 
# python builtin "range".
#   Memory  -- storage for 8 integers, regardless of parameters.
#   Compute -- at most 2*"maximum" steps required to generate sequence.
#
def random_range(start, stop=None, step=None):
    # Set a default values the same way "range" does.
    if (stop == None): start, stop = 0, start
    if (step == None): step = 1
    # Use a mapping to convert a standard range into the desired range.
    mapping = lambda i: (i*step) + start
    # Compute the number of numbers in this range.
    maximum = (stop - start) // step
    # Seed range with a random integer.
    value = random.randint(0, maximum-1)
    # 
    # Construct an offset, multiplier, and modulus for a linear
    # congruential generator. These generators are cyclic and
    # non-repeating when they maintain the properties:
    # 
    #   1) "modulus" and "offset" are relatively prime.
    #   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    #   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4. #rule three seems a little arbitrary. Why does it matter and are there any other multiples that we need to watch out for?
    # 
    offset = random.randint(0, maximum-1) * 2 + 1      # Pick a random odd-valued offset.
    #multiplier = 4*(maximum//4) + 1                 # Pick a multiplier 1 greater than a multiple of 4.
    modulus = int(2**math.ceil(math.log2(maximum))) # Pick a modulus just big enough to generate all numbers (power of 2).
    # Track how many random numbers have been returned.
    found = 0
    while found < maximum:
        # If this is a valid value, yield it in generator fashion.
        if value < maximum:
            found += 1
            yield mapping(value)
        # Calculate the next value in the sequence.
        #value = (value*multiplier + offset) % modulus
        value = (value +offset) % modulus #removing the multiplier makes it less random but more reliable for extremely large numbers (>1e13)
<<<<<<< HEAD:compressed_sensing/CS_brute_force.py
    

# made by openai (but debugged by me because robots are not taking over the world anytime soon!)
def find_nth_combination(N, r, idx):
    num_combinations = math.comb(N, r)
    
    if num_combinations <= idx :
        raise ValueError("idx is larger than the total number of combinations")
    
    result = []
    n = 0
    while r > 0:
        num_combinations = math.comb(N -n -1, r - 1)
        if num_combinations <= idx:
            n += 1
            idx -= num_combinations
        else:
            result.append(n)
            n += 1
            r -= 1
        
        if r == 0:
            return tuple(result)
    
    return None


# In[3]:


file_name = "1dmockanderrors16"
file_type = ".csv"

optlocs_file = "data\\" + file_name +"_optlocs.csv"
target, uncertainties = cs.open_dataset(file_name, file_type)
total_points = len(target)


# In[4]:


reduced_points = 4
regularization_coeffient = 1e-4

#initial_detectors = [] #custom
initial_detectors = cs.subsample_1d(total_points, reduced_points, subsampling_method= "centered")


# In[5]:
=======


file_name = "1dmockanderrors31"
training_file_name = "1dmockanderrors32"
file_type = ".csv"

optlocs_file = "data\\" + file_name +"_optlocs.csv"
target, target_err = cs.open_dataset(file_name, file_type)
training_target, training_err = cs.open_dataset(training_file_name, file_type)
total_points = len(target)


reduced_points = 45
regularization_coeffient = 1e-2
number_of_combonations = math.comb(total_points, reduced_points)
>>>>>>> 591a1f782899360c32c411574d659d001448ff5f:CS_brute_force.py


################ INITIALISE AND RESET BRUTE FORCE ######################

<<<<<<< HEAD:compressed_sensing/CS_brute_force.py
best_detectors = cs.subsample_1d(total_points, reduced_points, "regular")

best_score = cs.evaluate_score(best_detectors, target, uncertainties, regularization_coeffient)

combo_generator = (find_nth_combination(total_points, reduced_points, random_index) for random_index in random_range(math.comb(total_points, reduced_points)))


# In[6]:

=======
start_time = time.time()

combo_generator = (cs.find_nth_combination(total_points, reduced_points, random_index) for random_index in random_range(math.comb(total_points, reduced_points)))

iterations = 0; best_iteration = 0

best_detectors = np.array(next(combo_generator)) #cs.subsample_1d(total_points, reduced_points, "regular")
best_score = cs.evaluate_score(best_detectors, target, target_err, training_target, regularization_coeffient, "L2")
>>>>>>> 591a1f782899360c32c411574d659d001448ff5f:CS_brute_force.py

################# TRUE BRUTE FORCE ####################

for detectors in combo_generator: # THIS ITERABLE IS DANGEROUS!
<<<<<<< HEAD:compressed_sensing/CS_brute_force.py
    detectors = np.array(detectors)

    score = cs.evaluate_score(detectors, target, uncertainties, regularization_coeffient)
=======
    iterations += 1
    detectors = np.array(detectors)

    score = cs.evaluate_score(detectors, target, target_err, training_target, regularization_coeffient, "L2")
>>>>>>> 591a1f782899360c32c411574d659d001448ff5f:CS_brute_force.py

    if score < best_score:
        best_score = score
        cs.append_array_to_csv(detectors, optlocs_file)
        print("new best saved!")
<<<<<<< HEAD:compressed_sensing/CS_brute_force.py


# In[ ]:




=======
        best_iteration = iterations
        best_detectors = np.copy(detectors)
    if not iterations % 1000000: # give a progress update every million iterations
        print("{0:d} iterations complete. {1:.1f}% done".format(iterations, 100*iterations/number_of_combonations))

runtime = time.time() -start_time
print(f"Brute Force searched for {runtime} seconds and found a solution with a score of {best_score} after {runtime *best_iteration/(iterations+1)} seconds")
print(*best_detectors, sep= ",")
>>>>>>> 591a1f782899360c32c411574d659d001448ff5f:CS_brute_force.py
