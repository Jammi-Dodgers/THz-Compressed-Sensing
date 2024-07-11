import sys, random, math, time
import numpy as np
import CS_functions as cs
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size':16})
#np.set_printoptions(threshold=sys.maxsize)

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


file_name = "1dmockanderrors12"
file_type = ".csv"

optlocs_file = "data\\" + file_name +"_optlocs.csv"
target, uncertainties = cs.open_dataset(file_name, file_type)
total_points = len(target)


reduced_points = 4
regularization_coeffient = 1e-3
number_of_combonations = math.comb(total_points, reduced_points)

#initial_detectors = [] #custom
initial_detectors = cs.subsample_1d(total_points, reduced_points, subsampling_method= "centered")


################ INITIALISE AND RESET BRUTE FORCE ######################

start_time = time.time()

best_detectors = cs.subsample_1d(total_points, reduced_points, "regular")

best_score = cs.evaluate_score(best_detectors, target, uncertainties, regularization_coeffient)

combo_generator = (find_nth_combination(total_points, reduced_points, random_index) for random_index in random_range(math.comb(total_points, reduced_points)))

################# TRUE BRUTE FORCE ####################

iterations = 0
for detectors in combo_generator: # THIS ITERABLE IS DANGEROUS!
    iterations += 1
    detectors = np.array(detectors)

    score = cs.evaluate_score(detectors, target, uncertainties, regularization_coeffient)

    if score < best_score:
        best_score = score
        cs.append_array_to_csv(detectors, optlocs_file)
        print("new best saved!")
    if not iterations % 1000000: # give a progress update every million iterations
        print("{0:d} iterations complete. {1:.1f}% done".format(iterations, 100*iterations/number_of_combonations))

end_time = time.time()

print(end_time -start_time)