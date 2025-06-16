import warnings, os, random, math
import numpy as np
from scipy import fft as spfft
from scipy.constants import c as C
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category= ConvergenceWarning)

##############GENERIC AND BASIC FUNCTIONS##############

def argmin(array): # numpy argmin always flattens the array
    return np.unravel_index(np.argmin(array, axis=None), array.shape)

def gaussian(x, center, FWHM):
    sigma = (8 *np.log(2))**-0.5 *FWHM
    exponent = -(1/2) *(x -center)**2 /(sigma**2)
    normalisation_coeffient = 0.5/np.sum(np.abs(np.exp(exponent))) #1 /(sigma *(2*np.pi)**0.5) # This is vunrable to numerical errors if the exponent is too large or too small.
    return normalisation_coeffient *np.exp(exponent)

def subsample_1d(total_points, reduced_points, subsampling_method = "random"):

    if subsampling_method == "random":
        subsampled_points = np.random.choice(total_points, reduced_points, replace= False)
    elif subsampling_method == "regular":
        subsampled_points = np.round(np.linspace(0, total_points -1, reduced_points)).astype(int)
    elif subsampling_method == "centered":
        subsampled_points = np.arange((total_points-reduced_points)//2, (total_points+reduced_points)//2)

    subsampled_points = np.sort(subsampled_points) #Nessisary only for optimisation.

    return subsampled_points

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

def generate_interferogram(array_length, pixel_pitch, central_freq, FWHM_freq, theta, read_noise_sigma = 0): # (pixels), (m), (Hz), (Hz), (degrees), (as a fraction of the peak)
    central_wn = 2*np.sin(np.deg2rad(theta)) *(central_freq) /C #periodicity of the fringes as it appears on the camera in m^-1
    FWHM_wn = 2*np.sin(np.deg2rad(theta)) *(FWHM_freq) /C # in m^-1

    wns = np.fft.rfftfreq(array_length, pixel_pitch)
    amplitudes = gaussian(wns, central_wn, FWHM_wn)
    intensity = np.fft.irfft(amplitudes, norm= "forward", n= array_length)
    intensity = np.fft.fftshift(intensity)

    intensity += np.random.normal(0, read_noise_sigma,  array_length)

    return intensity

############FILE ORGANISATION FUNCTIONS#################

def open_dataset(file_name, file_type):
    if file_type == ".csv":
        array = np.genfromtxt("data\\" +file_name +file_type, delimiter=",", filling_values= np.nan)
        if array.ndim == 2:
            return array.T
    elif file_type == ".txt":
        array = np.genfromtxt("data\\" +file_name +file_type, delimiter=",", filling_values= np.nan)
    else:
        raise ValueError("{0:} is not a recognised file type.".format(file_type))
    return array

def open_training_dataset(training_dataset_number):

    training_directory = "data\\training_set{0:}\\".format(training_dataset_number)
    training_file_paths = [os.path.join(training_directory, file_name) for file_name in os.listdir(training_directory)]

    training_data = np.array([np.genfromtxt(file_path, delimiter=",", filling_values= np.nan) for file_path in training_file_paths])

    training_data = np.rollaxis(training_data, -1, 0) # move the last axis to the front
    return training_data # training_interferograms, training_uncertainty = training_data  # now we can seperate the interferograms from the uncertainties. :)

def open_csv(optlocs_file, number_of_columns= None): #works with inconsistant numbers of delimiters
    with open(optlocs_file, 'r') as file:
        lines = [line[:-1] for line in list(file)]
        if number_of_columns != None:
            lines = [line for line in lines if line.count(",") == number_of_columns-1] # filter by number of samples
        number_of_delimiters = [line.count(",") for line in lines]
        max_delimiters = max(number_of_delimiters)
        missing_delimiters = [max_delimiters -delimiters for delimiters in number_of_delimiters]
        data = [line.split(",") for line in lines]
        data = [[int(datapoint) for datapoint in line] for line in data] #2D list comprehention!!!!
        full_data = [data[n] + [np.nan]*missing_delimiters[n] for n in range(len(lines))]
        full_data = np.array(full_data)
        file.close()

    return full_data

def append_array_to_csv(array, csv_file):
    with open(csv_file, 'a') as file:
        array_string = np.array2string(array, separator=',').replace('\n', '')[1:-1]
        file.write(array_string +"\n")
        file.close()

############COMPRESSED SENSING FUNCTIONS#################

def compressed_sensing(samples, alpha, domain= "IDCT", ignore_mean= False, dct_type= 1): # samples should be a 1d array with np.nans to signify the missing data
    total_points = len(samples) # number of pixels to reconstruct
    locations = np.nonzero(~np.isnan(samples)) # pixel numbers of the known points

    cropping_matrix = np.identity(total_points, dtype= np.float16)
    cropping_matrix = cropping_matrix[locations] #cropping matrix operator
    dct_matrix = spfft.idct(np.identity(total_points), axis= 0, norm= "forward", type= dct_type)
    measurement_matrix = np.matmul(cropping_matrix, dct_matrix)

    lasso = Lasso(alpha= alpha, fit_intercept= ignore_mean)
    lasso.fit(measurement_matrix, samples[locations])

    if domain == "DCT":
        return lasso.coef_
    elif domain == "IDCT":
        result = spfft.idct(lasso.coef_, norm= "forward", type= dct_type)
        return result
    else:
        raise ValueError("{0:} is not a valid domain. Try 'DCT' or 'IDCT'.".format(domain))

def evaluate_score(detectors, targets, targets_uncertainty, regularization_coeffient= 1e-3, domain= "IDCT"): # finds the MAXIMUM chi-square from many interferograms.
    targets = np.atleast_2d(targets)
    targets_uncertainty = np.atleast_2d(targets_uncertainty)

    score = 0
    for target, uncertainty in zip(targets, targets_uncertainty):
        sample = np.full_like(target, np.nan)
        sample[detectors] = target[detectors]

        match domain:
            case "IDCT":
                result = compressed_sensing(sample, regularization_coeffient)
                chi_square = np.linalg.norm((target -result) /uncertainty) #This is the chi-squared
            case "DCT":
                result_DCT = compressed_sensing(sample, regularization_coeffient, domain= "DCT", dct_type= 1)
                target_DCT = spfft.dct(target, norm= "forward", type= 1)
                chi_square = np.linalg.norm((target_DCT -result_DCT)) #This is the least squares
            case "FFT":
                result = compressed_sensing(sample, regularization_coeffient)
                result_powspec = np.abs(np.fft.rfft(result, norm= "ortho"))
                target_powspec = np.abs(np.fft.rfft(target, norm= "ortho"))
                chi_square = np.linalg.norm((target_powspec -result_powspec)) #This is the least squares
            case _:
                raise ValueError("{0:s} is not a recognised domain! Try 'IDCT', 'DCT' or 'FFT'.".format(domain))

        if chi_square > score:
            score = chi_square

    return score

############OPTIMISATION FUNCTIONS#################

def simulated_annealing(reduced_points, target, uncertainty, regularization_coeffient =1e-3, subsampling_method= "regular", min_seperation= 1, iterations= 30000, max_temp= 31, cooling= 0.998):

    temps = []
    scores = np.array([])
    total_points = len(target)
    detectors = subsample_1d(total_points, reduced_points, subsampling_method)
    score = new_score = evaluate_score(detectors, target, uncertainty, regularization_coeffient)
    target_temp = max_temp
    improvement = True

    #######START SIMULATED ANNEALLING###########
    for n in range(iterations):
        t = round(target_temp) #reset steps
        new_detectors = np.copy(detectors) #reset detectors
        new_score = np.copy(score) #reset score

        while t > 0:
            random_detector = np.random.randint(0, reduced_points) #random number between 0 and reduced_points. Includes 0. Excludes reduced_points
            current = new_detectors[random_detector]
            previous = -1 if random_detector == 0 else new_detectors[random_detector -1] #consider making the end points fixed. It helps define the length of the detector array.
            next = total_points if random_detector == reduced_points -1 else new_detectors[random_detector +1]
            if previous +min_seperation < current and current < next -min_seperation:
                #detector has space to move forward or back.
                new_detectors[random_detector] += np.random.choice([-1,1])
                t -= 1
            elif previous +min_seperation < current:
                #detector has space to move back.
                new_detectors[random_detector] -= 1
                t -= 1
            elif current < next -min_seperation:
                #detector has space to move forward.
                new_detectors[random_detector] += 1
                t -= 1
            else:
                #detector can't move.
                pass

        temps = temps + [[target_temp, np.linalg.norm(new_detectors -detectors, ord= 1)]] #L1 norm represents the number of times that the detectors were moved
        new_score = evaluate_score(new_detectors, target, uncertainty, regularization_coeffient)

        if new_score < score:
            detectors = new_detectors
            score = new_score
            improvement = True

        if target_temp <= 1: #When cold, stop optimising and start exploring new possiblities.
            target_temp = max_temp
            improvement = False
        elif improvement: #When hot, stop exploring and start optimising this regime.
            target_temp *= cooling

        scores = np.append(scores, score)

    temps = np.array(temps).T

    return detectors, score


def MCMC_metropolis(reduced_points, target, uncertainty, regularization_coeffient =1e-3, subsampling_method= "regular", min_seperation= 1, iterations= 30000, stepsize= 31):

    total_points = len(target)

    detectors = subsample_1d(total_points, reduced_points, subsampling_method)
    detector_configerations = np.array(detectors)

    score = evaluate_score(detectors, target, uncertainty, regularization_coeffient)
    scores = np.array([score])

    #######START MCMC Metropolis###########
    for n in range(iterations):
        steps = stepsize #reset steps
        new_detectors = detectors #reset detectors
        new_score = score #reset score

        while steps > 0:
            random_detector = np.random.randint(0, reduced_points) #random number between 0 and reduced_points. Includes 0. Excludes reduced_points

            current = new_detectors[random_detector]
            previous = -1 if random_detector == 0 else new_detectors[random_detector -1]
            next = total_points if random_detector == reduced_points -1 else new_detectors[random_detector +1]

            if previous +min_seperation < current and current < next -min_seperation:
                #detector has space to move forward or back.
                new_detectors[random_detector] += np.random.choice([-1,1])
                steps -= 1
            elif previous +min_seperation < current:
                #detector has space to move back.
                new_detectors[random_detector] -= 1
                steps -= 1
            elif current < next -min_seperation:
                #detector has space to move forward.
                new_detectors[random_detector] += 1
                steps -= 1
            else:
                #detector can't move.
                pass

        new_score = evaluate_score(new_detectors, target, uncertainty, regularization_coeffient)
        acceptance = np.exp(score -new_score) # Normally MCMC uses `new_score /score` but I am looking for a minimum point so this scheme is better.

        detector_configerations = np.vstack((detector_configerations, new_detectors))

        if acceptance > np.random.rand():
            detectors = new_detectors
            score = new_score

        scores = np.append(scores, score)


    ###FINALISATION AFTER LOOP

    best_iteration = np.argmin(scores)
    detectors = detector_configerations[best_iteration]
    score = scores[best_iteration]

    return detectors, score
    

def douglas_peucker(reduced_points, target, uncertainty, regularization_coeffient =1e-3):

    detectors = np.array([], dtype= int)

    new_detector = np.argmax(np.abs(target)) # Without any samples, CS cannot find any frequencies so all amplitudes will go to zero. DP wants to locate the point that is furthest away from this zero line. Hence, this is a sensible way to intitalise the loop.
    detectors = np.append(detectors, new_detector)

    for n in range(1,reduced_points):
        samples = np.full_like(target, np.nan)
        samples[detectors] = target[detectors]
        result = compressed_sensing(samples, regularization_coeffient, ignore_mean= False, dct_type= 1)

        new_detector = np.argsort(np.abs(target -result))[::-1] # argsort sorts from smallest to largest but I want largest to smallest
        new_detector = np.setdiff1d(new_detector, detectors, assume_unique= True)[0] # pick the first (largest) item
        detectors = np.append(detectors, new_detector)

    score = evaluate_score(detectors, target, uncertainty, regularization_coeffient)

    return detectors, score


def greedy(reduced_points, target, uncertainty, regularization_coeffient =1e-3, subsampling_method= "regular", iterations= 20):

    ################ INITIALISE AND RESET BRUTE FORCE ######################

    total_points = len(target)

    best_detectors = subsample_1d(total_points, reduced_points, subsampling_method)
    best_score = evaluate_score(best_detectors, target, uncertainty, regularization_coeffient)

    ################# LIMITED BRUTE FORCE ###################

    for n in range(iterations):
        old_detectors = np.copy(best_detectors)

        pick_detectors = (find_nth_combination(reduced_points, 1, random_index) for random_index in random_range(math.comb(reduced_points, 1)))
        for moving_detectors in pick_detectors:

            pick_samples = (find_nth_combination(total_points, 1, random_index) for random_index in random_range(math.comb(total_points, 1)))
            for new_samples in pick_samples:
                detectors = np.copy(old_detectors)
                detectors[np.array(moving_detectors)] = new_samples

                score = evaluate_score(detectors, target, uncertainty, regularization_coeffient)

                if score < best_score:
                    best_detectors = np.copy(detectors)
                    best_score = np.copy(score)

        if np.all([old_detectors == best_detectors]):
            break

    return best_detectors, best_score