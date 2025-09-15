import warnings, os, random, math, itertools
import numpy as np
from scipy import fft as spfft, interpolate as spinterp
from scipy.constants import c as C
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category= ConvergenceWarning)

##############GENERIC AND BASIC FUNCTIONS##############

def argmin(array): # numpy argmin always flattens the array
    return np.unravel_index(np.argmin(array, axis=None), array.shape)

def chi_squared(measurement, model, uncertainty):
    return np.sum(((measurement -model)/uncertainty)**2)

def RSS(measurement, model):
    return np.sum((measurement -model)**2)

def closest(y, x, x0):
    idx = np.argmin(np.abs(x -x0))
    y0 = y[idx]
    return y0

def line_fit_through_origin(x, y):
    """Fits y = m*x using least squares, returns m."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    m = np.dot(x, y) / np.dot(x, x)
    return m

def gaussian(x, center, FWHM):
    sigma = (8 *np.log(2))**-0.5 *FWHM
    exponent = -(1/2) *(x -center)**2 /(sigma**2)
    if FWHM == 0: exponent[np.argmin(np.abs(x-center))] = 0.5
    normalisation_coeffient = 0.5 if FWHM == 0 else 0.5/np.sum(np.abs(np.exp(exponent)))
    #1 /(sigma *(2*np.pi)**0.5) # This is vunrable to numerical errors if the exponent is too large or too small.
    ## I've completely forgotten why but I'm normalising such that the intergral is 0.5.
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

def interpolate(y, method= "quadratic"):
    match method:
        case "pchip":
            index = np.arange(len(y))
            interp_index = np.linspace(0, len(y)-1, 10*len(y))
            interp = spinterp.PchipInterpolator(index, y) # discontinuities in the 2nd deriviative are allowed.
            return interp(interp_index)
        case "gaussian":
            index = np.atleast_2d(np.arange(len(y))).T
            interp_index = np.atleast_2d(np.linspace(0, len(y)-1, 10*len(y))).T
            interp = spinterp.RBFInterpolator(index, y, kernel= method, epsilon= 1) # not sure what this interpolater does?
            return interp(interp_index)
        case 'linear'|'nearest'|'nearest-up'|'zero'|'slinear'|'quadratic'|'cubic'|'previous'|'next':
            index = np.arange(len(y))
            interp_index = np.linspace(0, len(y)-1, 10*len(y))
            interp = spinterp.interp1d(index, y, kind= method)
            return interp(interp_index)
        case 'fourier':
            Y = np.fft.rfft(y, norm= "forward")
            Y = np.pad(Y, (0, int(9*len(Y)))) # ONLY WORKS IF THERE ARE NO HIGH FREQUENCY COMPONENTS
            return np.fft.irfft(Y, n= int(10*len(y)), norm= "forward")
        case _:
            print(f"ERROR! {method} is not a recognised interpolation method.")

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

def generate_interferogram(array_length, pixel_pitch, central_freq, FWHM_freq, theta, read_noise_sigma = 0): # (pixels), (m), (Hz), (Hz), (degrees), (as a fraction of the peak)
    central_wn = 2*np.sin(np.deg2rad(theta)) *(central_freq) /C #periodicity of the fringes as it appears on the camera in m^-1
    FWHM_wn = 2*np.sin(np.deg2rad(theta)) *(FWHM_freq) /C # in m^-1

    wns = np.fft.rfftfreq(array_length, pixel_pitch)
    amplitudes = gaussian(wns, central_wn, FWHM_wn)
    intensity = np.fft.irfft(amplitudes, norm= "forward", n= array_length)
    intensity = np.fft.fftshift(intensity)

    intensity += np.random.normal(0, read_noise_sigma,  array_length)

    return intensity

def generate_interferogram2(array_length, pixel_pitch, central_freqs, FWHM_envelope, theta, read_noise_sigma = 0, displacement_shift = 0): # (pixels), (m), (Hz), (m), (degrees), (as a fraction of the peak), (m)
    # FWHM_envelope should equal (2*C*ln(2)) / (sin(theta)*FWHM_freq*pi) but it is slightly different due to ?? windowing? picket fence effect?
    central_freqs = np.atleast_1d(central_freqs)
    interferogram = np.zeros(array_length, dtype= float)
    displacement = np.arange(-(array_length//2), (array_length+1)//2) *pixel_pitch +displacement_shift #in m

    for central_freq in central_freqs:
        kappa = 2*np.sin(np.deg2rad(theta))/C * central_freq # apparent wavenumber
        interferogram += np.cos(2*np.pi*displacement*kappa)

    interferogram *= np.exp(-4*np.log(2)*displacement**2 *FWHM_envelope**-2) # modulate by the beam size.
    interferogram += np.random.normal(0, read_noise_sigma, array_length)
    return interferogram

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
    if os.path.exists(csv_file):
        readwrite_mode = 'a' # append to the file
    else:
        readwrite_mode = 'w' # create a new file

    with open(csv_file, readwrite_mode) as file:
        array_string = np.array2string(array, separator=',').replace('\n', '')[1:-1]
        file.write(array_string +"\n")
        file.close()

############COMPRESSED SENSING FUNCTIONS#################

def compressed_sensing(samples, alpha, domain= "IDCT", ignore_mean= False, dct_type= 1, norm= "forward"): # samples should be a 1d array with np.nans to signify the missing data
    total_points = len(samples) # number of pixels to reconstruct
    locations = np.nonzero(~np.isnan(samples)) # pixel numbers of the known points

    cropping_matrix = np.identity(total_points, dtype= np.float16)
    cropping_matrix = cropping_matrix[locations] #cropping matrix operator
    dct_matrix = spfft.idct(np.identity(total_points), axis= 0, norm= norm, type= dct_type) # The transform does NOT get normalised by lasso and therefore the normalisation messes with alpha.
    measurement_matrix = np.matmul(cropping_matrix, dct_matrix)

    if norm == "ortho":
        # measurement_matrix *= np.sqrt(len(locations)) # normalisation used in the statistical learning with sparsity book. Very wierd, is this a mistake? "ortho" must be used.
        measurement_matrix *= np.sqrt(total_points) # standard normalisation used by most other papers. "ortho" must be used.

    lasso = Lasso(alpha= alpha, fit_intercept= ignore_mean)
    lasso.fit(measurement_matrix, samples[locations])

    if domain == "DCT":
        return lasso.coef_
    elif domain == "IDCT":
        result = spfft.idct(lasso.coef_, norm= norm, type= dct_type)
        if norm == "ortho": result *= np.sqrt(total_points)
        return result
    else:
        raise ValueError("{0:} is not a valid domain. Try 'DCT' or 'IDCT'.".format(domain))

def evaluate_score(detectors, targets, targets_uncertainty= None, noiseless= None, regularization_coeffient= 1e-3, error_type= "RSS"): # finds the MAXIMUM error from many interferograms.
    targets = np.atleast_2d(targets)

    if targets_uncertainty is None:
        targets_uncertainty = np.ones_like(targets)
    else:
        targets_uncertainty = np.atleast_2d(targets_uncertainty)

    if noiseless is None:
        noiseless = targets
        print("WARNING! Noiseless data not provided, assuming interferogram has no noise.")
    else:
        noiseless = np.atleast_2d(noiseless)

    score = []
    for target, uncertainty, noiseles in zip(targets, targets_uncertainty, noiseless):
        sample = np.full_like(target, np.nan)
        sample[detectors] = target[detectors]

        match error_type:
            case "RSS":
                result = compressed_sensing(sample, regularization_coeffient)
                error = RSS(target, result)
            case "chi squared":
                result = compressed_sensing(sample, regularization_coeffient)
                error = chi_squared(target, result, uncertainty)
            case "L2":
                result = compressed_sensing(sample, regularization_coeffient, domain= "DCT", norm= "ortho")
                noiseles_dct = spfft.dct(noiseles, norm= "ortho", type= 1) /np.sqrt(len(noiseles))
                error = np.linalg.norm(noiseles_dct -result) # L2 estimation error. estimation error is a crap name, it means the error in the DCT domain.
            case _:
                raise ValueError("{0:s} is not a recognised error type! Try 'RSS', 'L2' or 'chi squared'.".format(error_type))

        score.append(error)
    
    # score = max(score) # return the worst score
    score = np.percentile(score, 95) # return a bad score
    # score = np.mean(score) # return the average score

    return score

def RIP_from_Phi(Phi, s):
    """
    Generated by google gemini. RIP cannot be computed in polynomial time.
    Computes the Restricted Isometry Property (RIP) constant delta_s for a given
    matrix Phi and sparsity level s using the brute-force method.

    WARNING: This method has a computational complexity of O(C(n,s) * s^3),
    where C(n,s) is "n choose s". It is computationally infeasible for
    all but very small matrices and sparsity levels.

    Args:
        Phi (np.ndarray): The m x n sensing matrix.
        s (int): The sparsity level (the size of the submatrices to check).

    Returns:
        float: The RIP constant delta_s.
    """
    m, n = Phi.shape

    # --- Input Validation ---
    if not 1 <= s <= n:
        raise ValueError(f"Sparsity 's' must be between 1 and n (the number of columns), but got s={s} and n={n}.")

    print(f"Matrix dimensions (m, n): ({m}, {n})")
    print(f"Sparsity level (s): {s}")

    num_combinations = math.comb(n, s)
    print(f"Number of submatrices to check (C(n, s)): {num_combinations}\n")
    if num_combinations > 1_000_000:
        print("WARNING: The number of combinations is very large. This may take a very long time.")

    # --- Initialization ---
    overall_lambda_max = 0.0
    overall_lambda_min = np.inf

    # Get an iterator for all combinations of s column indices
    column_indices = range(n)
    combinations_iterator = itertools.combinations(column_indices, s)

    # --- Main Loop: Iterate through all submatrices ---
    for i, T in enumerate(combinations_iterator):
        if (i + 1) % 100000 == 0:
            print(f"  ... checked {i + 1} / {num_combinations} submatrices")

        # 1. Form the submatrix Phi_T
        Phi_T = Phi[:, list(T)] # Fancy indexing to select columns. ## WDYM fancy? This is just basic numpy stuff.

        # 2. Form the Gram matrix G_T = Phi_T^T * Phi_T
        # Using @ for matrix multiplication (requires Python 3.5+)
        G_T = Phi_T.T @ Phi_T

        # 3. Compute eigenvalues of the s x s Gram matrix.
        # np.linalg.eigvalsh is optimized for symmetric matrices like G_T.
        eigenvalues = np.linalg.eigvalsh(G_T)

        # 4. Update the overall min and max eigenvalues found so far
        overall_lambda_max = max(overall_lambda_max, np.max(eigenvalues))
        overall_lambda_min = min(overall_lambda_min, np.min(eigenvalues))

    # --- Final Calculation ---
    # delta_s is the maximum deviation from 1
    delta_s = max(overall_lambda_max - 1, 1 - overall_lambda_min)

    print("\n--- Computation Complete ---")
    print(f"Overall maximum eigenvalue found: {overall_lambda_max:.4f}")
    print(f"Overall minimum eigenvalue found: {overall_lambda_min:.4f}")

    return delta_s

def RIP(detector, target):
    total_points = len(target) # number of pixels to reconstruct
    dct_target = spfft.dct(target, norm= "ortho", type= 1)
    dct_target = np.abs(dct_target)
    sparsity = np.count_nonzero(dct_target > 0.1*dct_target.max())

    cropping_matrix = np.identity(total_points, dtype= np.float16)
    cropping_matrix = cropping_matrix[detector] #cropping matrix operator
    dct_matrix = spfft.idct(np.identity(total_points), axis= 0, norm= "ortho", type= 1)
    measurement_matrix = np.matmul(cropping_matrix, dct_matrix)

    return RIP_from_phi(measurement_matrix, sparsity)

############OPTIMISATION FUNCTIONS#################

def simulated_annealing(reduced_points, target, uncertainty= None, noiseless= None, regularization_coeffient =1e-3, error_type= "RSS", subsampling_method= "regular", min_seperation= 1, iterations= 10000, max_temp= 31, cooling= 0.995):

    if uncertainty is None:
        uncertainty = np.ones_like(target)
    if noiseless is None:
        noiseless = target
        print("WARNING! Noiseless data not provided, assuming interferogram has no noise.")

    temps = []
    scores = np.array([])
    total_points = len(target)
    detectors = subsample_1d(total_points, reduced_points, subsampling_method)
    score = new_score = evaluate_score(detectors, target, uncertainty, noiseless, regularization_coeffient, error_type)
    target_temp = max_temp
    improvement = True

    if reduced_points == len(target):
        print("VALUE ERROR! Signal is well sampled.")
        return detectors, score

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
        new_score = evaluate_score(new_detectors, target, uncertainty, noiseless, regularization_coeffient, error_type)

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


def MCMC_metropolis(reduced_points, target, uncertainty= None, noiseless= None, regularization_coeffient =1e-3, error_type= "RSS", subsampling_method= "regular", min_seperation= 1, iterations= 10000, stepsize= 3):

    if uncertainty is None:
        uncertainty = np.ones_like(target)
    if noiseless is None:
        noiseless = target
        print("WARNING! Noiseless data not provided, assuming interferogram has no noise.")

    total_points = len(target)

    detectors = subsample_1d(total_points, reduced_points, subsampling_method)
    detector_configerations = np.array(detectors)

    score = evaluate_score(detectors, target, uncertainty, noiseless, regularization_coeffient, error_type)
    scores = np.array([score])

    if reduced_points == len(target):
        print("VALUE ERROR! Signal is well sampled.")
        return detectors, score

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

        new_score = evaluate_score(new_detectors, target, uncertainty, noiseless, regularization_coeffient, error_type)
        acceptance = np.exp(1 -new_score/score) # Normally MCMC uses `new_score /score` but I am looking for a minimum point so this scheme is better.

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
    

def douglas_peucker(reduced_points, target, uncertainty= None, noiseless= None, regularization_coeffient =1e-3, error_type= "RSS"):

    if uncertainty is None:
        uncertainty = np.ones_like(target)
    if noiseless is None:
        noiseless = target
        print("WARNING! Noiseless data not provided, assuming interferogram has no noise.")

    detectors = np.array([], dtype= int)

    new_detector = np.argmax(np.abs(target)) # Without any samples, CS cannot find any frequencies so all amplitudes will go to zero. DP wants to locate the point that is furthest away from this zero line. Hence, this is a sensible way to intitalise the loop.
    detectors = np.append(detectors, new_detector)

    for n in range(1,reduced_points):
        samples = np.full_like(target, np.nan)
        samples[detectors] = target[detectors]
        result = compressed_sensing(samples, regularization_coeffient, ignore_mean= False, dct_type= 1, norm= "ortho")

        new_detector = np.argsort(np.abs(target -result))[::-1] # argsort sorts from smallest to largest but I want largest to smallest
        new_detector = np.setdiff1d(new_detector, detectors, assume_unique= True)[0] # pick the first (largest) item
        detectors = np.append(detectors, new_detector)

    score = evaluate_score(detectors, target, uncertainty, noiseless, regularization_coeffient, error_type)

    return detectors, score


def greedy(reduced_points, target, uncertainty= None, noiseless= None, regularization_coeffient =1e-3, error_type= "RSS", subsampling_method= "regular", iterations= 20):

    if uncertainty is None:
        uncertainty = np.ones_like(target)
    if noiseless is None:
        noiseless = target
        print("WARNING! Noiseless data not provided, assuming interferogram has no noise.")

    ################ INITIALISE AND RESET BRUTE FORCE ######################

    total_points = len(target)

    best_detectors = subsample_1d(total_points, reduced_points, subsampling_method)
    best_score = evaluate_score(best_detectors, target, uncertainty, noiseless, regularization_coeffient, error_type)

    ################# LIMITED BRUTE FORCE ###################

    for n in range(iterations):
        old_detectors = np.copy(best_detectors)

        pick_detectors = (index for index in range(reduced_points)) 
        for moving_detectors in pick_detectors:

            pick_samples = (index for index in range(total_points) if index not in old_detectors)
            for new_samples in pick_samples:
                detectors = np.copy(old_detectors)
                detectors[np.array(moving_detectors)] = new_samples

                score = evaluate_score(detectors, target, uncertainty, noiseless, regularization_coeffient, error_type)

                if score < best_score:
                    best_detectors = np.copy(detectors)
                    best_score = np.copy(score)

        if np.all([old_detectors == best_detectors]):
            break

    return best_detectors, best_score