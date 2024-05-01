import warnings, os
import numpy as np
from scipy import fft as spfft
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category= ConvergenceWarning)

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


def compressed_sensing(samples, alpha, domain= "IDCT"): # samples should be a 1d array with np.nans to signify the missing data
    total_points = len(samples) # number of pixels to reconstruct
    locations = np.nonzero(~np.isnan(samples)) # pixel numbers of the known points

    cropping_matrix = np.identity(total_points, dtype= np.float16)
    cropping_matrix = cropping_matrix[locations] #cropping matrix operator
    dct_matrix = spfft.idct(np.identity(total_points), axis= 0, norm= "forward")
    measurement_matrix = np.matmul(cropping_matrix, dct_matrix)

    lasso = Lasso(alpha= alpha)
    lasso.fit(measurement_matrix, samples[locations])

    if domain == "DCT":
        return lasso.coef_
    elif domain == "IDCT":
        result = spfft.idct(lasso.coef_, norm= "forward")
        return result
    else:
        raise ValueError("{0:} is not a valid domain. Try 'DCT' or 'IDCT'.".format(domain))


def subsample_1d(total_points, reduced_points, subsampling_method = "random"):

    if subsampling_method == "random":
        subsampled_points = np.random.choice(total_points, reduced_points, replace= False)
    elif subsampling_method == "regular":
        subsampled_points = np.round(np.linspace(0, total_points -1, reduced_points)).astype(int)
    elif subsampling_method == "centered":
        subsampled_points = np.arange((total_points-reduced_points)//2, (total_points+reduced_points)//2)

    subsampled_points = np.sort(subsampled_points) #Nessisary only for optimisation.

    return subsampled_points

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

def argmin(array): # numpy argmin always flattens the array
    return np.unravel_index(np.argmin(array, axis=None), array.shape)
