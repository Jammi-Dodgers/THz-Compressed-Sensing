import os
import numpy as np
import CS_functions as cs

os.chdir(os.path.dirname(__file__))

file_number = 32
file_name = f"1dmockanderrors{file_number:d}"
file_type = ".csv"

optlocs_file = f"data/1dmockanderrors{file_number:d}_randoptlocs.csv"

target, target_err = cs.open_dataset(file_name, ".csv")

iterations = 1000
number_of_sensors = 8
interferogram_length = len(target)


for n in range(iterations):

    detectors, score = cs.simulated_annealing(number_of_sensors, target, np.ones_like(target_err),
                                              regularization_coeffient =1e-3,
                                              subsampling_method= "random",
                                              iterations= 5000,
                                              max_temp= 21,
                                              cooling= 0.998)

    cs.append_array_to_csv(detectors, optlocs_file)

    if n % 20 == 0:
        print("Iteration {:d} of {:d}".format(n, iterations))
