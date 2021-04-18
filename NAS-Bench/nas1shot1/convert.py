import pickle
import pandas as pd
import os
import json
import glob

import numpy as np
from optimizers.utils import Model, Architecture

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3

from nasbench_analysis.utils import INPUT, OUTPUT, CONV1X1, NasbenchWrapper, upscale_to_nasbench_format, natural_keys

path = "experiments/discrete_optimizers/"
ssp = 3
algo = "EE"
n_runs = 500

search_space = eval('SearchSpace{}()'.format(ssp))
y_star_valid, y_star_test, inc_config = (search_space.valid_min_error,
                                         search_space.test_min_error, None)

class DotAccess():
    def __init__(self, valid, info, test):
        self.valid = valid
        self.info = info
        self.test = test
        
def process_and_save(all_runs):
    global y_star_valid, y_star_test
    valid_incumbents = []
    runtimes = []
    test_incumbents = []
    inc = np.inf
    test_regret = 1

    for k in range(len(all_runs)):
        print('Iteration {:<3}/{:<3}'.format(k+1, len(all_runs)), end="\r", flush=True)
        regret = all_runs[k].valid - y_star_valid
        # Update test regret only when incumbent changed by validation regret
        if regret <= inc:
            inc = regret
            test_regret = all_runs[k].test - y_star_test
        valid_incumbents.append(inc)
        test_incumbents.append(test_regret)
        runtimes.append(all_runs[k].info)
    runtimes = np.cumsum(runtimes).tolist()
    return valid_incumbents, runtimes, test_incumbents


# with open(os.path.join(path, 'config.json')) as fp:
#     config = json.load(fp)

re_archs = glob.glob(os.path.join(path, 'algo_{}_0_ssp_{}_seed_*.obj'.format(algo, ssp)))

# Sort them by date
re_archs.sort(key=natural_keys)

for i in range(n_runs):
    res = pickle.load(open(re_archs[i], 'rb'))
    all_runs = []
    for j in range(len(res)):
        all_runs.append(DotAccess(valid = 1 - res[j].validation_accuracy,
                                  info  = res[j].training_time,
                                  test  = 1 - res[j].test_accuracy))
        
    valid_incumbents, runtimes, test_incumbents = process_and_save(all_runs)
    directory  = os.path.join(path, '{}/{}/'.format(algo, ssp))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(path, '{}/{}/'
              'run_{}.json'.format(algo, ssp, i)), 'w') as f:
        json.dump({'runtime': runtimes, 'regret_validation': valid_incumbents,
                   'regret_test': test_incumbents}, f)
