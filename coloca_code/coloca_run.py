#!/usr/bin/env python
#####################################################################################################
### Colocalization using Simulation Data to test the algorthms 
# from scaling import max_weight_matching_scaling
# from gabow import gabow_tarjan_bipartite_matching
# from fibonacci import max_weight_matching_fb
### Function to import other colocalization functions
from package_func import *
from coloca_func import * 
from support_func import *
import multiprocessing
## Tag Distance betweeen the GPCR and G-Protein 
tag_dist = 0.01
## Number of True Pairs and number of extra/background points added

data = pd.read_csv('Example.txt', delimiter='\t', encoding='ISO-8859-1')
filtered_data = data[(data['Precision [nm]'] >= 5) & (data['Precision [nm]'] <= 40)]
prec_channel_1 = filtered_data['Precision [nm]']/1000


def main(density):
    num_points = int(density*100*1/2)
    num_R_extra = int(density*100*1/2)
    num_G_extra = int(density*100*1/2) 
    num_pair_exp, num_pair_exp_mean, MC_last, MC_last_mean, pair_est_last, pair_est_last_mean = \
        run_exp_iterative_MC_parallel_two_channel(tag_dist = tag_dist, num_points = num_points, \
            num_R_extra = num_R_extra, num_G_extra = num_G_extra, \
            prec1 = prec_channel_1, prec2 =  prec_channel_1, \
            d_true_thre = 20/1000, dis_tree_thre_factor = 4, area = 100, \
            iteration_full = 20, iteration_in_step = 20, iteration_step = 25, num_MC_points= int(1e5))

def run_parallel(d_values):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Use all available CPU cores
    pool.map(main, d_values)
    pool.close()
    pool.join()

def run(d_values):
    for item in d_values:
        main(item)

if __name__ == "__main__":
    d_range = np.arange(4, 8, 4)
    # run_parallel(d_range)
    run(d_range)
