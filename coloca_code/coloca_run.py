
#####################################################################################################
### Colocalization using Simulation Data to test the algorthms 

### Import related functions 
from package_func import *
from coloca_func import * 
from support_func import *

## Tag Distance betweeen two proteins for example  GPCR (R) and G-Protein (G), 0.01um = 10nm
tag_dist = 0.01
simu_area = 50
## Read the sample localization distribution
data = pd.read_csv('Example_SMLM.txt', delimiter='\t', encoding='ISO-8859-1')
filtered_data = data[(data['Precision [nm]'] >= 5) & (data['Precision [nm]'] <= 40)]
prec_channel_1 = filtered_data['Precision [nm]']/1000


def main(density):
    # Number of true pairs 
    num_points = int(density*simu_area*1/2)
    # Number of extra background points for each species R and G 
    num_R_extra = int(density*simu_area*1/2)
    num_G_extra = int(density*simu_area*1/2) 
    # Run the colocalization algorithm 
    num_pair_exp, num_pair_exp_mean, MC_last, MC_last_mean, pair_est_last, pair_est_last_mean = \
        run_exp_iterative_MC_parallel_two_channel(
            tag_dist = tag_dist, 
            num_points = num_points, 
            num_R_extra = num_R_extra, 
            num_G_extra = num_G_extra, 
            prec1 = prec_channel_1, 
            prec2 =  prec_channel_1, 
            d_true_thre = 20/1000, 
            dis_tree_thre_factor = 4, 
            area = simu_area, 
            iteration_full = 20, 
            iteration_in_step = 20, 
            iteration_step = 25, 
            num_MC_points= int(1e5)
        )


def run(d_values):
    for item in d_values:
        main(item)

if __name__ == "__main__":
    d_range = np.arange(4, 8, 4)
    run(d_range)
