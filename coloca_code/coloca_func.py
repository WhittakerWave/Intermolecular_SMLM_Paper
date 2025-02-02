
#####################################################################################################
from package_func import *
from support_func import *
#####################################################################################################
### Method I: Using Pairwise distance
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def closest_pairs_index(R_filtered, G_filtered, threshold):
    ## Hard thresholding, doesn't depend on precision
    ## Or use cKDTree and query_ball_point
    R_tree = KDTree(R_filtered)
    G_tree = KDTree(G_filtered)
    pairs_R = G_tree.query_radius(R_filtered, threshold)
    pairs_G = R_tree.query_radius(G_filtered, threshold)
    pairs_distance = []
    for i, indices in enumerate(pairs_R):
        for j in indices:
            dist = np.linalg.norm(R_filtered[i] - G_filtered[j])
            pairs_distance.append((dist, [i, j]))
    for i, indices in enumerate(pairs_G):
        for j in indices:
            dist = np.linalg.norm(G_filtered[i] - R_filtered[j])
            pairs_distance.append((dist, [j, i]))
    pairs_distance.sort(key=lambda x: x[0])
    final_pairs = []
    used_points_R = set()
    used_points_G = set()
    for dist, pair in pairs_distance:
         if pair[0] not in used_points_R and pair[1] not in used_points_G:
            used_points_R.add(pair[0])
            used_points_G.add(pair[1])
            final_pairs.append((dist, pair))
    return final_pairs

def closest_pairs_index_precision(R_pos, G_pos, R_prec, G_prec, factor, threshold, R_frame, G_frame):
    ## Combine the precision
    ## Or use cKDTree and query_ball_point
    R_tree = KDTree(R_pos)
    G_tree = KDTree(G_pos)
    pairs_R = G_tree.query_radius(R_pos, threshold)
    pairs_G = R_tree.query_radius(G_pos, threshold)
    pairs_distance = []
    for i, indices in enumerate(pairs_R):
        for j in indices:
            dist = np.linalg.norm(R_pos[i] - G_pos[j])
            pairs_distance.append((dist, [i, j]))
    for i, indices in enumerate(pairs_G):
        for j in indices:
            dist = np.linalg.norm(G_pos[i] - R_pos[j])
            pairs_distance.append((dist, [j, i]))
    pairs_distance.sort(key=lambda x: x[0])
    final_pairs = []
    used_points_R = set()
    used_points_G = set()
    frame_pairs = []
    for dist, pair in pairs_distance:
        vary_thre = factor * np.sqrt(R_prec[pair[0]]**2 + G_prec[pair[1]]**2)
        if dist <= vary_thre:
            if pair[0] not in used_points_R and pair[1] not in used_points_G:
                used_points_R.add(pair[0])
                used_points_G.add(pair[1])
                if R_frame is not None and G_frame is not None:
                    frame_pairs.append([R_frame[pair[0]], G_frame[pair[1]]])
                final_pairs.append((dist, pair))
    return final_pairs, frame_pairs

#####################################################################################################
#### Prob pf the pair by the Monte Carlo estimation 
def prob_pair(point1, point2, sigma1, sigma2, num_points, threshold):
    ## Sample the first point with precision sigma1
    point1_prec = np.random.normal(loc = point1, scale = sigma1, size=(num_points, 2))
    ## Sample the second point with precision sigma2
    point2_prec = np.random.normal(loc = point2, scale = sigma2, size=(num_points, 2))
    distances = np.sqrt(np.sum((point2_prec - point1_prec)**2, axis=1))
    # Count the number of points that fall within the circle of radius r
    num_points_within_circle = np.count_nonzero(distances <= threshold)
    probability = num_points_within_circle / num_points
    return probability
# probability = prob_pair([0,0], [0,0], sigma1 = [20,20], sigma2 = [20,20], num_points = 10**4, threshold = 1.24*20)

def closest_pairs_index_prec(R_filtered, G_filtered, R_prec, G_prec, factor, threshold, R_frame, G_frame):
    ## Combine the precision and using the real expression to find the maximum pairs
    ## Or use cKDTree and query_ball_point
    R_tree = KDTree(R_filtered)
    G_tree = KDTree(G_filtered)
    pairs_R = G_tree.query_radius(R_filtered, threshold)
    pairs_G = R_tree.query_radius(G_filtered, threshold)
    pairs_distance = []
    for i, indices in enumerate(pairs_R):
        for j in indices:
            dist = np.linalg.norm(R_filtered[i] - G_filtered[j])
            pairs_distance.append((dist, [i, j]))
    for i, indices in enumerate(pairs_G):
        for j in indices:
            dist = np.linalg.norm(G_filtered[i] - R_filtered[j])
            pairs_distance.append((dist, [j, i]))
    pairs_distance.sort(key=lambda x: x[0])
    final_pairs = []
    used_points_R = set()
    used_points_G = set()
    frame_pairs = []
    for dist, pair in pairs_distance:
        vary_thre = factor * np.sqrt(R_prec[pair[0]]**2 + G_prec[pair[1]]**2)
        if dist <= vary_thre:
            if pair[0] not in used_points_R and pair[1] not in used_points_G:
                used_points_R.add(pair[0])
                used_points_G.add(pair[1])
                final_pairs.append((dist, pair))
                if R_frame is not None and G_frame is not None:
                    frame_pairs.append([R_frame[pair[0]], G_frame[pair[1]]])
    return final_pairs, frame_pairs

def pair_matching_max_weight_nx(R_pos, G_pos, R_prec, G_prec, d_true_thre, dis_tree_thre_factor, num_MC_points):
    ### Using NetworkX Package to find the pairs
    R_tree = KDTree(R_pos)
    G_tree = KDTree(G_pos)
    ### new maxifum dis_tree_thre 
    dis_tree_thre = np.max(np.concatenate([R_prec, G_prec]))*np.sqrt(2)*dis_tree_thre_factor
    ### Query a ball with some threshold distance
    pairs_R = G_tree.query_radius(R_pos, dis_tree_thre)
    pairs_G = R_tree.query_radius(G_pos, dis_tree_thre)
    pairs_distance = []
    for i, indices in enumerate(pairs_R):
        for j in indices:
            dist = np.linalg.norm(R_pos[i] - G_pos[j])
            if dist/np.sqrt(R_prec[i]**2 + G_prec[j]**2) <= dis_tree_thre_factor:
                pairs_distance.append((dist, [i, j]))
    for i, indices in enumerate(pairs_G):
        for j in indices:
            dist = np.linalg.norm(G_pos[i] - R_pos[j])
            if dist/np.sqrt(G_prec[i]**2 + R_prec[j]**2) <= dis_tree_thre_factor:
                pairs_distance.append((dist, [j, i]))
    weights_matrix = np.zeros((len(R_pos), len(G_pos)))
    for _, item in pairs_distance:
        item[0] = int(item[0])
        item[1] = int(item[1])
        prob = prob_pair(R_pos[item[0]], G_pos[item[1]], R_prec[item[0]], G_prec[item[1]], num_MC_points, d_true_thre)
        weights_matrix[item[0], item[1]] = prob
    # create an empty bipartite graph
    # weights_matrix = np.array(weights_matrix).astype(float)
    # weights_matrix = np.where(weights_matrix > 0, weights_matrix, 0)
    weights_matrix = np.array(weights_matrix*100000).astype(int)
    G = nx.Graph()
    # add the vertices from each partition
    G.add_nodes_from(range(len(R_pos)), bipartite=0)
    G.add_nodes_from(range(len(R_pos), len(R_pos)+len(G_pos)), bipartite=1)
    # add the weighted edges between the vertices that need to be matched
    for i in range(len(R_pos)):
        for j in range(len(G_pos)):
            if weights_matrix[i,j]!= 0:
                G.add_edge(i, j+len(R_pos), weight = weights_matrix[i,j])
    # compute the maximum weight matching
    matching = nx.max_weight_matching(G, maxcardinality=False, weight='weight')
    left_nodes = set(n for n, d in G.nodes(data=True) if d['bipartite']==0)
    right_nodes = set(G) - left_nodes
    matched_edges = [(i, j) if i in left_nodes else (j, i) for i, j in matching]
    ## renormalized the second coordinate to j-len(R_pos)
    matched_edges1 = [(i, j-len(R_pos)) for i, j in matched_edges]
    return  matched_edges1

########################################################################################################
#### Function to run the summary of the pairs
def summary_pairs(pair, num_points, num_R_extra, num_G_extra):
    ### Need two filtering: 1) same or not 2) smaller than num_points to ensure recall is smaller than 1
    ## matching_pairs = [subarray[0] for subarray in pair if subarray[0] == subarray[1]] 
    matching_pairs = [subarray[0] for subarray in pair if subarray[0] == subarray[1] and subarray[0] < num_points]
    TP = len(matching_pairs)
    ##  False negatives (FN)
    FN = num_points - len(matching_pairs)
    ##  False positives
    FP = len(pair) - len(matching_pairs)
    ##  True negatives
    TN = num_R_extra - (len(pair) - len(matching_pairs))
    print(TP, FN, FP, TN)
    recall  = TP/(TP+FN)
    print(f"Recall: {recall}")
    ## Precision = True positives/(True positives + False Positives)
    Precision = TP/(TP+FP)
    print(f"Precision: {Precision}")
    Accuracy = (TP+TN)/(TP+FP+TN+FN)
    print(f"Accuracy: {Accuracy}")
    print(f"True Pairs in whole region: {len(matching_pairs)}")
    return TP, FN, FP, TN, recall, Precision, Accuracy

########################################################################################################
#### Function to run the iterative monte carlo 
def run_background(num_pair_exp_mean, num_R, num_G, tag_dist, precision_value, d_true_thre, dis_tree_thre_factor,  area_value, number_iter, num_MC_points):
    ### Calculate the first pairing result
    num_R_initial = num_R
    num_G_initial = num_G
    MC_hist = []
    pair_est = []
    j = 0
    #### number_iter is the number of steps for MC to estimate the background to go
    while j < number_iter:
        j = j + 1
        RG_overlap_pair_simu_list = []
        ######## run 5 Monte Carlo trials to estimate the number of pairs from MC simulations
        iteration_sub = 5
        for k in range(iteration_sub):
            R_pos_simu, G_pos_simu, R_prec_list, G_prec_list = \
               Generate_Points(tag_dist, num_points = 0, num_R_extra = num_R, num_G_extra = num_G, precision=precision_value, area=area_value)
            RG_overlap_pair_simu = pair_matching_max_weight_nx(R_pos_simu, G_pos_simu, R_prec_list, G_prec_list, \
                d_true_thre = d_true_thre, dis_tree_thre_factor = dis_tree_thre_factor, num_MC_points = num_MC_points)
            RG_overlap_pair_simu_list.append(len(RG_overlap_pair_simu))
        ## Calculate the average of the overlap list from MC simulation given the num_R and num_G
        RG_overlap_pair_simu_ave = int(np.mean(RG_overlap_pair_simu_list))
        ## append the MC for RG overlap pair 
        MC_hist.append(RG_overlap_pair_simu_ave)
        pair_est.append(num_pair_exp_mean - RG_overlap_pair_simu_ave)
        ## Calculate the next iteration number 
        num_R = num_R_initial - (num_pair_exp_mean - RG_overlap_pair_simu_ave)
        num_G = num_G_initial - (num_pair_exp_mean - RG_overlap_pair_simu_ave)
    return MC_hist, pair_est

########################################################################################################

def run_background_two_channel(num_pair_exp_mean, num_R, num_G, tag_dist, prec1, prec2, d_true_thre, dis_tree_thre_factor,  area_value, number_iter, num_MC_points):
    ### Calculate the first pairing result
    num_R_initial = num_R
    num_G_initial = num_G
    MC_hist = []
    pair_est = []
    j = 0
    #### number_iter is the number of steps for MC to estimate the background to go
    while j < number_iter:
        j = j + 1
        RG_overlap_pair_simu_list = []
        ######## run 5 Monte Carlo trials to estimate the number of pairs from MC simulations
        iteration_sub = 5
        for k in range(iteration_sub):
            R_pos_simu, G_pos_simu, R_prec_list, G_prec_list = \
               Generate_Points_two_channel(tag_dist, num_points = 0, num_R_extra = num_R, num_G_extra = num_G, prec1=prec1, prec2=prec2, area=area_value)
            RG_overlap_pair_simu = pair_matching_max_weight_nx(R_pos_simu, G_pos_simu, R_prec_list, G_prec_list, \
                d_true_thre = d_true_thre, dis_tree_thre_factor = dis_tree_thre_factor, num_MC_points = num_MC_points)
            RG_overlap_pair_simu_list.append(len(RG_overlap_pair_simu))
        ## Calculate the average of the overlap list from MC simulation given the num_R and num_G
        RG_overlap_pair_simu_ave = int(np.mean(RG_overlap_pair_simu_list))
        ## append the MC for RG overlap pair 
        MC_hist.append(RG_overlap_pair_simu_ave)
        pair_est.append(num_pair_exp_mean - RG_overlap_pair_simu_ave)
        ## Calculate the next iteration number 
        num_R = num_R_initial - (num_pair_exp_mean - RG_overlap_pair_simu_ave)
        num_G = num_G_initial - (num_pair_exp_mean - RG_overlap_pair_simu_ave)
    return MC_hist, pair_est

########################################################################################################

def run_exp_iterative_MC(tag_dist, num_points, num_R_extra, num_G_extra, precision, d_true_thre, dis_tree_thre_factor, area, iteration_full, iteration_in_step, iteration_step, num_MC_points):
    ### Calculate the first pairing result
    num_pair_exp = []
    TP_list, FN_list, FP_list, TN_list, recall_list, Precision_list, Accuracy_list = [], [], [], [], [], [], []
    for iteration in range(iteration_full):
        print("Current iteration:", iteration)
        R_pos, G_pos, R_prec, G_prec = \
            Generate_Points(tag_dist, num_points, num_R_extra, num_G_extra, precision = precision, area = area)
        pair_exp = pair_matching_max_weight_nx(R_pos, G_pos, R_prec, G_prec, \
                d_true_thre = d_true_thre , dis_tree_thre_factor = dis_tree_thre_factor, num_MC_points = num_MC_points)
        num_pair_exp.append(len(pair_exp))
        summary_results = summary_pairs(pair_exp, num_points, num_R_extra, num_G_extra)
        TP_list.append(summary_results[0])
        FN_list.append(summary_results[1])
        FP_list.append(summary_results[2])
        TN_list.append(summary_results[3])
        recall_list.append(summary_results[4])
        Precision_list.append(summary_results[5])
        Accuracy_list.append(summary_results[6])
    mean_TP = np.mean(TP_list)
    mean_FN = np.mean(FN_list)
    mean_FP = np.mean(FP_list)
    mean_TN = np.mean(TN_list)
    mean_recall = np.mean(recall_list)
    mean_Precision = np.mean(Precision_list)
    mean_Accuracy = np.mean(Accuracy_list)
    ################################################################
    ##### Part II: MC to find the true number
    num_R = len(R_pos)
    num_G = len(G_pos)
    num_pair_exp_mean = round(np.mean(num_pair_exp))
    # Save the results to a text file
    with open(f"summary_results_Precision_{precision}.txt", "w") as file:
        file.write(f"Number of True Pairs: {num_points}\n")
        file.write(f"Number of R extra: {num_R_extra}\n")
        file.write(f"Number of G extra: {num_G_extra}\n")
        file.write(f"Area: {area}\n")
        file.write(f"Tag distance: {tag_dist}\n")
        file.write(f"True distance threshold: {d_true_thre*1000} nm\n")
        file.write(f"Distance Tree threshold Factor (Normalized): {dis_tree_thre_factor}\n")
        file.write(f"Number of MC in est prob: {num_MC_points} \n")
        file.write(f"Precision: {precision} um\n")
        file.write(f"Number of whole simulations: {iteration_full}\n")
        file.write("Metric\tMean Value\tList\n")
        file.write("Total Number of Pairs Found: \t{}\t{}\n".format(num_pair_exp_mean, num_pair_exp))
        file.write("TP\t{}\t{}\n".format(mean_TP, TP_list))
        file.write("FN\t{}\t{}\n".format(mean_FN, FN_list))
        file.write("FP\t{}\t{}\n".format(mean_FP, FP_list))
        file.write("TN\t{}\t{}\n".format(mean_TN, TN_list))
        file.write("Recall\t{}\t{}\n".format(mean_recall, recall_list))
        file.write("Precision_Stat\t{}\t{}\n".format(mean_Precision, Precision_list))
        file.write("Accuracy\t{}\t{}\n".format(mean_Accuracy, Accuracy_list))
        file.write(f"Number of steps in MC estimation of Background: {iteration_step}\n")
        file.write(f"Number of MC in each step for Background: {iteration_in_step}\n")
        MC_last_value_list = []
        pair_est_last_value_list = []
        for j in range(iteration_in_step):
            print(f"Interation: {j}")
            MC_hist, pair_est = run_background(num_pair_exp_mean, num_R, num_G, tag_dist, precision_value = precision, \
                    d_true_thre = d_true_thre, dis_tree_thre_factor = dis_tree_thre_factor, area_value = area,\
                 number_iter = iteration_step, num_MC_points = num_MC_points )
            MC_last_value_list.append(MC_hist[-1])
            pair_est_last_value_list.append(pair_est[-1])
            file.write(f"MC_hist{j}: {MC_hist}\n")
            file.write(f"pair_est{j}: {pair_est}\n")
        MC_last_mean = round(np.mean(MC_last_value_list))
        pair_est_last_mean = round(np.mean(pair_est_last_value_list))
        error_count = abs(num_points - np.array(pair_est_last_value_list))/num_points
        error_count_mean = np.mean(error_count)
        file.write(f"MC Last: {MC_last_value_list}\n")
        file.write(f"Pair_est_last: {pair_est_last_value_list}\n")
        file.write(f'Mean MC Last Value: {MC_last_mean}\n')
        file.write(f'Mean pair_est Last Value: {pair_est_last_mean}\n')
        file.write("Error of Count of pairs\t{}\t{}\n".format(error_count_mean, error_count))
    print("Summary results saved as summary_results.txt")
    return num_pair_exp, num_pair_exp_mean, MC_last_value_list, MC_last_mean, pair_est_last_value_list, pair_est_last_mean

########################################################################################################

def run_background_task(j, num_pair_exp_mean, num_R, num_G, tag_dist, prec1, prec2, d_true_thre, dis_tree_thre_factor, area_value, iteration_step, num_MC_points):
    # This function should handle a single iteration of the background computation
    print(f"Iteration: {j}")
    np.random.seed(j)
    MC_hist, pair_est = run_background_two_channel(num_pair_exp_mean, num_R, num_G, tag_dist, prec1=prec1, prec2=prec2, \
                                        d_true_thre=d_true_thre, dis_tree_thre_factor=dis_tree_thre_factor, area_value=area_value,\
                                        number_iter=iteration_step, num_MC_points=num_MC_points)
    return j, MC_hist, pair_est

########################################################################################################

def parallel_execution(iteration_in_step, num_pair_exp_mean, num_R, num_G, tag_dist, prec1, prec2, d_true_thre, dis_tree_thre_factor, area, iteration_step, num_MC_points):
    # Create a pool of workers
    with ProcessPoolExecutor(max_workers=None) as executor:
        # Submit tasks to the pool
        futures = {executor.submit(run_background_task, j, num_pair_exp_mean, num_R, num_G, tag_dist, prec1, prec2, d_true_thre, dis_tree_thre_factor, area, iteration_step, num_MC_points): j for j in range(iteration_in_step)}
        
        # Collect results
        MC_last_value_list = []
        pair_est_last_value_list = []
        final_results = []
        
        for future in as_completed(futures):
            j, MC_hist, pair_est = future.result()
            MC_last_value_list.append(MC_hist[-1])
            pair_est_last_value_list.append(pair_est[-1])
            final_results.append((j, MC_hist, pair_est))
    
    return final_results, MC_last_value_list, pair_est_last_value_list

########################################################################################################

def run_exp_iterative_MC_parallel(tag_dist, num_points, num_R_extra, num_G_extra, precision, d_true_thre, dis_tree_thre_factor, area, iteration_full, iteration_in_step, iteration_step, num_MC_points):
    # Existing code for Part I
    num_pair_exp = []
    TP_list, FN_list, FP_list, TN_list, recall_list, Precision_list, Accuracy_list = [], [], [], [], [], [], []
    
    for iteration in range(iteration_full):
        print("Current iteration:", iteration)
        R_pos, G_pos, R_prec, G_prec = Generate_Points(tag_dist, num_points, num_R_extra, num_G_extra, precision=precision, area=area)
        pair_exp = pair_matching_max_weight_nx(R_pos, G_pos, R_prec, G_prec, d_true_thre=d_true_thre, dis_tree_thre_factor=dis_tree_thre_factor, num_MC_points=num_MC_points)
        num_pair_exp.append(len(pair_exp))
        summary_results = summary_pairs(pair_exp, num_points, num_R_extra, num_G_extra)
        TP_list.append(summary_results[0])
        FN_list.append(summary_results[1])
        FP_list.append(summary_results[2])
        TN_list.append(summary_results[3])
        recall_list.append(summary_results[4])
        Precision_list.append(summary_results[5])
        Accuracy_list.append(summary_results[6])
    
    mean_TP = np.mean(TP_list)
    mean_FN = np.mean(FN_list)
    mean_FP = np.mean(FP_list)
    mean_TN = np.mean(TN_list)
    mean_recall = np.mean(recall_list)
    mean_Precision = np.mean(Precision_list)
    mean_Accuracy = np.mean(Accuracy_list)
    
    # Prepare to run the background computations in parallel
    num_R = len(R_pos)
    num_G = len(G_pos)
    density_R = num_R/area
    density_G = num_G/area
    num_pair_exp_mean = round(np.mean(num_pair_exp))
    
    MC_last_value_list = []
    pair_est_last_value_list = []
    results = []

    results, MC_last_value_list, pair_est_last_value_list = parallel_execution(
        iteration_in_step, num_pair_exp_mean, num_R, num_G, tag_dist, precision, d_true_thre, dis_tree_thre_factor, area, iteration_step, num_MC_points
    )
    
    # Write results to file sequentially after all parallel tasks have completed
    with open(f"Parallel_results_Density_{density_R}.txt", "w") as file:
        file.write(f"Number of True Pairs: {num_points}\n")
        file.write(f"Number of R extra: {num_R_extra}\n")
        file.write(f"Number of G extra: {num_G_extra}\n")
        file.write(f"Area: {area}\n")
        file.write(f"Density R: {density_R}\n")
        file.write(f"Density G: {density_G}\n")
        file.write(f"Tag distance: {tag_dist}\n")
        file.write(f"True distance threshold: {d_true_thre*1000} nm\n")
        file.write(f"Distance Tree threshold Factor (Normalized): {dis_tree_thre_factor}\n")
        file.write(f"Number of MC in est prob: {num_MC_points}\n")
        file.write(f"Precision: {precision} um\n")
        file.write(f"Number of whole simulations: {iteration_full}\n")
        file.write("Metric\tMean Value\tList\n")
        file.write("Total Number of Pairs Found: \t{}\t{}\n".format(num_pair_exp_mean, num_pair_exp))
        file.write("TP\t{}\t{}\n".format(mean_TP, TP_list))
        file.write("FN\t{}\t{}\n".format(mean_FN, FN_list))
        file.write("FP\t{}\t{}\n".format(mean_FP, FP_list))
        file.write("TN\t{}\t{}\n".format(mean_TN, TN_list))
        file.write("Recall\t{}\t{}\n".format(mean_recall, recall_list))
        file.write("Precision_Stat\t{}\t{}\n".format(mean_Precision, Precision_list))
        file.write("Accuracy\t{}\t{}\n".format(mean_Accuracy, Accuracy_list))
        file.write(f"Number of steps in MC estimation of Background: {iteration_step}\n")
        file.write(f"Number of MC in each step for Background: {iteration_in_step}\n")
        
        for j, MC_hist, pair_est in results:
            file.write(f"MC_hist{j}: {MC_hist}\n")
            file.write(f"pair_est{j}: {pair_est}\n")
        
        MC_last_mean = round(np.mean(MC_last_value_list))
        pair_est_last_mean = round(np.mean(pair_est_last_value_list))
        error_count = abs(num_points - np.array(pair_est_last_value_list)) / num_points
        error_count_mean = np.mean(error_count)
        file.write(f"MC Last: {MC_last_value_list}\n")
        file.write(f"Pair_est_last: {pair_est_last_value_list}\n")
        file.write(f'Mean MC Last Value: {MC_last_mean}\n')
        file.write(f'Mean pair_est Last Value: {pair_est_last_mean}\n')
        file.write("Error of Count of pairs\t{}\t{}\n".format(error_count_mean, error_count))
    
    print("Summary results saved as summary_results.txt")
    return num_pair_exp, num_pair_exp_mean, MC_last_value_list, MC_last_mean, pair_est_last_value_list, pair_est_last_mean

########################################################################################################

def run_exp_iterative_MC_parallel_two_channel(tag_dist, num_points, num_R_extra, num_G_extra, prec1, prec2, d_true_thre, dis_tree_thre_factor, area, iteration_full, iteration_in_step, iteration_step, num_MC_points):
    """ Main function for 

    Parameters
    ----------
    tag_dist: Distance betweeen two proteins R and G
    num_points: total number of pairs RG
    num_R_extra: number of background R
    num_G_extra: number of background G
    prec1: localization precision array of R
    prec2: localization precision array of G
    d_true_thre: the upper bound for d_true
    dis_tree_thre_factor: the factor for distance tree threshold 
    area: area 
    iteration_full: number of full iterations 
    iteration_in_step: 
    iteration_step: 
    num_MC_points: number of Monte Carlo points 
    """
    ## 
    num_pair_exp = []
    ## True positive, False negative, True negative, recall, precision, accuracy
    TP_list, FN_list, FP_list, TN_list, recall_list, Precision_list, Accuracy_list = [], [], [], [], [], [], []
    
    for iteration in range(iteration_full):
        print("Current iteration:", iteration)
        # Generate simulation points, positions of R, G, precisions of R and G
        R_pos, G_pos, R_prec, G_prec = Generate_Points_two_channel(tag_dist, num_points, num_R_extra, num_G_extra, prec1, prec2, area=area)
        # Build the maximum weighted bipartitle graph, return the matching
        pair_exp = pair_matching_max_weight_nx(R_pos, G_pos, R_prec, G_prec, d_true_thre=d_true_thre, dis_tree_thre_factor=dis_tree_thre_factor, num_MC_points=num_MC_points)
        num_pair_exp.append(len(pair_exp))
        # Get the summary statistics for the results 
        summary_results = summary_pairs(pair_exp, num_points, num_R_extra, num_G_extra)
        TP_list.append(summary_results[0])
        FN_list.append(summary_results[1])
        FP_list.append(summary_results[2])
        TN_list.append(summary_results[3])
        recall_list.append(summary_results[4])
        Precision_list.append(summary_results[5])
        Accuracy_list.append(summary_results[6])
    # mean TP, FN, FP, TN, recall, precision and accuracy
    mean_TP = np.mean(TP_list)
    mean_FN = np.mean(FN_list)
    mean_FP = np.mean(FP_list)
    mean_TN = np.mean(TN_list)
    mean_recall = np.mean(recall_list)
    mean_Precision = np.mean(Precision_list)
    mean_Accuracy = np.mean(Accuracy_list)
    
    # Prepare to run the number background estimation in parallel
    num_R = len(R_pos)
    num_G = len(G_pos)
    density_R = num_R/area
    density_G = num_G/area
    num_pair_exp_mean = round(np.mean(num_pair_exp))
    
    MC_last_value_list = []
    pair_est_last_value_list = []
    results = []

    results, MC_last_value_list, pair_est_last_value_list = parallel_execution(
        iteration_in_step, 
        num_pair_exp_mean, 
        num_R, 
        num_G, 
        tag_dist, 
        prec1, 
        prec2, 
        d_true_thre, 
        dis_tree_thre_factor, 
        area, 
        iteration_step, 
        num_MC_points
    )
    
    # Write results to file sequentially after all parallel tasks have completed
    with open(f"Parallel_results_Density_{density_R}.txt", "w") as file:
        file.write(f"Number of True Pairs: {num_points}\n")
        file.write(f"Number of R extra: {num_R_extra}\n")
        file.write(f"Number of G extra: {num_G_extra}\n")
        file.write(f"Area: {area}\n")
        file.write(f"Density R: {density_R}\n")
        file.write(f"Density G: {density_G}\n")
        file.write(f"Tag distance: {tag_dist}\n")
        file.write(f"True distance threshold: {d_true_thre*1000} nm\n")
        file.write(f"Distance Tree threshold Factor (Normalized): {dis_tree_thre_factor}\n")
        file.write(f"Number of MC in est prob: {num_MC_points}\n")
        file.write(f"Precision: Different for Channel 1 and 2\n")
        file.write(f"Number of whole simulations: {iteration_full}\n")
        file.write("Metric\tMean Value\tList\n")
        file.write("Total Number of Pairs Found: \t{}\t{}\n".format(num_pair_exp_mean, num_pair_exp))
        file.write("TP\t{}\t{}\n".format(mean_TP, TP_list))
        file.write("FN\t{}\t{}\n".format(mean_FN, FN_list))
        file.write("FP\t{}\t{}\n".format(mean_FP, FP_list))
        file.write("TN\t{}\t{}\n".format(mean_TN, TN_list))
        file.write("Recall\t{}\t{}\n".format(mean_recall, recall_list))
        file.write("Precision_Stat\t{}\t{}\n".format(mean_Precision, Precision_list))
        file.write("Accuracy\t{}\t{}\n".format(mean_Accuracy, Accuracy_list))
        file.write(f"Number of steps in MC estimation of Background: {iteration_step}\n")
        file.write(f"Number of MC in each step for Background: {iteration_in_step}\n")
        
        for j, MC_hist, pair_est in results:
            file.write(f"MC_hist{j}: {MC_hist}\n")
            file.write(f"pair_est{j}: {pair_est}\n")
        
        MC_last_mean = round(np.mean(MC_last_value_list))
        pair_est_last_mean = round(np.mean(pair_est_last_value_list))
        error_count = abs(num_points - np.array(pair_est_last_value_list)) / num_points
        error_count_mean = np.mean(error_count)
        file.write(f"MC Last: {MC_last_value_list}\n")
        file.write(f"Pair_est_last: {pair_est_last_value_list}\n")
        file.write(f'Mean MC Last Value: {MC_last_mean}\n')
        file.write(f'Mean pair_est Last Value: {pair_est_last_mean}\n')
        file.write("Error of Count of pairs\t{}\t{}\n".format(error_count_mean, error_count))
    
    print("Summary results saved as summary_results.txt")
    return num_pair_exp, num_pair_exp_mean, MC_last_value_list, MC_last_mean, pair_est_last_value_list, pair_est_last_mean
