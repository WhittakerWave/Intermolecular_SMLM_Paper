
#####################################################################################################
##### Support Function for colocalization program
from package_func import *

#####################################################################################################
def Random_Point(num_R, num_G, scale):
    """ Function to generate random points with scale 

    Parameters
    ----------
    num_R: number of R 
    num_G: number of G
    scale: scale of the domain
    """
    ## Generate Random points in the domian 
    ## Here we haven't used precision, maybe use later, the precision of each points will be added later
    R_pos = np.random.random((num_R, 2))*scale
    G_pos = np.random.random((num_G, 2))*scale
    return R_pos, G_pos

#####################################################################################################
def homo_Possion_Process1(num_R, num_G, R_precision, G_precision, area, frame_len):
    """ Function to generate homogeneous Possion process

    Parameters
    ----------
    num_R: number of R 
    num_G: number of G
    R_precision: localization precisions of R
    G_precision: localization precisions of G
    area: area 
    frame_len: length of frame 
    """
    ## Homogeneous Possion process
    grid = [[0,np.sqrt(area)], [0,np.sqrt(area)]]
    # Simulation window parameters
    xMin = 0
    xMax = grid[0][1]
    yMin = 0
    yMax = grid[0][1]
    xDelta = xMax - xMin
    yDelta = yMax - yMin
    areaTotal = xDelta*yDelta
    # Point process parameters
    lambda_R = num_R / areaTotal #intensity (ie mean density) of the Poisson process
    lambda_G = num_G / areaTotal
    # Simulate Poisson point process
    pointsNumber_R = scipy.stats.poisson(lambda_R*areaTotal).rvs()           #Poisson number of points
    pointsNumber_G = scipy.stats.poisson(lambda_G*areaTotal).rvs()           #Poisson number of points
    xx_R = xDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber_R,1))) + xMin #x coordinates of Poisson points
    yy_R = yDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber_R,1))) + yMin #y coordinates of Poisson points
    points_R = np.concatenate([xx_R, yy_R], axis=1)
    xx_G = xDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber_G,1))) + xMin #x coordinates of Poisson points
    yy_G = yDelta * scipy.stats.uniform.rvs(0,1,((pointsNumber_G,1))) + yMin #y coordinates of Poisson points
    points_G = np.concatenate([xx_G, yy_G], axis=1)
    if len(points_R) <= len(R_precision):
        R_prec_list = random.sample(list(R_precision), len(points_R))
    else:
        R_prec_list1 = random.sample(list(R_precision), len(R_precision))
        R_prec_arr = np.array(list(R_precision))
        probs = R_prec_arr / np.sum(R_prec_arr)  # Normalize probabilities
        R_prec_list2 = np.random.choice(R_prec_arr, size=len(points_R)-len(R_precision), p=probs, replace=True).tolist()
        R_prec_list = R_prec_list1 + R_prec_list2
    if len(points_G) <= len(G_precision):
        G_prec_list = random.sample(list(G_precision), len(points_G))
    else:
        G_prec_list1 = random.sample(list(G_precision), len(G_precision))
        G_prec_arr = np.array(list(G_precision))
        probs = G_prec_arr / np.sum(G_prec_arr)  # Normalize probabilities
        G_prec_list2 = np.random.choice(G_prec_arr, size = len(points_G)-len(G_precision), p=probs, replace=True).tolist()
        G_prec_list = G_prec_list1 + G_prec_list2

    # Generate two uniform distributions
    uniform_dist1 = np.arange(0, frame_len +1)
    uniform_dist2 = np.arange(0, frame_len +1)
    R_frame = []
    G_frame = []
    for i in range(len(points_R)):
        R_frame.append(np.random.choice(uniform_dist1, size=1)[0].astype(float))
    for j in range(len(points_G)):
        G_frame.append(np.random.choice(uniform_dist2, size=1)[0].astype(float))
    
    ### Need to pertube the points with Precision 
    R_prec_array = np.array(R_prec_list).reshape(len(points_R), 1)/1000
    G_prec_array = np.array(G_prec_list).reshape(len(points_G), 1)/1000
    gaussian_array_R = np.random.normal(loc = np.zeros((len(points_R), 2)), \
            scale = R_prec_array, size=(len(points_R), 2))
    gaussian_array_G = np.random.normal(loc = np.zeros((len(points_G), 2)), \
            scale = G_prec_array, size=(len(points_G), 2))
    # add the Gaussian array to R_pos
    points_R = points_R + gaussian_array_R
    points_G = points_G + gaussian_array_G
    return points_R, points_G, R_prec_list, G_prec_list, R_frame, G_frame

#####################################################################################################
def frame_diff(frame_pairs, title, cell):
    """ Function to calculate the frame difference

    Parameters
    ----------
    frame_pairs: the frame of pairs
    title: input title
    cell: name of the cell 
    """
    ## Calculate the frame difference between the pairs
    frame_diff = []
    for item in frame_pairs:
        frame_diff.append(abs(item[0] - item[1]))
    fig, ax = plt.subplots()
    sns.histplot(frame_diff)
    # add title and axis labels
    if title == False:
        ax.set_title('Monte Carlo Simulation Frame Difference', fontsize = 20)
    else:
        ax.set_title(f'Frame Difference of {cell}', fontsize = 20)
    ax.set_xlabel('Difference of frames in a pair (abs)')
    ax.set_ylabel('Count')
    plt.show()

#############################################################################################################################
def get_vertices(tag_dist, point):
    # Define a function to calculate the coordinates of the R and G vertices
    angle = random.uniform(0, 2*math.pi)
    x, y = point
    R = (x + tag_dist/2*math.cos(angle), y + tag_dist/2*math.sin(angle))
    G = (x - tag_dist/2*math.cos(angle), y - tag_dist/2*math.sin(angle))
    return R, G

#############################################################################################################################
def Generate_Points(tag_dist, num_points, num_R_extra, num_G_extra, precision, area):
    """ Function to generate the points 

    Parameters
    ----------
    tag_dist: Distance betweeen two proteins R and G
    num_points: number of RG 
    num_R_extra: number of background R 
    num_G_extra: number of background G
    precision: localization precisions
    area: area
    """
    x_length, y_length = np.sqrt(area), np.sqrt(area)
    common_points = [(random.uniform(0, x_length), random.uniform(0, y_length)) for i in range(num_points)]
    # Create a list of tuples, where each tuple contains the point, R vertex, G vertex, and a random tag
    ### tag dist to be uniform randomly sampled
    common_data = [(*get_vertices(random.uniform(0, tag_dist), point), point) for point in common_points]
    # Extract R and G vertices into separate lists
    R_pos = np.array([item[0] for item in common_data])
    G_pos = np.array([item[1] for item in common_data])
    R_pos_extra, G_pos_extra = Random_Point(num_R_extra, num_G_extra, scale = x_length)
    ## Add the R_pos and extra background points together
    R_pos = R_pos_extra if len(R_pos) == 0 else np.concatenate((R_pos, R_pos_extra))
    G_pos = G_pos_extra if len(G_pos) == 0 else np.concatenate((G_pos, G_pos_extra))
    ## Add noise of from constant precision, can also use a distribution for the precision 
    R_prec = np.ones((num_points + num_R_extra, 2))*precision
    G_prec = np.ones((num_points + num_G_extra, 2))*precision
    ## Add Gaussian Noise 
    gaussian_array_R = np.random.normal(loc = np.zeros((num_points + num_R_extra, 2)), scale=R_prec, size=(num_points + num_R_extra, 2))
    gaussian_array_G = np.random.normal(loc = np.zeros((num_points + num_G_extra, 2)), scale=G_prec, size=(num_points + num_G_extra, 2))
    #sns.histplot(gaussian_array_R[:,0])
    #sns.histplot(gaussian_array_R[:,1])
    # Add the Gaussian noise array to R_pos and G_pos
    R_pos = R_pos + gaussian_array_R
    G_pos = G_pos + gaussian_array_G
    return R_pos, G_pos, R_prec[:,0], G_prec[:,0]

#############################################################################################################################
def Generate_Points_two_channel(tag_dist, num_points, num_R_extra, num_G_extra, prec1, prec2, area):
    """ Function to generate the points 

    Parameters
    ----------
    tag_dist: Distance betweeen two proteins R and G
    num_points: number of RG 
    num_R_extra: number of background R 
    num_G_extra: number of background G
    prec1: localization precision for R
    prec2: localization precision for G
    area: area
    """
    x_length, y_length = np.sqrt(area), np.sqrt(area)
    common_points = [(random.uniform(0, x_length), random.uniform(0, y_length)) for i in range(num_points)]
    # Create a list of tuples, where each tuple contains the point, R vertex, G vertex, and a random tag
    ### tag dist to be uniform randomly sampled
    common_data = [(*get_vertices(random.uniform(0, tag_dist), point), point) for point in common_points]
    # Extract R and G vertices into separate lists
    R_pos = np.array([item[0] for item in common_data])
    G_pos = np.array([item[1] for item in common_data])
    R_pos_extra, G_pos_extra = Random_Point(num_R_extra, num_G_extra, scale = x_length)
    ## Add the R_pos and extra background points together
    R_pos = R_pos_extra if len(R_pos) == 0 else np.concatenate((R_pos, R_pos_extra))
    G_pos = G_pos_extra if len(G_pos) == 0 else np.concatenate((G_pos, G_pos_extra))
    ## Add noise of from constant precision, can also use a distribution for the precision 
    R_prec = np.random.choice(prec1, num_points + num_R_extra)
    R_prec_2d = np.column_stack((R_prec, R_prec))
    G_prec = np.random.choice(prec2, num_points + num_G_extra)
    G_prec_2d = np.column_stack((G_prec, G_prec))
    ## Add Gaussian Noise 
    gaussian_array_R = np.random.normal(loc = np.zeros((num_points + num_R_extra, 2)), scale=R_prec_2d, size=(num_points + num_R_extra, 2))
    gaussian_array_G = np.random.normal(loc = np.zeros((num_points + num_G_extra, 2)), scale=G_prec_2d, size=(num_points + num_G_extra, 2))
    #sns.histplot(gaussian_array_R[:,0])
    #sns.histplot(gaussian_array_R[:,1])
    # Add the Gaussian noise array to R_pos and G_pos
    R_pos = R_pos + gaussian_array_R
    G_pos = G_pos + gaussian_array_G
    return R_pos, G_pos, R_prec_2d[:,0], G_prec_2d[:,0]

#####################################################################################################
def random_select(R_pos, G_pos, R_prec, G_prec, sample_size):
    """ Function to randomly select the points in the list

    Parameters
    ----------
    R_pos: positions of R
    G_pos: positions of G
    R_prec: localization precision of R
    G_prec: localization precision of G
    sample_size: sample size
    """
    random_indices_R = np.random.choice(R_pos.shape[0], size = sample_size, replace=False)
    R_pos = R_pos[random_indices_R]
    R_prec = R_prec[random_indices_R]
    random_indices_G = np.random.choice(G_pos.shape[0], size = sample_size, replace=False)
    G_pos = G_pos[random_indices_G]
    G_prec = G_prec[random_indices_G]
    return R_pos, G_pos, R_prec, G_prec, random_indices_R, random_indices_G

#####################################################################################################
def plot_coloca(pair_index, R_pos, G_pos, filtered_R_Prec, filtered_G_Prec, name):
    """ Function to plot colocalization and non-colocalization

    Parameters
    ----------
    pair_index: index of pairs
    R_pos: positions of R
    G_pos: positions of G
    filtered_R_Prec: localization precisions of R
    filtered_G_Prec: localization precisions of G
    name: title
    """
    index_R_C = np.array(pair_index).T[0]
    index_G_C = np.array(pair_index).T[1]
    
    total_index_R = list(range(0, len(R_pos)))
    total_index_G = list(range(0, len(G_pos)))
    
    index_R_NoC = [x for x in total_index_R if x not in index_R_C]
    index_G_NoC = [x for x in total_index_G if x not in index_G_C]

    R_pos_C = R_pos[index_R_C]
    R_pos_NoC = R_pos[index_R_NoC]
    R_prec_C = filtered_R_Prec[index_R_C]/1000
    R_prec_NoC = filtered_R_Prec[index_R_NoC]/1000
    
    G_pos_C = G_pos[index_G_C]
    G_pos_NoC = G_pos[index_G_NoC]
    G_prec_C = filtered_G_Prec[index_G_C]/1000
    G_prec_NoC = filtered_G_Prec[index_G_NoC]/1000

    fig, ax = plt.subplots()
    color = ['tab:green', 'tab:purple', 'tab:blue', 'tab:blue']
    b2AR_color = tuple(c/255 for c in [0, 176, 80])
    Gs_color = tuple(c/255 for c in [255, 0, 255])
    complex_color = tuple(c/255 for c in [63, 83, 135])
    
    circles = []
    legend_labels = []
    for i in range(len(R_prec_NoC)):
        circle = plt.Circle((R_pos_NoC[i, 0], R_pos_NoC[i, 1]), R_prec_NoC[i], color=b2AR_color)
        ax.add_artist(circle)
        circles.append(circle)
        legend_labels.append('HALO No Colocalization')

    for i in range(len(G_prec_NoC)):
        circle = plt.Circle((G_pos_NoC[i, 0], G_pos_NoC[i, 1]), G_prec_NoC[i], color=Gs_color)
        ax.add_artist(circle)
        circles.append(circle)
        legend_labels.append('SNAP No Colocalization')

    for i in range(len(R_pos_C)):
        circle = plt.Circle((R_pos_C[i, 0], R_pos_C[i, 1]), R_prec_C[i], color=complex_color)
        ax.add_artist(circle)
        circles.append(circle)
        legend_labels.append('HALO-SNAP Colocalization')

    for i in range(len(G_pos_C)):
        circle = plt.Circle((G_pos_C[i, 0], G_pos_C[i, 1]), G_prec_C[i], color=complex_color)
        ax.add_artist(circle)
        circles.append(circle)
        legend_labels.append('HALO-SNAP Colocalization')

    unique_labels = {}
    for i in range(len(circles)):
        if legend_labels[i] not in unique_labels:
            unique_labels[legend_labels[i]] = circles[i]
    ax.legend(unique_labels.values(), unique_labels.keys())
    
    ## 
    ax.plot([R_pos_C[:,0],G_pos_C[:,0]], [R_pos_C[:,1],G_pos_C[:,1]], color=color[3])
    ax.set_xlabel(r"x/$\mu m$")
    ax.set_ylabel(r"y/$\mu m$")
    if name is not None:
        ax.set_title(name, fontsize = 20)
        
    plt.show()

