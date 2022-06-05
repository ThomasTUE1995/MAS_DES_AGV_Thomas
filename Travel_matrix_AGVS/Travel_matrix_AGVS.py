"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import itertools
import random
from collections import defaultdict


import numpy as np
import pandas as pd
import numpy.random







def choose_distance_matrix(agvsPerWC_input, machinesPerWC, seed):
    load_time = 0.25
    travel_time_matrix_static = create_distance_matrix(machinesPerWC, load_time, seed)
    agvsPerWC = agvsPerWC_input  # Number of AGVs per workcenter
    agv_number_WC = []
    agv_count = 1
    for agv_WC in agvsPerWC:
        WC_AGV_list = []
        for No_agv_WC in range(agv_WC):
            WC_AGV_list.append(No_agv_WC + agv_count)
        agv_count += agv_WC
        agv_number_WC.append(WC_AGV_list)



    return travel_time_matrix_static, agvsPerWC, agv_number_WC


def create_distance_matrix(machinesPerWC, load_time, seed):
    """Creates distance matrix where distance can be requested by inputting:
    distance_maxtrix[actual location][destination]"""

    numpy.random.seed(seed)


    noOfWC = range(len(machinesPerWC))

    # All distances are in meters
    distance_matrix = {
        "depot": {(ii, jj): np.random.uniform(0.5, 1) + load_time for jj in noOfWC for ii in range(machinesPerWC[jj])}}

    distance_matrix["depot"].update({"depot": 0})

    for jj in noOfWC:
        for ii in range(machinesPerWC[jj]):
            distance_matrix[(ii, jj)] = {(ii, jj): np.random.uniform(0.5, 1) + load_time for jj in noOfWC for ii in
                                         range(machinesPerWC[jj])}

            distance_matrix[(ii, jj)].update({"depot": np.random.uniform(0.5, 1) + load_time})

            distance_matrix[ii, jj][ii, jj] = 0


    return distance_matrix





