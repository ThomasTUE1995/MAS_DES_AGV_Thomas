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



machinesPerWC = [3, 4, 3, 3, 4]  # Number of machines per workcenter
machine_number_WC = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16, 17]]  # Index of machines
noOfWC = range(len(machinesPerWC))
load_time = 0.25



"""machinesPerWC = [3, 2, 2, 4, 4, 2, 4, 5]  # Number of machines per workcenter
machine_number_WC = [[1, 2, 3], [4, 5], [6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17], [18, 19, 20, 21], [22, 23, 24, 25, 26]]  # Index of machines
noOfWC = range(len(machinesPerWC))
load_time = 0.25"""





def choose_distance_matrix(agvsPerWC_input, machinesPerWC):
    load_time = 0.25
    travel_time_matrix_static = create_distance_matrix(machinesPerWC, load_time)
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


def create_distance_matrix(machinesPerWC, load_time):
    """Creates distance matrix where distance can be requested by inputting:
    distance_maxtrix[actual location][destination]"""

    noOfWC = range(len(machinesPerWC))

    # All distances are in meters
    distance_matrix = {
        "depot": {(ii, jj): random.uniform(0.5, 1) + load_time for jj in noOfWC for ii in range(machinesPerWC[jj])}}

    distance_matrix["depot"].update({"depot": 0})

    for jj in noOfWC:
        for ii in range(machinesPerWC[jj]):
            distance_matrix[(ii, jj)] = {(ii, jj): random.uniform(0.5, 1) + load_time for jj in noOfWC for ii in
                                         range(machinesPerWC[jj])}

            distance_matrix[(ii, jj)].update({"depot": random.uniform(0.5, 1) + load_time})

            distance_matrix[ii, jj][ii, jj] = 0


    return distance_matrix









"""# 6 AGV per WC - ZERO TRAVEL TIME!
if AGV_selection == 0:
    travel_time_matrix_static = {
        'depot': {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                  (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                  (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (0, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (1, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (2, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (3, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (0, 1): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (1, 1): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (0, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (1, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (2, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (3, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (4, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (0, 3): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (1, 3): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (2, 3): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (0, 4): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (1, 4): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, (4, 0): 0, (2, 1): 0,
                 (5, 2): 0, (3, 3): 0, (2, 4): 0, 'depot': 0},
        (4, 0): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0,
                 (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0},
        (2, 1): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0,
                 (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0},
        (5, 2): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0,
                 (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0},
        (3, 3): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0,
                 (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0},
        (2, 4): {(0, 0): 0, (1, 0): 0, (2, 0): 0, (3, 0): 0, (0, 1): 0, (1, 1): 0, (0, 2): 0, (1, 2): 0, (2, 2): 0,
                 (3, 2): 0, (4, 2): 0, (0, 3): 0, (1, 3): 0, (2, 3): 0, (0, 4): 0, (1, 4): 0, 'depot': 0, (4, 0): 0,
                 (2, 1): 0, (5, 2): 0, (3, 3): 0, (2, 4): 0}}
    agvsPerWC = [6, 6, 6, 6, 6]  # Number of AGVs per workcenter
    agv_number_WC = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24],
                     [25, 26, 27, 28, 29, 30]]  # Index of AGVs

# Scenario 1
if AGV_selection == 1:
    travel_time_matrix_static = {'depot': {(0, 0): 1.1850234015136274, (1, 0): 0.9995834607312896, (2, 0): 1.0949007486965356, (0, 1): 0.8843732965791278, (1, 1): 0.8539190334715572, (2, 1): 1.242379074210409, (3, 1): 0.9185849117011334, (0, 2): 1.1494057327566662, (1, 2): 0.7919570914596376, (2, 2): 1.0090484885246302, (0, 3): 0.9168293029270931, (1, 3): 0.7675036020174357, (2, 3): 0.7518318846845684, (0, 4): 1.047196895407294, (1, 4): 0.7930330628668834, (2, 4): 1.2328246257806081, (3, 4): 1.075759274626027, 'depot': 0}, (0, 0): {(0, 0): 0, (1, 0): 0.9802893832388808, (2, 0): 0.9656886881558182, (0, 1): 0.9270359830294888, (1, 1): 1.1160223728428238, (2, 1): 0.9850207587592653, (3, 1): 1.2237437261873785, (0, 2): 1.1964872504409623, (1, 2): 0.7697750651768867, (2, 2): 0.9755619457413611, (0, 3): 1.2417072101656852, (1, 3): 1.051931694586893, (2, 3): 1.062262001737222, (0, 4): 0.9848648420491911, (1, 4): 0.9993621763854403, (2, 4): 1.1450163859429092, (3, 4): 1.1717324715018658, 'depot': 1.0344622600883242}, (1, 0): {(0, 0): 0.793224895033207, (1, 0): 0, (2, 0): 0.9459764102325998, (0, 1): 1.0672715599374452, (1, 1): 0.81536676358039, (2, 1): 0.8454424244298129, (3, 1): 1.1898404779361418, (0, 2): 0.8055952588982191, (1, 2): 1.1607218483814525, (2, 2): 0.7651234633518602, (0, 3): 0.8713684992962661, (1, 3): 0.9444606998925665, (2, 3): 1.0658557849999288, (0, 4): 0.8221619461616534, (1, 4): 1.22886211199293, (2, 4): 1.2282807402485603, (3, 4): 0.8969659019998122, 'depot': 0.7559561644959523}, (2, 0): {(0, 0): 1.0011527667649234, (1, 0): 1.0287412201073745, (2, 0): 0, (0, 1): 1.1167002022245305, (1, 1): 1.0769965779923907, (2, 1): 1.2081810667043227, (3, 1): 1.2235768481190616, (0, 2): 1.1591420147603724, (1, 2): 1.1191850909189536, (2, 2): 0.8408911590529923, (0, 3): 0.8135088655768384, (1, 3): 0.8938860953432759, (2, 3): 0.8991624090647101, (0, 4): 1.0404964996136299, (1, 4): 0.8949032494909083, (2, 4): 1.2256715641874278, (3, 4): 1.1012660693054785, 'depot': 0.7737614406444242}, (0, 1): {(0, 0): 1.219365190813623, (1, 0): 1.0573130907469421, (2, 0): 0.7718534616948629, (0, 1): 0, (1, 1): 1.0290607468196793, (2, 1): 1.2473073991660377, (3, 1): 1.2139262102514163, (0, 2): 0.8459332749664559, (1, 2): 0.9754363671017343, (2, 2): 0.9752848843216255, (0, 3): 1.1772058583477232, (1, 3): 1.048421007188227, (2, 3): 0.8809397779505554, (0, 4): 0.8485326240113681, (1, 4): 1.1642025319060212, (2, 4): 0.8633903897451719, (3, 4): 0.9890530073549089, 'depot': 1.0080278679258678}, (1, 1): {(0, 0): 0.9749077800850827, (1, 0): 0.8739219495590687, (2, 0): 1.0984602251087088, (0, 1): 0.8345119054568794, (1, 1): 0, (2, 1): 0.800797036161372, (3, 1): 0.9863606566639365, (0, 2): 0.7676188080038014, (1, 2): 0.9128510836344164, (2, 2): 1.155467080020808, (0, 3): 0.8858748596974231, (1, 3): 0.8588540128343644, (2, 3): 1.1329749262428965, (0, 4): 1.141686782922903, (1, 4): 0.857605738510477, (2, 4): 1.0543920040482662, (3, 4): 0.7637869369197623, 'depot': 1.1230111694875506}, (2, 1): {(0, 0): 1.0903353018813942, (1, 0): 1.1072983540187793, (2, 0): 0.9717163995769693, (0, 1): 1.248527841265338, (1, 1): 0.9787987391514079, (2, 1): 0, (3, 1): 1.1805392728950486, (0, 2): 1.1854534397849465, (1, 2): 1.2024439482724376, (2, 2): 0.7850511447780969, (0, 3): 0.9637193282528819, (1, 3): 0.8101099940936253, (2, 3): 1.241559653480974, (0, 4): 1.2222013291835254, (1, 4): 0.7511108372537298, (2, 4): 0.9074373468416166, (3, 4): 0.9397856409418426, 'depot': 1.0678035576650697}, (3, 1): {(0, 0): 1.0153959012922542, (1, 0): 1.0145493763904285, (2, 0): 1.1208485647728452, (0, 1): 1.1656099696712996, (1, 1): 1.02661146867867, (2, 1): 0.8449577117404743, (3, 1): 0, (0, 2): 1.0129136515969124, (1, 2): 1.073774989190725, (2, 2): 0.8781449751443255, (0, 3): 0.8907798271022225, (1, 3): 1.2212032785792584, (2, 3): 1.1476666640097917, (0, 4): 0.9362037902458258, (1, 4): 0.924451944934052, (2, 4): 0.8580864670726401, (3, 4): 1.1825122391151814, 'depot': 0.7983478396655936}, (0, 2): {(0, 0): 1.0378899568362372, (1, 0): 0.8002361793433563, (2, 0): 0.877600862714881, (0, 1): 1.1271555075799384, (1, 1): 0.9506982971611233, (2, 1): 1.0218438100686025, (3, 1): 1.0183118033473348, (0, 2): 0, (1, 2): 1.1831487231462592, (2, 2): 0.7688831809059125, (0, 3): 0.9798229582741413, (1, 3): 0.9643153746722652, (2, 3): 0.7764888102853309, (0, 4): 1.167231084035621, (1, 4): 0.9675078220952789, (2, 4): 0.9528882394230844, (3, 4): 1.1521047134377724, 'depot': 0.9782243190330289}, (1, 2): {(0, 0): 1.1168522789092337, (1, 0): 0.9855134285346148, (2, 0): 0.7877044551460319, (0, 1): 1.2488060277956987, (1, 1): 0.9238798651207654, (2, 1): 0.7527768051594508, (3, 1): 0.9764201655692799, (0, 2): 0.9578637483351495, (1, 2): 0, (2, 2): 1.2010689374489951, (0, 3): 1.1122231682495582, (1, 3): 1.1510181389352248, (2, 3): 1.2339195078179688, (0, 4): 0.9480193691403289, (1, 4): 0.90179507411312, (2, 4): 1.212266851063247, (3, 4): 1.0684595874114717, 'depot': 0.7650620308748775}, (2, 2): {(0, 0): 0.7939664059143534, (1, 0): 0.7761945422235501, (2, 0): 1.156435334046472, (0, 1): 0.7511321901681971, (1, 1): 1.1300389075968607, (2, 1): 1.2000476400044584, (3, 1): 1.1312721564127197, (0, 2): 0.9625495406793108, (1, 2): 1.1095324006181142, (2, 2): 0, (0, 3): 0.8212908523158498, (1, 3): 0.94315711372322, (2, 3): 1.0214952963438189, (0, 4): 0.7558669756670711, (1, 4): 0.8197824149331983, (2, 4): 0.8851622374553743, (3, 4): 1.038607361115052, 'depot': 0.8432599170455328}, (0, 3): {(0, 0): 0.9474906225038169, (1, 0): 0.8834604588223623, (2, 0): 0.8361428457951043, (0, 1): 0.9343297211278754, (1, 1): 0.8396486142419826, (2, 1): 0.9043733718073463, (3, 1): 1.2341775616686337, (0, 2): 0.7536181960052628, (1, 2): 0.8465490248289592, (2, 2): 0.8743614459223469, (0, 3): 0, (1, 3): 0.9722433644194999, (2, 3): 0.8077178463073852, (0, 4): 1.0460153210829288, (1, 4): 1.1292682069531903, (2, 4): 0.9058543351898254, (3, 4): 0.9875117030768723, 'depot': 0.9850836655181687}, (1, 3): {(0, 0): 0.9661640068921608, (1, 0): 1.0924562778705078, (2, 0): 1.0483424785351214, (0, 1): 1.1287008314234959, (1, 1): 0.7923230895765079, (2, 1): 1.1478131714111486, (3, 1): 1.1386004269793253, (0, 2): 1.0996651400024624, (1, 2): 0.8969442354424293, (2, 2): 1.038716928084496, (0, 3): 1.0866141195055385, (1, 3): 0, (2, 3): 0.9356780945586974, (0, 4): 1.0515878086901191, (1, 4): 0.7650009039212313, (2, 4): 1.0925319798745514, (3, 4): 1.2304668887872343, 'depot': 0.8727058944801724}, (2, 3): {(0, 0): 0.8343357388220581, (1, 0): 1.1580818469981145, (2, 0): 0.8722176734202167, (0, 1): 0.8178556102824788, (1, 1): 1.1693964684207305, (2, 1): 0.967844603038738, (3, 1): 1.1474543245645736, (0, 2): 1.0665430949632733, (1, 2): 0.8712468328165697, (2, 2): 0.8992491772295156, (0, 3): 1.1748010781675482, (1, 3): 0.8619072661427, (2, 3): 0, (0, 4): 0.8079064320553928, (1, 4): 0.9075484900525166, (2, 4): 0.9000321980193834, (3, 4): 1.2006956317897972, 'depot': 1.2052909496201951}, (0, 4): {(0, 0): 1.2458018603088867, (1, 0): 1.0839478794632327, (2, 0): 1.2197086339270617, (0, 1): 0.99308932516574, (1, 1): 0.7533929427016539, (2, 1): 0.8059406615105713, (3, 1): 1.0516849537277506, (0, 2): 0.9649078117408171, (1, 2): 0.9361268894408272, (2, 2): 0.8069242469199335, (0, 3): 1.22475493135559, (1, 3): 1.132446780631982, (2, 3): 1.1603185389634127, (0, 4): 0, (1, 4): 0.9126995385310201, (2, 4): 1.1018589671721903, (3, 4): 0.8145160484117866, 'depot': 0.9224382778732398}, (1, 4): {(0, 0): 1.0270400447255434, (1, 0): 1.2091645881126296, (2, 0): 0.8372274227084073, (0, 1): 1.0869632114764995, (1, 1): 0.8373687155197651, (2, 1): 1.1552637117265165, (3, 1): 1.161510332844856, (0, 2): 0.8889634066486114, (1, 2): 1.0022206682989465, (2, 2): 0.802748614488771, (0, 3): 0.8316825338516749, (1, 3): 0.814140776923543, (2, 3): 1.102912040474344, (0, 4): 0.9864541595597585, (1, 4): 0, (2, 4): 0.9431752367008497, (3, 4): 0.7615200448218054, 'depot': 1.045352140550032}, (2, 4): {(0, 0): 1.0670826445725412, (1, 0): 0.9279510420569357, (2, 0): 1.2101784293017914, (0, 1): 0.9648876696577904, (1, 1): 0.9608906295405202, (2, 1): 1.2136711471469042, (3, 1): 1.16551152689357, (0, 2): 0.8632965763195822, (1, 2): 1.111361511746413, (2, 2): 0.8643889453189577, (0, 3): 1.15131406462539, (1, 3): 1.0768505620751876, (2, 3): 0.9022492492775205, (0, 4): 1.0229505510829688, (1, 4): 0.8304913533201165, (2, 4): 0, (3, 4): 0.8934858631336742, 'depot': 1.0874880264031186}, (3, 4): {(0, 0): 0.9123018141378364, (1, 0): 0.9250373531038703, (2, 0): 0.879663468175793, (0, 1): 0.7825275463810448, (1, 1): 0.8884207878072166, (2, 1): 0.7655447224624833, (3, 1): 0.7696874806167429, (0, 2): 0.8301646189759901, (1, 2): 0.8049489763930704, (2, 2): 0.7652414262462903, (0, 3): 1.1022587600886111, (1, 3): 0.8267132459134217, (2, 3): 0.9391444511837881, (0, 4): 0.8305288427080301, (1, 4): 0.9137129117241652, (2, 4): 1.1552604321458317, (3, 4): 0, 'depot': 0.902089302671107}}

    agvsPerWC = agvsPerWC_input  # Number of AGVs per workcenter
    agv_number_WC = []
    agv_count = 1
    for agv_WC in agvsPerWC:
        WC_AGV_list = []
        for No_agv_WC in range(agv_WC):
            WC_AGV_list.append(No_agv_WC + agv_count)
        agv_count += agv_WC
        agv_number_WC.append(WC_AGV_list)

# Scenario 2
if AGV_selection == 2:
    travel_time_matrix_static = {'depot': {(0, 0): 1.1846124087807195, (1, 0): 1.1790340431830697, (2, 0): 0.9613489811278979, (0, 1): 1.1335249027495813, (1, 1): 0.7588148732615743, (0, 2): 0.9887318837735783, (1, 2): 1.1429620503297773, (0, 3): 0.9254589871108735, (1, 3): 1.1174509406469084, (2, 3): 0.9688803070233487, (3, 3): 0.8980459983714959, (0, 4): 0.7986616316221349, (1, 4): 0.9941293954234647, (2, 4): 0.8046674431166847, (3, 4): 1.0234555873273503, (0, 5): 0.792020229216238, (1, 5): 0.8178662623488324, (0, 6): 0.9398789716911049, (1, 6): 0.940086411099907, (2, 6): 0.8979118750307242, (3, 6): 0.8165086842937422, (0, 7): 0.7854679553885884, (1, 7): 1.0935099973466196, (2, 7): 0.923543846895915, (3, 7): 1.0607278597917398, (4, 7): 0.8308787891570677, 'depot': 0}, (0, 0): {(0, 0): 0, (1, 0): 1.1598631319300343, (2, 0): 0.8988068791539637, (0, 1): 1.086655308916354, (1, 1): 0.7827524726485564, (0, 2): 1.024300677616169, (1, 2): 0.9827148294059131, (0, 3): 0.8732410027622906, (1, 3): 1.1306977666536262, (2, 3): 1.0465625208610476, (3, 3): 1.1779850505014484, (0, 4): 1.0289928671607171, (1, 4): 1.2424699980943514, (2, 4): 0.89944937052031, (3, 4): 1.149179949126026, (0, 5): 1.0598394649276954, (1, 5): 1.1583978744482364, (0, 6): 0.9307979336786567, (1, 6): 0.9140996766735705, (2, 6): 1.1465380214790553, (3, 6): 1.2197318257152774, (0, 7): 0.9421839439741415, (1, 7): 0.9552285091681577, (2, 7): 1.0118596344295785, (3, 7): 0.9263571235297305, (4, 7): 0.8058976794660544, 'depot': 0.9574495859931976}, (1, 0): {(0, 0): 0.8773235459304765, (1, 0): 0, (2, 0): 1.2141832723532615, (0, 1): 1.0634044937083929, (1, 1): 0.8128645148754918, (0, 2): 0.8422336979308722, (1, 2): 0.786614720696314, (0, 3): 1.0121287040547644, (1, 3): 0.7813926344126935, (2, 3): 1.1663365464593838, (3, 3): 0.8875065073898183, (0, 4): 1.0025784738122263, (1, 4): 1.1109627804376767, (2, 4): 1.206710593264443, (3, 4): 1.1613118667631321, (0, 5): 0.905197109382498, (1, 5): 0.8576637812789754, (0, 6): 0.9006590957514258, (1, 6): 0.7620638048412787, (2, 6): 1.2283469176893231, (3, 6): 1.0682549938257653, (0, 7): 0.9259416288180106, (1, 7): 1.1585507175005834, (2, 7): 0.848010646329324, (3, 7): 0.9218737463923515, (4, 7): 1.0861445234285327, 'depot': 0.9500194182708805}, (2, 0): {(0, 0): 0.7703170298768991, (1, 0): 0.796735894434911, (2, 0): 0, (0, 1): 1.0051456423779996, (1, 1): 1.2325287193006367, (0, 2): 1.0096631562842064, (1, 2): 1.2057070698965613, (0, 3): 0.9563478471985403, (1, 3): 0.7800905141691857, (2, 3): 0.9174152626972982, (3, 3): 1.2312398163485234, (0, 4): 1.1334122367497035, (1, 4): 0.8700824615214942, (2, 4): 0.7606224656334328, (3, 4): 1.0728822546395849, (0, 5): 0.782717058088617, (1, 5): 0.8063159177477665, (0, 6): 1.0069359971535778, (1, 6): 0.8730147097393517, (2, 6): 1.1052169298515286, (3, 6): 0.9112760178682522, (0, 7): 1.2478314042614862, (1, 7): 0.8133698416593886, (2, 7): 0.754705933635455, (3, 7): 1.2146099165661255, (4, 7): 1.0815146542861769, 'depot': 0.8598581653039552}, (0, 1): {(0, 0): 0.9535795402659976, (1, 0): 0.7586574990142599, (2, 0): 0.8416865032961098, (0, 1): 0, (1, 1): 1.0610719191521734, (0, 2): 1.2011138461863082, (1, 2): 0.8904412949225997, (0, 3): 1.1353023371650763, (1, 3): 0.8470535294463558, (2, 3): 1.1300564351043891, (3, 3): 1.062360243731971, (0, 4): 0.9537579421176956, (1, 4): 0.8400277631091637, (2, 4): 1.143838681200257, (3, 4): 0.8587589768154626, (0, 5): 0.8145823682992895, (1, 5): 1.1416659416888835, (0, 6): 1.167016455407814, (1, 6): 1.1556132473785872, (2, 6): 1.202788599065899, (3, 6): 0.9816179476601394, (0, 7): 1.1250924075759994, (1, 7): 1.1238719079514015, (2, 7): 1.1548472175483242, (3, 7): 1.0939912091251376, (4, 7): 1.0191369891814228, 'depot': 0.7502093710226451}, (1, 1): {(0, 0): 1.0797745183258747, (1, 0): 1.1257831932994695, (2, 0): 1.2159091920537706, (0, 1): 0.9908739851709135, (1, 1): 0, (0, 2): 1.0540404314292382, (1, 2): 1.0907493494499083, (0, 3): 0.7800947807253837, (1, 3): 0.8472880072690631, (2, 3): 1.0209948126480137, (3, 3): 1.1460838091902539, (0, 4): 1.2051885871111976, (1, 4): 0.7741281877591644, (2, 4): 0.9541511278147056, (3, 4): 1.2271076307320117, (0, 5): 1.1729611761062768, (1, 5): 0.9754723433926842, (0, 6): 1.0498311718634001, (1, 6): 0.806672503377152, (2, 6): 1.1431747049783276, (3, 6): 0.9061686087262033, (0, 7): 0.927049870690932, (1, 7): 1.0951528188973096, (2, 7): 0.9486836390675848, (3, 7): 0.7635084384573794, (4, 7): 1.2401861647562353, 'depot': 1.0126370872743984}, (0, 2): {(0, 0): 1.0564887569703025, (1, 0): 0.9993226339517525, (2, 0): 1.1617718126718597, (0, 1): 1.00295998376142, (1, 1): 0.8764035633824274, (0, 2): 0, (1, 2): 1.185408918454594, (0, 3): 0.9632148444849673, (1, 3): 0.9471693127881109, (2, 3): 0.8455431850447919, (3, 3): 1.115581690799817, (0, 4): 0.7710810775891371, (1, 4): 1.2485961089196467, (2, 4): 0.8909003761980597, (3, 4): 0.8266765583756486, (0, 5): 1.1914880739942868, (1, 5): 1.0035273927287314, (0, 6): 1.2379675797216092, (1, 6): 1.2320399539598565, (2, 6): 0.8799354689557969, (3, 6): 1.0750849165777974, (0, 7): 0.994676292777851, (1, 7): 0.898486626457194, (2, 7): 1.116558066537931, (3, 7): 0.8141754560850142, (4, 7): 1.2469406442571842, 'depot': 0.824950127736729}, (1, 2): {(0, 0): 0.7737842985271012, (1, 0): 0.8758045869197029, (2, 0): 1.1282084207837908, (0, 1): 0.8388174320223017, (1, 1): 1.1237201133910975, (0, 2): 1.0089990820232757, (1, 2): 0, (0, 3): 1.0884384674107725, (1, 3): 0.8927471918858401, (2, 3): 0.9134080813881562, (3, 3): 1.1051835591430392, (0, 4): 1.1202223444621149, (1, 4): 0.9420935335521827, (2, 4): 0.9626318909913758, (3, 4): 0.7644255763922847, (0, 5): 0.9260576843094532, (1, 5): 1.1015016784615197, (0, 6): 1.0505636737848523, (1, 6): 1.1316884399301863, (2, 6): 1.1595929882874414, (3, 6): 0.9520555056877814, (0, 7): 1.2096052279455096, (1, 7): 1.0841102393313389, (2, 7): 1.031757998426205, (3, 7): 0.839451218617138, (4, 7): 0.8953725922657481, 'depot': 1.0334137055933081}, (0, 3): {(0, 0): 1.1118945063295018, (1, 0): 1.0366135417162368, (2, 0): 1.144645134085422, (0, 1): 0.7870325684307578, (1, 1): 0.8005985444348581, (0, 2): 1.103583928021507, (1, 2): 0.9914061498147423, (0, 3): 0, (1, 3): 1.190746443385816, (2, 3): 1.231607620506571, (3, 3): 0.8362187796746365, (0, 4): 1.1141635019210459, (1, 4): 1.1883673021830683, (2, 4): 1.1708988359411072, (3, 4): 1.0297578575191122, (0, 5): 0.8726674540709782, (1, 5): 1.143214724697088, (0, 6): 1.2304326622214932, (1, 6): 0.8623634642817353, (2, 6): 0.7682400036938745, (3, 6): 0.8175861968640592, (0, 7): 0.8243387786614155, (1, 7): 1.155316414023526, (2, 7): 1.2356105164028506, (3, 7): 1.1125846579865817, (4, 7): 1.2398844471662882, 'depot': 0.937035588282201}, (1, 3): {(0, 0): 1.2193812678197538, (1, 0): 0.9935859562612677, (2, 0): 0.8778382548519272, (0, 1): 0.7635108346312857, (1, 1): 1.1531407830095368, (0, 2): 0.9282500798366289, (1, 2): 1.0694360885415577, (0, 3): 1.1793353716081942, (1, 3): 0, (2, 3): 0.8587243051729222, (3, 3): 0.9533665247419194, (0, 4): 1.161542019591279, (1, 4): 0.9806898014565852, (2, 4): 0.9256871262297715, (3, 4): 0.7585520641793976, (0, 5): 0.9361319107140629, (1, 5): 1.2433694035219358, (0, 6): 1.0351414350020658, (1, 6): 0.7931707910883965, (2, 6): 0.7837618333284313, (3, 6): 1.1253301076040516, (0, 7): 0.8054748134065828, (1, 7): 1.200014215750876, (2, 7): 1.1564842798790038, (3, 7): 1.1311123694967522, (4, 7): 0.8554186640332324, 'depot': 0.8724991908218562}, (2, 3): {(0, 0): 0.9234339852266684, (1, 0): 1.2121682811176924, (2, 0): 0.971350346125843, (0, 1): 0.8748478884029982, (1, 1): 0.8012906263553217, (0, 2): 0.8088179509459916, (1, 2): 1.1152453423760376, (0, 3): 1.2226757065571556, (1, 3): 1.1888697245062383, (2, 3): 0, (3, 3): 1.1811631386667594, (0, 4): 1.0604787121153938, (1, 4): 0.8125701089135648, (2, 4): 0.828733419207324, (3, 4): 1.0282270815258474, (0, 5): 1.2393641618353481, (1, 5): 1.1652856943256076, (0, 6): 1.1451399229014008, (1, 6): 1.0854838004619471, (2, 6): 0.9213817500774308, (3, 6): 1.048662196499751, (0, 7): 0.8619468618946741, (1, 7): 1.0504942069459497, (2, 7): 1.0289248932458808, (3, 7): 0.8315352061386068, (4, 7): 0.9087051416518335, 'depot': 0.944493573130883}, (3, 3): {(0, 0): 0.7806933446943499, (1, 0): 1.1126113737151972, (2, 0): 0.9138627249331055, (0, 1): 0.9399963199849035, (1, 1): 0.8462810445010477, (0, 2): 1.0423648988830647, (1, 2): 1.1466647659766358, (0, 3): 1.0851293769548642, (1, 3): 1.0179538740856586, (2, 3): 1.200438707646808, (3, 3): 0, (0, 4): 1.0164129265603459, (1, 4): 1.0574630487156629, (2, 4): 0.815656573258916, (3, 4): 0.9710374948537492, (0, 5): 1.1771819555464633, (1, 5): 0.9936122480259881, (0, 6): 1.1914975082326151, (1, 6): 0.9035200500182992, (2, 6): 0.7580401348748659, (3, 6): 1.1703196935034892, (0, 7): 0.9067813970461966, (1, 7): 1.050349619538307, (2, 7): 1.035562521835053, (3, 7): 0.9024142479416728, (4, 7): 0.8111689134329856, 'depot': 0.8670359536326505}, (0, 4): {(0, 0): 0.9425987833967546, (1, 0): 0.7966816411908111, (2, 0): 1.0490199468933366, (0, 1): 1.1191399032358054, (1, 1): 1.1987246377447902, (0, 2): 0.768921406164587, (1, 2): 1.2037786094655991, (0, 3): 1.1228359670462698, (1, 3): 1.008990642862158, (2, 3): 1.1072767340841074, (3, 3): 0.9956334220084524, (0, 4): 0, (1, 4): 0.9604602107334572, (2, 4): 0.9733409367147765, (3, 4): 0.7550734803133844, (0, 5): 1.2323025549938027, (1, 5): 0.8587470862024662, (0, 6): 0.8355053345579573, (1, 6): 0.9321187959133825, (2, 6): 0.8405597700158882, (3, 6): 1.0972390254947662, (0, 7): 0.88535312017268, (1, 7): 1.1085805172084728, (2, 7): 1.131459703525565, (3, 7): 1.128331603610034, (4, 7): 1.0919262650502188, 'depot': 1.123453801745324}, (1, 4): {(0, 0): 1.0976369346718986, (1, 0): 0.8576559125481946, (2, 0): 1.220803244254733, (0, 1): 0.7722244030207384, (1, 1): 1.0998272517369427, (0, 2): 1.0597436037964472, (1, 2): 0.9633219687278369, (0, 3): 1.0890781258297935, (1, 3): 0.7669299954266348, (2, 3): 0.7886628425709084, (3, 3): 1.035812172455877, (0, 4): 1.1367652190874526, (1, 4): 0, (2, 4): 1.226851463482078, (3, 4): 0.753175120659118, (0, 5): 1.0973948546909393, (1, 5): 1.1765332087954647, (0, 6): 1.0152244194787299, (1, 6): 1.1982290205563633, (2, 6): 1.2485923474936025, (3, 6): 1.0124220560810566, (0, 7): 1.1255432993235623, (1, 7): 0.8881450609310731, (2, 7): 1.1597198347081388, (3, 7): 1.1649037855636286, (4, 7): 0.9866050814290968, 'depot': 1.1642263131117998}, (2, 4): {(0, 0): 0.866066020783214, (1, 0): 1.0418638400150364, (2, 0): 1.0777096188018835, (0, 1): 0.9152774420567409, (1, 1): 1.0465602849022848, (0, 2): 0.7670864699816418, (1, 2): 1.1067884710089557, (0, 3): 0.8909986585711778, (1, 3): 0.9729363091901215, (2, 3): 1.2054030835632015, (3, 3): 0.9755800151113586, (0, 4): 1.0195628420456977, (1, 4): 1.183508112738457, (2, 4): 0, (3, 4): 0.9684319032842195, (0, 5): 0.8671991083966297, (1, 5): 1.1937167931303778, (0, 6): 0.9102234822626926, (1, 6): 1.2340030846060164, (2, 6): 0.7865732981639066, (3, 6): 1.1349735687364042, (0, 7): 0.933000223948142, (1, 7): 0.7734572258632649, (2, 7): 0.7519957518568727, (3, 7): 1.1978370293366774, (4, 7): 0.7972061075496314, 'depot': 0.7667650016567142}, (3, 4): {(0, 0): 0.8930127363247964, (1, 0): 1.116228264454106, (2, 0): 0.9199167549126119, (0, 1): 0.7522916136479623, (1, 1): 0.901494936456195, (0, 2): 0.9295909831296048, (1, 2): 1.1446939530096105, (0, 3): 1.0152500119108439, (1, 3): 0.7951672797955494, (2, 3): 1.0136545651834634, (3, 3): 1.1294235220031126, (0, 4): 0.7818743580902379, (1, 4): 0.7825667134382233, (2, 4): 1.0227325742914224, (3, 4): 0, (0, 5): 0.8917740529544729, (1, 5): 0.8115007259536825, (0, 6): 1.0199241528630318, (1, 6): 1.0157749640688296, (2, 6): 0.8261233319617153, (3, 6): 0.8374520864014388, (0, 7): 1.0374663821350267, (1, 7): 0.8759215891844563, (2, 7): 0.7656679054660493, (3, 7): 0.8882693566558563, (4, 7): 0.9928404467705485, 'depot': 1.1146529641869627}, (0, 5): {(0, 0): 0.9319661526606213, (1, 0): 1.2125997582973553, (2, 0): 1.1588611318732958, (0, 1): 0.7569548791835603, (1, 1): 1.170670352169569, (0, 2): 0.9305633770325918, (1, 2): 1.2436340029424269, (0, 3): 0.7939942593059317, (1, 3): 0.867473906826727, (2, 3): 1.0403976482310144, (3, 3): 0.7997464175878912, (0, 4): 1.2235549248924242, (1, 4): 1.09653934041311, (2, 4): 1.249117861331715, (3, 4): 1.2104442086911247, (0, 5): 0, (1, 5): 0.8444322922116219, (0, 6): 0.8865943478172222, (1, 6): 0.8847640379076565, (2, 6): 0.994404195569258, (3, 6): 1.042511445850573, (0, 7): 1.217297518451334, (1, 7): 1.1913757693032032, (2, 7): 0.8718900119811908, (3, 7): 0.7600487087698925, (4, 7): 0.8815178417985052, 'depot': 0.7894877240827611}, (1, 5): {(0, 0): 1.0478555521845418, (1, 0): 1.0357202387584679, (2, 0): 0.8147197171448542, (0, 1): 1.1996638460637612, (1, 1): 0.8595924834745011, (0, 2): 1.0024477469495714, (1, 2): 0.862394821630049, (0, 3): 0.944044062158128, (1, 3): 0.8821607301642159, (2, 3): 0.795199184425565, (3, 3): 0.917898654967879, (0, 4): 1.222863517897344, (1, 4): 1.010294678995382, (2, 4): 0.8124826354596901, (3, 4): 1.211242769356898, (0, 5): 0.8016267355521638, (1, 5): 0, (0, 6): 0.7828002009076904, (1, 6): 1.0978045577267894, (2, 6): 0.8957727736746628, (3, 6): 0.9071673133996803, (0, 7): 1.001161549213955, (1, 7): 1.228179181248016, (2, 7): 0.8657871773044323, (3, 7): 1.1336404803719187, (4, 7): 1.0133657558093998, 'depot': 0.8278433452018189}, (0, 6): {(0, 0): 0.787682190977204, (1, 0): 0.8304833413497879, (2, 0): 1.1457024848976818, (0, 1): 1.1933172632221138, (1, 1): 1.1341526311648544, (0, 2): 0.9973477426642179, (1, 2): 0.8247367082130128, (0, 3): 1.0476786910670888, (1, 3): 0.8435637936890024, (2, 3): 1.080881722192216, (3, 3): 1.0392876198775696, (0, 4): 0.7562230021047293, (1, 4): 1.1451547862986087, (2, 4): 1.0859766660711951, (3, 4): 0.7538190148700086, (0, 5): 0.7970739129893922, (1, 5): 0.9370226031209465, (0, 6): 0, (1, 6): 0.9001990965110716, (2, 6): 0.7529066274721742, (3, 6): 0.9741682223144394, (0, 7): 0.9003627039933464, (1, 7): 0.9133396620523948, (2, 7): 1.107725828517302, (3, 7): 1.1417316232512258, (4, 7): 0.7829253185028047, 'depot': 0.8242104230120977}, (1, 6): {(0, 0): 1.084378136433708, (1, 0): 1.0766608407817215, (2, 0): 0.797387616320276, (0, 1): 0.9842854129377114, (1, 1): 1.1857981650099707, (0, 2): 0.8215895797317045, (1, 2): 1.0415111327387434, (0, 3): 0.8279676192023908, (1, 3): 1.2043551555285075, (2, 3): 0.8473439308896134, (3, 3): 1.0899417633557715, (0, 4): 0.8467438639917905, (1, 4): 1.1912262507677356, (2, 4): 0.8980238674949692, (3, 4): 1.1252958873687637, (0, 5): 1.0577312895304884, (1, 5): 1.189900664054099, (0, 6): 0.9040750940684756, (1, 6): 0, (2, 6): 1.223935454816035, (3, 6): 1.1317144206811156, (0, 7): 1.24928107643242, (1, 7): 1.0130969954148281, (2, 7): 1.1414415394675432, (3, 7): 0.953821898552554, (4, 7): 1.1949511882151054, 'depot': 1.243834801197314}, (2, 6): {(0, 0): 0.9533690449175682, (1, 0): 0.7820377689233304, (2, 0): 0.8369131087885904, (0, 1): 0.8765344926048275, (1, 1): 0.8642445756209698, (0, 2): 0.8447062866920458, (1, 2): 1.1609301109963177, (0, 3): 1.1134537005543446, (1, 3): 0.8849982013425433, (2, 3): 0.9295627855050418, (3, 3): 0.943025149714534, (0, 4): 0.9397070291137214, (1, 4): 0.9253875729628165, (2, 4): 1.1302892767901875, (3, 4): 1.0907271570226353, (0, 5): 1.1869181091169347, (1, 5): 0.8517241615529694, (0, 6): 0.9181973739410703, (1, 6): 0.9509169173794202, (2, 6): 0, (3, 6): 0.9277051333335231, (0, 7): 0.7588925821555457, (1, 7): 0.7706179841207021, (2, 7): 0.838879922051828, (3, 7): 0.946370039994163, (4, 7): 1.1755460874149475, 'depot': 1.145376691341663}, (3, 6): {(0, 0): 0.8320858126977215, (1, 0): 1.1534866705485007, (2, 0): 1.1244260980547556, (0, 1): 0.9278685927405542, (1, 1): 1.197074090765527, (0, 2): 1.148600359744463, (1, 2): 1.0146046572102017, (0, 3): 0.9221867371745323, (1, 3): 0.8393640673557918, (2, 3): 0.8029820253846833, (3, 3): 1.0389825796429966, (0, 4): 1.0935903215020661, (1, 4): 0.7756122684350009, (2, 4): 0.9326260880430896, (3, 4): 1.0432963514850546, (0, 5): 0.9430440008058539, (1, 5): 0.8635301370527977, (0, 6): 1.1607993624807957, (1, 6): 0.8298858674381384, (2, 6): 1.2059057460498301, (3, 6): 0, (0, 7): 0.7882528759760452, (1, 7): 0.9498566701729444, (2, 7): 0.90614409918519, (3, 7): 1.2028520609357918, (4, 7): 1.0064363588837733, 'depot': 0.9135912728529805}, (0, 7): {(0, 0): 1.0427863184233186, (1, 0): 0.8698515396477857, (2, 0): 1.1928031694569827, (0, 1): 0.7896549640018462, (1, 1): 1.1441407564425747, (0, 2): 1.1250115482427896, (1, 2): 0.8603364675767766, (0, 3): 1.2463817662378762, (1, 3): 1.1681538225720194, (2, 3): 0.9693141953580136, (3, 3): 0.8182355392218916, (0, 4): 1.0955375798519638, (1, 4): 1.2440478005019746, (2, 4): 1.2098883601018795, (3, 4): 0.8773293899379289, (0, 5): 0.9611276655667437, (1, 5): 0.8671838151123394, (0, 6): 0.8205707836982494, (1, 6): 0.9060642128598764, (2, 6): 0.8942497856809641, (3, 6): 1.1327845574920077, (0, 7): 0, (1, 7): 1.2181759934676313, (2, 7): 1.165968537283654, (3, 7): 1.1128765940397045, (4, 7): 1.1909571837239006, 'depot': 0.9899952799219891}, (1, 7): {(0, 0): 1.0976146100401936, (1, 0): 1.1258033714256928, (2, 0): 1.0573082323777108, (0, 1): 1.1536648707806494, (1, 1): 1.1003209951484307, (0, 2): 0.7913971373455322, (1, 2): 1.2437661576548904, (0, 3): 0.9278819444746007, (1, 3): 0.9932487976919562, (2, 3): 1.2255082941335493, (3, 3): 0.8279618448143085, (0, 4): 0.8743477949152282, (1, 4): 0.9355251892159793, (2, 4): 1.1448175932360551, (3, 4): 0.9172713593077362, (0, 5): 1.0494038889927773, (1, 5): 1.011277614241819, (0, 6): 0.7816026383225394, (1, 6): 1.1250643250360826, (2, 6): 1.1750590002805261, (3, 6): 1.0063535773133219, (0, 7): 0.890904956961287, (1, 7): 0, (2, 7): 1.1768222693074337, (3, 7): 1.193776564572177, (4, 7): 1.0056570365249686, 'depot': 0.7524595399386613}, (2, 7): {(0, 0): 0.9265054316209997, (1, 0): 0.8139015148264455, (2, 0): 0.8478909651570228, (0, 1): 0.8430566495226114, (1, 1): 1.078969754874454, (0, 2): 0.9260327499594074, (1, 2): 1.19840513259865, (0, 3): 1.1772662895820365, (1, 3): 1.0722510079059149, (2, 3): 1.0207853622417604, (3, 3): 1.1865036508678832, (0, 4): 1.2314115539127257, (1, 4): 1.0869054559728273, (2, 4): 0.8852527198196458, (3, 4): 1.015548182824691, (0, 5): 1.0071750910100263, (1, 5): 0.893478526510459, (0, 6): 1.172514146865721, (1, 6): 1.1210207402924708, (2, 6): 0.8145958054936274, (3, 6): 1.1643248095066814, (0, 7): 0.8705184107350904, (1, 7): 1.1186348959798078, (2, 7): 0, (3, 7): 1.1006908905774944, (4, 7): 1.1141606061076317, 'depot': 0.938832933361149}, (3, 7): {(0, 0): 0.9632073958595619, (1, 0): 0.9683255524985924, (2, 0): 0.9004915049932267, (0, 1): 1.0542947014435418, (1, 1): 1.1026983255516627, (0, 2): 0.8755991364659119, (1, 2): 0.9751728444370771, (0, 3): 0.7537006960221289, (1, 3): 1.178118930448179, (2, 3): 0.9142300419901157, (3, 3): 1.076354226756235, (0, 4): 1.243687101223477, (1, 4): 1.1958251879357011, (2, 4): 0.8422395927178128, (3, 4): 1.2445281311625327, (0, 5): 0.8734130693122213, (1, 5): 1.210684696639857, (0, 6): 1.0603641984948258, (1, 6): 1.1563483606304659, (2, 6): 1.0514173537505433, (3, 6): 1.1041045192046433, (0, 7): 0.9972810995891288, (1, 7): 0.7898241815435278, (2, 7): 0.8494338594560935, (3, 7): 0, (4, 7): 1.095787304869928, 'depot': 1.0089387164449062}, (4, 7): {(0, 0): 1.2468389362523302, (1, 0): 1.0820785451282757, (2, 0): 1.0008743743131312, (0, 1): 0.9978837131735736, (1, 1): 0.9826420183183688, (0, 2): 1.148530750362058, (1, 2): 0.9276641883931676, (0, 3): 1.2378650508967708, (1, 3): 1.08012992962512, (2, 3): 0.7593251622537474, (3, 3): 0.902917971843997, (0, 4): 0.934750175928152, (1, 4): 0.9806579590206381, (2, 4): 0.8437122720000619, (3, 4): 1.1215243792376433, (0, 5): 1.1004310159702142, (1, 5): 1.2276288068678387, (0, 6): 1.1958469547143133, (1, 6): 0.8246930409429603, (2, 6): 1.0469904891142336, (3, 6): 1.1928855015565212, (0, 7): 0.8693583976500912, (1, 7): 0.9532670087363231, (2, 7): 0.8101215224391145, (3, 7): 0.8117320190074668, (4, 7): 0, 'depot': 0.774234781084673}}
    travel_time_matrix_static = create_distance_matrix()
    agvsPerWC = agvsPerWC_input  # Number of AGVs per workcenter
    agv_number_WC = []
    agv_count = 1
    for agv_WC in agvsPerWC:
        WC_AGV_list = []
        for No_agv_WC in range(agv_WC):
            WC_AGV_list.append(No_agv_WC+agv_count)
        agv_count += agv_WC
        agv_number_WC.append(WC_AGV_list)"""