"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""

import itertools
import random
import simpy
import warnings
import csv
import math
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
import numpy as np
import pandas as pd

import Random_Numpy_Parser as Random_Numpy
from Travel_matrix_AGVS import Travel_matrix_AGVS as Travel_matrix
import generate_scenario as generate_scenario

from collections import defaultdict
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from itertools import groupby
from simpy import *
from datetime import datetime

warnings.filterwarnings('ignore')

# General Settings
number = 0  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10_000  # Runtime limit

SAVE = True

scenario = "scenario_4"

#              SCENARIO ------ WC - JT - MACH - PROC ---- AGVS -- MAXWIP - TIME - SEED - UTI
#              =======================================================================================================
situations = {'scenario_1': [[5, 2], 5, 16, [2, 9], [1, 1, 1, 1, 1], 250, 10_000, 150, 0.95],
              'scenario_2': [[5, 2], 5, 16, [10, 50], [0, 0, 1, 0, 0], 300, 30_000, 150, 0.95],
              'scenario_3': [[5, 2], 5, 32, [2, 9], [2, 2, 3, 2, 2], 350, 10_000, 150, 0.85],
              'scenario_4': [[5, 2], 5, 32, [10, 50], [0, 0, 2, 0, 0], 400, 30_000, 150, 0.85],  #   !--------
              'scenario_5': [[5, 2], 20, 16, [2, 9], [2, 2, 2, 2, 2], 250, 10_000, 150, 0.90],
              'scenario_6': [[5, 2], 20, 16, [10, 50], [2, 2, 2, 2, 2], 300, 40_000, 150, 0.75],
              'scenario_7': [[5, 2], 20, 32, [2, 9], [2, 2, 2, 2, 2], 400, 10_000, 150, 0.75],
              'scenario_8': [[5, 2], 20, 32, [10, 50], [2, 2, 2, 2, 2], 300, 30_000, 150, 0.55],
              'scenario_9': [[10, 2], 5, 16, [2, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 400, 20_000, 150, 0.45],
              'scenario_10': [[10, 2], 5, 16, [10, 50], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 500, 40_000, 150, 0.60],
              'scenario_11': [[10, 2], 5, 32, [2, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 500, 10_000, 150, 0.45],
              'scenario_12': [[10, 2], 5, 32, [10, 50], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 250, 10_000, 150, 0.90],
              'scenario_13': [[10, 2], 20, 16, [2, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 500, 10_000, 150, 0.45],
              'scenario_14': [[10, 2], 20, 16, [10, 50], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 250, 10_000, 150, 0.90],
              'scenario_15': [[10, 2], 20, 32, [2, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 500, 10_000, 150, 0.55],
              'scenario_16': [[10, 2], 20, 32, [10, 50], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 250, 10_000, 150, 0.90]}


# Working situations:
# - scenario_1


max_workcenters = situations[scenario][0][0]
min_workcenters = situations[scenario][0][1]
no_of_jobs = situations[scenario][1]
total_machines = situations[scenario][2]
min_proc = situations[scenario][3][0]
max_proc = situations[scenario][3][1]
agvsPerWC_new = situations[scenario][4]
max_wip = situations[scenario][5]
maxTime = situations[scenario][6]
seed = situations[scenario][7]
uti = situations[scenario][8]
setup_factor = 0.20

processingTimes, operationOrder, machinesPerWC, setupTime, demand, job_priority, arrival_rate, machine_number_WC, CR, DDT = generate_scenario.new_scenario(
        max_workcenters, min_workcenters, no_of_jobs, total_machines, min_proc, max_proc, setup_factor, uti, seed)


numberOfOperations = [len(i) for i in operationOrder]
noOfWC = range(len(machinesPerWC))

arrival_rate = [arrival_rate[0] - 0]

created_travel_time_matrix, agvsPerWC, agv_number_WC = Travel_matrix.choose_distance_matrix(agvsPerWC_new, machinesPerWC, seed)

max_job_priority = max(job_priority)
max_setup_time = np.amax(setupTime)
max_processing_time = 0
max_remaining_processing_time = 0
for idx, job_type in enumerate(processingTimes):
    max_pt = max(job_type)
    summed_pt = sum(processingTimes[idx])

    if max_pt > max_processing_time:
        max_processing_time = max_pt
    if summed_pt > max_remaining_processing_time:
        max_remaining_processing_time = summed_pt

job_location_set = {"d": 1}
for location in noOfWC:
    job_location_set[location] = location+2



"Initial parameters of the GES"
noAttributesMA = 9
noAttributesAGV = 7

noAttributesJobMA = 4
noAttributesJobAGV = 6



totalAttributes = max(noAttributesMA + noAttributesJobMA, noAttributesAGV + noAttributesJobAGV)

FIFO_agv_queue = False  # True is FIFO enabled

no_generation = 1000


# %% Dispatching rules


# 3: Nearest Idle AGV Rule (AGV & JOB)
def dispatch_rule_3(env, jobs, noOfAGVs, currentWC, job_shop):  # Closest AGV dispatching rule
    """ Dispatching rule that searches for the nearest idle AGV per job"""

    no_of_jobs = len(jobs)
    dispatched_jobs = []
    dispatched_agvs = []

    available_agvs, relative_agv_numbers, no_avail_agvs = check_available_agvs(noOfAGVs, currentWC, job_shop)

    driving_times_per_job = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):

        driving_times_agvs = [0] * no_avail_agvs

        for agv_idx, agv in enumerate(available_agvs):
            pick_time = job_shop.travel_time_matrix[agv[1]][job.location]

            driving_times_agvs[agv_idx] = pick_time + 0.00001

        driving_times_per_job[job_idx] = driving_times_agvs

    arr = np.array(driving_times_per_job)

    for _ in jobs:

        if no_avail_agvs == 0:
            break

        result = np.where(arr == np.amin(arr))

        best_job = result[0][0]
        best_agv = result[1][0]

        arr[:][best_job] = float("inf")

        for job_idx, _ in enumerate(jobs):
            arr[job_idx][best_agv] = float("inf")

        no_avail_agvs -= 1

        dispatched_jobs.append(best_job)
        dispatched_agvs.append(best_agv)

    dispatched_agvs_enumerated = []
    for ii, vv in enumerate(dispatched_jobs):
        dispatched_agvs_enumerated.append(relative_agv_numbers[dispatched_agvs[ii]])

    return dispatched_jobs, dispatched_agvs_enumerated


# 4: Random AGV Rule (AGV) - Random Job Rule (Job)
def dispatch_rule_4(env, jobs, noOfAGVs, currentWC, job_shop):
    """ Dispatching rule that selects job and machines at random  """

    dispatched_jobs = []
    dispatched_agvs = []

    available_agvs, relative_agv_numbers, no_avail_agvs = check_available_agvs(noOfAGVs, currentWC, job_shop)

    job_list = []
    for job in range(len(jobs)):
        job_list.append(job)

    for _ in jobs:

        if no_avail_agvs == 0:
            break

        best_agv = np.random.choice(relative_agv_numbers)
        best_job = np.random.choice(job_list)

        dispatched_agvs.append(best_agv)
        dispatched_jobs.append(best_job)

        relative_agv_numbers.remove(best_agv)
        job_list.remove(best_job)

        no_avail_agvs -= 1

    return dispatched_jobs, dispatched_agvs


# 5: Longest Time In System Rule (JOB) - Minimal Distance Rule (AGV)
def dispatch_rule_5(env, jobs, noOfAGVs, currentWC, job_shop):
    """ Dispatching rules that selects the Job that is longest is the system and then
      checks which available AGV has the minimal driving time from pickup to unload"""

    no_of_jobs = len(jobs)
    dispatched_jobs = []
    dispatched_agvs = []

    available_agvs, relative_agv_numbers, no_avail_agvs = check_available_agvs(noOfAGVs, currentWC, job_shop)

    job_in_system_times = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):
        job_in_system_time = env.now - job.arrival_time_system
        job_in_system_times[job_idx] = job_in_system_time

    driving_times_per_job = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):

        driving_times_agvs = [0] * no_avail_agvs

        for agv_idx, agv in enumerate(available_agvs):
            driving_times_agvs[agv_idx] = job_shop.travel_time_matrix[agv[1]][job.location] + 0.00001

        driving_times_per_job[job_idx] = driving_times_agvs

    arr = np.array(driving_times_per_job)

    for _ in jobs:

        if no_avail_agvs == 0:
            break

        best_job = np.argmax(job_in_system_times)
        result = np.where(arr[best_job] == np.amin(arr[best_job]))
        best_agv = result[0][0]

        arr[:][best_job] = float("inf")

        for job_idx, _ in enumerate(jobs):
            arr[job_idx][best_agv] = float("inf")

        dispatched_jobs.append(best_job)
        dispatched_agvs.append(best_agv)

        no_avail_agvs -= 1
        job_in_system_times[best_job] = -1

    dispatched_agvs_enumerated = []
    for ii, vv in enumerate(dispatched_jobs):
        dispatched_agvs_enumerated.append(relative_agv_numbers[dispatched_agvs[ii]])

    return dispatched_jobs, dispatched_agvs_enumerated


# 6: Longest Waiting Time at Pickup Point (JOB) - Minimal Transfer Rule (AGV)
def dispatch_rule_6(env, jobs, noOfAGVs, currentWC, job_shop):
    """ Dispatching rule that selects the job with the longest waiting time at a pickup
    point and then selects an available AGV with the minimal pick and unload time"""

    no_of_jobs = len(jobs)
    dispatched_jobs = []
    dispatched_agvs = []

    available_agvs, relative_agv_numbers, no_avail_agvs = check_available_agvs(noOfAGVs, currentWC, job_shop)

    job_at_pick_up_times = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):
        job_waiting_time = env.now - job.finishing_time_machine
        job_at_pick_up_times[job_idx] = job_waiting_time

    driving_times_per_job = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):

        driving_times_agvs = [0] * no_avail_agvs

        for agv_idx, agv in enumerate(available_agvs):
            driving_times_agvs[agv_idx] = job_shop.travel_time_matrix[agv[1]][job.location] + 0.00001

        driving_times_per_job[job_idx] = driving_times_agvs

    arr = np.array(driving_times_per_job)

    for _ in jobs:

        if no_avail_agvs == 0:
            break

        best_job = np.argmax(job_at_pick_up_times)
        result = np.where(arr[best_job] == np.amin(arr[best_job]))

        best_agv = result[0][0]

        arr[:][best_job] = float("inf")

        for job_idx, _ in enumerate(jobs):
            arr[job_idx][best_agv] = float("inf")

        dispatched_jobs.append(best_job)
        dispatched_agvs.append(best_agv)

        no_avail_agvs -= 1
        job_at_pick_up_times[best_job] = -1

    dispatched_agvs_enumerated = []
    for ii, vv in enumerate(dispatched_jobs):
        dispatched_agvs_enumerated.append(relative_agv_numbers[dispatched_agvs[ii]])

    return dispatched_jobs, dispatched_agvs_enumerated


# 7: Longest Average Waiting Time At Pickup Point (JOB) - Minimal Transfer Rule (AGV)
def dispatch_rule_7(env, jobs, noOfAGVs, currentWC, job_shop):
    """ Dispatching rule that selects the job with the longest average waiting time at a pickup
    point and then selects an available AGV with the minimal pick and unload time"""

    no_of_jobs = len(jobs)
    dispatched_jobs = []
    dispatched_agvs = []

    available_agvs, relative_agv_numbers, no_avail_agvs = check_available_agvs(noOfAGVs, currentWC, job_shop)

    job_at_pick_up_average_times = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):
        job_waiting_time = env.now - job.average_waiting_time_pickup
        job_at_pick_up_average_times[job_idx] = job_waiting_time

    driving_times_per_job = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):

        driving_times_agvs = [0] * no_avail_agvs

        for agv_idx, agv in enumerate(available_agvs):
            driving_times_agvs[agv_idx] = job_shop.travel_time_matrix[agv[1]][job.location] + 0.00001

        driving_times_per_job[job_idx] = driving_times_agvs

    arr = np.array(driving_times_per_job)

    for _ in jobs:

        if no_avail_agvs == 0:
            break

        best_job = np.argmax(job_at_pick_up_average_times)
        result = np.where(arr[best_job] == np.amin(arr[best_job]))

        best_agv = result[0][0]

        arr[:][best_job] = float("inf")

        for job_idx, _ in enumerate(jobs):
            arr[job_idx][best_agv] = float("inf")

        dispatched_jobs.append(best_job)
        dispatched_agvs.append(best_agv)

        no_avail_agvs -= 1
        job_at_pick_up_average_times[best_job] = -1

    dispatched_agvs_enumerated = []
    for ii, vv in enumerate(dispatched_jobs):
        dispatched_agvs_enumerated.append(relative_agv_numbers[dispatched_agvs[ii]])

    return dispatched_jobs, dispatched_agvs_enumerated


# 8: Earliest Due Time (JOB) - Minimal Transfer Rule (AGV)
def dispatch_rule_8(env, jobs, noOfAGVs, currentWC, job_shop):
    """ Dispatching rule that selects the job with the earliest due time
     and then selects an available AGV with the minimal pick and unload time"""

    no_of_jobs = len(jobs)
    dispatched_jobs = []
    dispatched_agvs = []

    available_agvs, relative_agv_numbers, no_avail_agvs = check_available_agvs(noOfAGVs, currentWC, job_shop)

    job_earliest_due_times = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):
        job_earliest_due_time = job.dueDate[-1]
        job_earliest_due_times[job_idx] = job_earliest_due_time

    driving_times_per_job = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):

        driving_times_agvs = [0] * no_avail_agvs

        for agv_idx, agv in enumerate(available_agvs):
            driving_times_agvs[agv_idx] = job_shop.travel_time_matrix[agv[1]][job.location] + 0.00001

        driving_times_per_job[job_idx] = driving_times_agvs

    arr = np.array(driving_times_per_job)

    for _ in jobs:

        if no_avail_agvs == 0:
            break

        best_job = np.argmin(job_earliest_due_times)
        result = np.where(arr[best_job] == np.amin(arr[best_job]))

        best_agv = result[0][0]

        arr[:][best_job] = float("inf")

        for job_idx, _ in enumerate(jobs):
            arr[job_idx][best_agv] = float("inf")

        dispatched_jobs.append(best_job)
        dispatched_agvs.append(best_agv)

        no_avail_agvs -= 1
        job_earliest_due_times[best_job] = float("inf")

    dispatched_agvs_enumerated = []
    for ii, vv in enumerate(dispatched_jobs):
        dispatched_agvs_enumerated.append(relative_agv_numbers[dispatched_agvs[ii]])

    return dispatched_jobs, dispatched_agvs_enumerated


# 9: Earliest Release Time (JOB) - Minimal Transfer Rule (AGV)
def dispatch_rule_9(env, jobs, noOfAGVs, currentWC, job_shop):
    """ Dispatching rule that selects the job with the earliest release time
     and then selects an available AGV with the minimal pick and unload time"""

    no_of_jobs = len(jobs)
    dispatched_jobs = []
    dispatched_agvs = []

    available_agvs, relative_agv_numbers, no_avail_agvs = check_available_agvs(noOfAGVs, currentWC, job_shop)

    job_earliest_release_times = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):
        job_earliest_release_time = job.dueDate[0]
        job_earliest_release_times[job_idx] = job_earliest_release_time

    driving_times_per_job = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):

        driving_times_agvs = [0] * no_avail_agvs

        for agv_idx, agv in enumerate(available_agvs):
            driving_times_agvs[agv_idx] = job_shop.travel_time_matrix[agv[1]][job.location] + 0.00001

        driving_times_per_job[job_idx] = driving_times_agvs

    arr = np.array(driving_times_per_job)

    for _ in jobs:

        if no_avail_agvs == 0:
            break

        best_job = np.argmin(job_earliest_release_times)
        result = np.where(arr[best_job] == np.amin(arr[best_job]))

        best_agv = result[0][0]

        arr[:][best_job] = float("inf")

        for job_idx, _ in enumerate(jobs):
            arr[job_idx][best_agv] = float("inf")

        dispatched_jobs.append(best_job)
        dispatched_agvs.append(best_agv)

        no_avail_agvs -= 1
        job_earliest_release_times[best_job] = float("inf")

    dispatched_agvs_enumerated = []
    for ii, vv in enumerate(dispatched_jobs):
        dispatched_agvs_enumerated.append(relative_agv_numbers[dispatched_agvs[ii]])

    return dispatched_jobs, dispatched_agvs_enumerated


# %% General functions

def check_available_agvs(noOfAGVs, currentWC, job_shop):
    available_agvs = []
    relative_agv_numbers = []
    relative_count = 0

    for jj in range(noOfAGVs):
        agv = job_shop.agv_queue_per_wc[jj, currentWC - 1]
        available_agvs.append(agv)
        relative_agv_numbers.append(relative_count)
        no_avail_agvs = len(available_agvs)
        relative_count += 1

    return available_agvs, relative_agv_numbers, no_avail_agvs


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


def dispatch_control(env, jobs, noOfAGVs, currentWC, job_shop, agvs, AGVstore, dispatch_rule_no, agvsPerWC,
                     agv_number_WC):
    """  With the use of dispatching rules the APA looks for the best job and best available AGV"""

    # 3: Nearest Idle AGV rule (AGV & JOB)
    if dispatch_rule_no == 3:
        dispatched_jobs, dispatched_agvs = dispatch_rule_3(env, jobs, agvsPerWC[currentWC - 1], currentWC, job_shop)

    # 4: Random AGV Rule (AGV) - Random Job Rule (Job)
    if dispatch_rule_no == 4:
        dispatched_jobs, dispatched_agvs = dispatch_rule_4(env, jobs, agvsPerWC[currentWC - 1], currentWC, job_shop)

    # 5: Longest Time In System Rule (JOB) - Minimal Distance Rule (AGV)
    if dispatch_rule_no == 5:
        dispatched_jobs, dispatched_agvs = dispatch_rule_5(env, jobs, agvsPerWC[currentWC - 1], currentWC, job_shop)

    # 6: Longest Waiting Time at Pickup Point (JOB) - Minimal Transfer Rule (AGV)
    if dispatch_rule_no == 6:
        dispatched_jobs, dispatched_agvs = dispatch_rule_6(env, jobs, agvsPerWC[currentWC - 1], currentWC, job_shop)

    # 7: Longest Average Waiting Time At Pickup Point (JOB) - Minimal Transfer Rule (AGV)
    if dispatch_rule_no == 7:
        dispatched_jobs, dispatched_agvs = dispatch_rule_7(env, jobs, agvsPerWC[currentWC - 1], currentWC, job_shop)

    # 8: Earliest Due Time (JOB) - Minimal Transfer Rule (AGV)
    if dispatch_rule_no == 8:
        dispatched_jobs, dispatched_agvs = dispatch_rule_8(env, jobs, agvsPerWC[currentWC - 1], currentWC, job_shop)

    # 9: Earliest Release Time (JOB) - Minimal Transfer Rule (AGV)
    if dispatch_rule_no == 9:
        dispatched_jobs, dispatched_agvs = dispatch_rule_9(env, jobs, agvsPerWC[currentWC - 1], currentWC, job_shop)

    # Put the winning jobs in the AGV queues
    for ii, vv in enumerate(dispatched_jobs):
        put_job_in_agv_queue(currentWC, dispatched_agvs[ii], jobs[vv], job_shop, agvs)

    # Remove job from queue of the APA
    for ii in reversed(sorted(dispatched_jobs)):
        yield AGVstore.get(lambda mm: mm == jobs[ii])


def bid_winner_agv_all_WC(env, jobs, noOfAGVs, currentWC, job_shop, agvs, AGVstore,
                          normaliziation_range, agv_number_WC):
    """Used to calulcate the bidding values for each job in the pool, for each AGV in the system.
        Then checks which AGV gets which job based on these bidding values."""

    current_bid = [0] * sum(noOfAGVs)
    current_job = [0] * sum(noOfAGVs)
    best_bid = []
    best_job = []
    no_of_jobs = len(jobs)
    total_rp = [0] * no_of_jobs

    # Get the remaning processing time
    for jj in range(no_of_jobs):
        total_rp[jj] = (remain_processing_time(jobs[jj]))

    # Get the bids for all AGVs for each job
    for wc in range(len(agv_number_WC)):
        for jj in range(agvsPerWC[wc]):

            agv_number = agv_number_WC[wc][jj]
            agv = job_shop.agv_queue_per_wc[jj, wc]
            agv_queue_total_distance = (queue_total_distance(agv, job_shop))
            new_bid = [0] * no_of_jobs

            for ii, job in enumerate(jobs):
                attributes = bid_calculation_agv(job_shop.weights, agv_number,
                                                 normaliziation_range, agv, job, job_shop,
                                                 job.processingTime[job.currentOperation - 1], job.priority,
                                                 total_rp[ii], job.dueDate[job.numberOfOperations], env.now,
                                                 agv_queue_total_distance)
                new_bid[ii] = attributes

            ind_winning_job = new_bid.index(max(new_bid))

            current_bid[agv_number - 1] = new_bid[ind_winning_job]
            current_job[agv_number - 1] = ind_winning_job

    # Determine the winning bids
    sorted_list = sorted(list_duplicates(current_job))
    for dup in sorted_list:
        bestAGV = dup[1][0]
        bestbid = current_bid[bestAGV]

        for ii in dup[1]:
            if bestbid <= current_bid[ii]:
                bestbid = current_bid[ii]
                bestAGV = ii

        best_bid.append(bestAGV)  # AGV winner
        best_job.append(int(dup[0]))  # Job to be transfered

    # Put the winning jobs in the AGV queues
    for ii, vv in enumerate(best_job):
        agv_number = best_bid[ii] + 1

        for WC_value in range(len(agv_number_WC)):
            if agv_number in agv_number_WC[WC_value]:
                dedicated_WC = WC_value + 1
                dedicated_AGV = agv_number_WC[WC_value].index(agv_number)

        put_job_in_agv_queue(dedicated_WC, dedicated_AGV, jobs[vv], job_shop, agvs)


    # Remove job from queue of the APA
    for ii in reversed(best_job):
        yield AGVstore.get(lambda mm: mm == jobs[ii])


def bid_winner_agv_per_WC(env, jobs, noOfAGVs, currentWC, job_shop, agvs, AGVstore,
                          normaliziation_range, agv_number_WC):
    """Used to calulcate the bidding values for each job in the pool, for each AGV.
        Then checks which AGV gets which job based on these bidding values."""

    current_bid = [0] * noOfAGVs
    current_job = [0] * noOfAGVs
    best_bid = []
    best_job = []
    no_of_jobs = len(jobs)
    total_rp = [0] * no_of_jobs

    # Get the remaning processing time
    for jj in range(no_of_jobs):
        total_rp[jj] = (remain_processing_time(jobs[jj]))

    # Get the bids for all AGVs for each job
    for jj in range(noOfAGVs):
        agv = job_shop.agv_queue_per_wc[jj, currentWC - 1]
        agv_queue_total_distance = (queue_total_distance(agv, job_shop))
        new_bid = [0] * no_of_jobs
        for ii, job in enumerate(jobs):
            attributes = bid_calculation_agv(job_shop.weights, agv_number_WC[currentWC - 1][jj],
                                             normaliziation_range, agv, job, job_shop,
                                             job.processingTime[job.currentOperation - 1], job.priority,
                                             total_rp[ii], job.dueDate[job.numberOfOperations], env.now,
                                             agv_queue_total_distance)
            new_bid[ii] = attributes

        ind_winning_job = new_bid.index(max(new_bid))
        current_bid[jj] = new_bid[ind_winning_job]
        current_job[jj] = ind_winning_job

    # Determine the winning bids
    sorted_list = sorted(list_duplicates(current_job))
    for dup in sorted_list:
        bestAGV = dup[1][0]
        bestbid = current_bid[bestAGV]

        for ii in dup[1]:
            if bestbid <= current_bid[ii]:
                bestbid = current_bid[ii]
                bestAGV = ii

        best_bid.append(bestAGV)  # AGV winner
        best_job.append(int(dup[0]))  # Job to be transfered

    # Put the winning jobs in the AGV queues
    for ii, vv in enumerate(best_job):
        put_job_in_agv_queue(currentWC, best_bid[ii], jobs[vv], job_shop, agvs)


    # Remove job from queue of the APA
    for ii in reversed(best_job):
        yield AGVstore.get(lambda mm: mm == jobs[ii])


def bid_winner_ma(env, jobs, noOfMachines, currentWC, job_shop, machine, store,
                  normaliziation_range):
    """Used to calulcate the bidding values for each job in the pool, for each machine.
    Then checks which machine gets which job based on these bidding values."""
    current_bid = [0] * noOfMachines
    current_job = [0] * noOfMachines
    best_bid = []
    best_job = []
    no_of_jobs = len(jobs)
    total_rp = [0] * no_of_jobs

    # Get the remaning processing time
    for jj in range(no_of_jobs):
        total_rp[jj] = (remain_processing_time(jobs[jj]))

    # Get the bids for all machines
    for jj in range(noOfMachines):
        queue_length = len(machine[(jj, currentWC - 1)].items)
        new_bid = [0] * no_of_jobs
        total_pt_ma_queue = (total_processing_time_ma_queue(machine[(jj, currentWC - 1)].items))
        for ii, job in enumerate(jobs):
            attributes = bid_calculation_ma(job_shop.weights, machine_number_WC[currentWC - 1][jj],
                                            job.processingTime[job.currentOperation - 1], job.currentOperation,
                                            total_rp[ii], job.dueDate[job.numberOfOperations],
                                            env.now, job.priority, queue_length, total_pt_ma_queue,
                                            normaliziation_range, job.numberOfOperations)

            new_bid[ii] = attributes

        ind_winning_job = new_bid.index(max(new_bid))
        current_bid[jj] = new_bid[ind_winning_job]
        current_job[jj] = ind_winning_job

    # Determine the winning bids
    sorted_list = sorted(list_duplicates(current_job))

    for dup in sorted_list:
        bestmachine = dup[1][0]
        bestbid = current_bid[bestmachine]

        for ii in dup[1]:
            if bestbid <= current_bid[ii]:
                bestbid = current_bid[ii]
                bestmachine = ii

        best_bid.append(bestmachine)  # Machine winner
        best_job.append(int(dup[0]))  # Job to be processed

    # APA Trigger flag always starts false
    trigger = False

    # Put the job in the AGV agent pool queue and link the winning machines with the jobs
    for ii, vv in enumerate(best_job):

        machine_loc = (best_bid[ii], currentWC - 1)
        jobs[vv].cfp_wc_ma_result = machine_loc

        if not jobs[vv].job_destination_set.triggered:
            jobs[vv].job_destination_set.succeed()

        put_job_in_ma_queue(currentWC, best_bid[ii], jobs[vv], job_shop, machine)

        if not jobs[vv].agv_requested:
            # Put job in APA
            AGVstore = job_shop.AGVstoreWC[currentWC - 1]
            AGVstore.put(jobs[vv])
            trigger = True



    # If one of the enumerated jobs is triggered, trigger the APA
    if trigger:

        # Trigger the APA that there is a Job
        if not job_shop.condition_flag_CFP_AGV[currentWC - 1].triggered:
            job_shop.condition_flag_CFP_AGV[currentWC - 1].succeed()

    # Remove job from queue of the JPA
    for ii in reversed(best_job):
        yield store.get(lambda mm: mm == jobs[ii])


def bid_calculation_agv(bid_weights, agvnumber, normalization, agv, job, job_shop, processing_time,
                        job_priority, total_rp, due_date, now, queue_distance):
    """Calulcates the bidding value of a job for AGVS."""


    attribute = [0] * noAttributesAGV
    attribute[0] = (queue_distance / 25) * \
                   bid_weights[sum(machinesPerWC) + agvnumber - 1][0]  # Total distance AGV queue
    attribute[1] = processing_time / max_processing_time * bid_weights[sum(machinesPerWC) + agvnumber - 1][
        1]  # Processing time
    attribute[2] = (job_priority - 1) / (max_job_priority - 1) * bid_weights[sum(machinesPerWC) + agvnumber - 1][2]  # Job Priority
    attribute[3] = len(agv[0].items) / 25 * bid_weights[sum(machinesPerWC) + agvnumber - 1][3]  # AGV Queue length
    attribute[4] = total_rp / max_remaining_processing_time * bid_weights[sum(machinesPerWC) + agvnumber - 1][
        4]  # Remaining processing time
    attribute[5] = (due_date - now - normalization[2]) / (normalization[3] - normalization[2]) * \
                   bid_weights[sum(machinesPerWC) + agvnumber - 1][5]  # Due date
    attribute[6] = 0

    return sum(attribute)


def bid_calculation_ma(bid_weights, machinenumber, processing_time,
                       current, total_rp, due_date, now, job_priority, queue_length, total_pt_queue,
                       normalization, number_operations):
    """Calulcates the bidding value of a job for MAs."""



    attribute = [0] * noAttributesMA
    attribute[0] = processing_time / max_processing_time * bid_weights[machinenumber - 1][0]  # processing time
    attribute[1] = (current - 1) / (number_operations - 1) * bid_weights[machinenumber - 1][1]  # remaing operations
    attribute[2] = (due_date - now - normalization[0]) / (normalization[1] - normalization[0]) * \
                   bid_weights[machinenumber - 1][2]  # slack
    attribute[3] = total_rp / max_remaining_processing_time * bid_weights[machinenumber - 1][3]  # remaining processing time
    attribute[4] = (((due_date - now) / total_rp) - normalization[2]) / (normalization[3] - normalization[2]) * \
                   bid_weights[machinenumber - 1][4]  # Critical Ratio
    attribute[5] = (job_priority - 1) / (max_job_priority - 1) * bid_weights[machinenumber - 1][5]  # Job Priority
    attribute[6] = queue_length / 25 * bid_weights[machinenumber - 1][6]  # Queue length
    attribute[7] = (total_pt_queue - normalization[4]) / (normalization[5] - normalization[4]) * \
                   bid_weights[machinenumber - 1][7]  # Total processing time queue
    attribute[8] = 0

    return sum(attribute)


def queue_total_distance(agv, job_shop):
    distance = 0

    # TODO: Make shorter
    if len(agv[0].items) == 0:
        pass
    else:

        for job in agv[0].items:
            distance += job_shop.travel_time_matrix[agv[1]][job.location]

    return distance


def total_processing_time_ma_queue(jobs):
    total_pt_queue = 0

    for job in jobs:
        total_pt_queue += job.processingTime[job.currentOperation - 1]

    return total_pt_queue


def remain_processing_time(job):
    """Calculate the remaining processing time."""
    total_rp = 0

    for ii in range(job.currentOperation - 1, job.numberOfOperations):
        total_rp += job.processingTime[ii]

    return total_rp


def next_workstation(job, job_shop, env, min_job, max_job, max_wip):
    """Used to send a job to the next workstation or to complete the job.
    If a job has finished all of its operation, the relevant information (tardiness, flowtime)
    is stored. It is also checked if 2000 jobs have finished process, or if the max wip/time
    is exceded. In this, the end_event is triggered and the simulation is stopped."""
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]

        store = job_shop.MAstoreWC[nextWC - 1]
        store.put(job)

    else:

        currentWC = operationOrder[job.type - 1][job.currentOperation - 1]

        # Set job destination to depot
        job.cfp_wc_ma_result = "depot"

        if not job.job_destination_set.triggered:
            job.job_destination_set.succeed()

        if not job.agv_requested:

            # Put job in APA
            AGVstore = job_shop.AGVstoreWC[currentWC - 1]
            AGVstore.put(job)

            # Trigger the APA that there is a Job
            if not job_shop.condition_flag_CFP_AGV[currentWC - 1].triggered:
                job_shop.condition_flag_CFP_AGV[currentWC - 1].succeed()

        finish_time = env.now
        job_shop.totalWIP.append(job_shop.WIP)
        job_shop.tardiness[job.number] = max(job.priority * (finish_time - job.dueDate[job.numberOfOperations]), 0)

        job_shop.WIP -= 1

        # print(job.number ,job_shop.WIP, env.now)

        job_shop.priority[job.number] = job.priority
        job_shop.flowtime[job.number] = finish_time - job.dueDate[0]

        if job.number > max_job:
            if np.count_nonzero(job_shop.flowtime[min_job:max_job]) == 2000:
                job_shop.finish_time = env.now
                job_shop.end_event.succeed()

        if (job_shop.WIP > max_wip) | (env.now > maxTime):

            """if job_shop.WIP > max_wip:
                print("To much WIP")
            elif env.now > 10_000:
                print("Time eslaped")
            else:
                print("bad simulation")"""

            job_shop.end_event.succeed()
            job_shop.early_termination = 1
            job_shop.finish_time = env.now


def set_makespan(current_makespan, job, env, setup_time):
    """Sets the makespan of a machine"""
    add = current_makespan + job.processingTime[job.currentOperation - 1] + setup_time
    new = env.now + job.processingTime[job.currentOperation - 1] + setup_time

    return max(add, new)


def put_job_in_agv_queue(currentWC, choice, job, job_shop, agvs):
    """Puts a job in an AGV queue. Also checks if the AGV is currently active
        or has a job in its queue. If not, it succeeds an event to tell the AGV
        that a new job has been added to the queue."""

    agvs[(choice, currentWC - 1)][0].put(job)

    if not job_shop.condition_flag_agv[(choice, currentWC - 1)].triggered:
        job_shop.condition_flag_agv[(choice, currentWC - 1)].succeed()


def put_job_in_ma_queue(currentWC, choice, job, job_shop, machines):
    """Puts a job in a machine queue. Also checks if the machine is currently active
    or has a job in its queue. If not, it succeeds an event to tell the machine
    that a new job has been added to the queue."""
    machines[(choice, currentWC - 1)].put(job)

    if not job_shop.condition_flag_ma[(choice, currentWC - 1)].triggered:
        job_shop.condition_flag_ma[(choice, currentWC - 1)].succeed()


def choose_job_queue_ma(job_weights, machinenumber, processing_time, due_date, env,
                        setup_time, job_priority, normalization, job_present):
    """Calculates prioirities of jobs in a machines queue"""


    attribute_job = [0] * noAttributesJobMA

    attribute_job[0] = (due_date - processing_time - setup_time - env.now - normalization[6]) / (
            normalization[7] - normalization[6]) * \
                       job_weights[machinenumber - 1][noAttributesMA]
    attribute_job[1] = (job_priority - 1) / (max_job_priority - 1) * job_weights[machinenumber - 1][noAttributesMA + 1]
    attribute_job[2] = setup_time / max_setup_time * job_weights[machinenumber - 1][noAttributesMA + 2]
    attribute_job[3] = job_present * job_weights[machinenumber - 1][noAttributesMA + 3]

    return sum(attribute_job)


def choose_job_queue_agv(job_weights, job, normalization, agv, agvnumber, env, due_date, job_priority, job_shop, remaining_pt, JAFAMT):
    """Calculates prioirities of jobs in an agv queue"""

    if job.location[0] == "d":
        job_location = "d"
    else:
        job_location = job.location[1]

    attribute_job = [0] * noAttributesJobAGV
    attribute_job[0] = (job_priority - 1) / (max_job_priority - 1) * job_weights[sum(machinesPerWC) + agvnumber - 1][
        noAttributesAGV]  # Job priority
    attribute_job[1] = (job_shop.travel_time_matrix[agv[1]][job.location] / 1.5) * \
                       job_weights[sum(machinesPerWC) + agvnumber - 1][
                           noAttributesAGV + 1]  # Job travel distance
    attribute_job[2] = (job_location_set[job_location] / len(job_location_set)) * \
                       job_weights[sum(machinesPerWC) + agvnumber - 1][
                           noAttributesAGV + 2]  # Job location
    attribute_job[3] = ((due_date - env.now - normalization[4]) / (normalization[5] - normalization[4])) * \
                       job_weights[sum(machinesPerWC) + agvnumber - 1][
                           noAttributesAGV + 3]  # Due date
    attribute_job[4] = (remaining_pt / (JAFAMT + 0.00000001)) * job_weights[sum(machinesPerWC) + agvnumber - 1][
        noAttributesAGV + 4]  # remaining_pt

    attribute_job[5] = 0

    return sum(attribute_job)


def agv_processing(job_shop, currentWC, agv_number, env, agv, normalization, agv_buf,
                   agv_number_WC, JAFAMT):
    """This refers to an AGV Agent in the system. It checks which jobs it wants to transfer
        next to machines and stores relevant information regarding it."""

    while True:

        relative_agv = agv_number_WC[currentWC - 1].index(agv_number)

        if agv[0].items:

            agv_location = agv[1]
            priority_list = []

            if FIFO_agv_queue:
                next_job = agv[0].items[0]

            else:
                for job in agv[0].items:

                    if job.finished_job:
                        remaining_pt = 0
                    elif job.currentOperation == 1:
                        remaining_pt = 0
                    else:
                        remaining_pt = job.actual_finish_proc_time - env.now

                    job_queue_priority = choose_job_queue_agv(job_shop.weights, job, normalization, agv, agv_number,
                                                              env,
                                                              job.dueDate[job.currentOperation],
                                                              job.priority, job_shop, remaining_pt, JAFAMT)  # Calulate the job priorities
                    priority_list.append(job_queue_priority)

                ind_processing_job = priority_list.index(max(priority_list))  # Get the job with the highest value

                # Remember job and job destination
                next_job = agv[0].items[ind_processing_job]

            if agv_location != next_job.location:
                driving_time = job_shop.travel_time_matrix[agv_location][next_job.location]
                yield env.timeout(driving_time)

                # Change AGV location
                agv[1] = next_job.location
                agv_location = agv[1]

            #  Load the job on the AGV and remove job from depot or machine
            if next_job in job_shop.depot_queue.items:
                job_shop.depot_queue.items.remove(next_job)

            elif next_job in job_shop.machine_buffer_per_wc[agv_location][0].items:

                yield next_job.job_in_progress

                next_job.average_waiting_time_pickup = (next_job.average_waiting_time_pickup + (
                        env.now - next_job.finishing_time_machine)) / next_job.currentOperation

                job_shop.machine_buffer_per_wc[agv_location][0].items.remove(next_job)

            # Always reset the job in progress condition
            next_job.job_in_progress = simpy.Event(env)

            # Put job on AGV buffer
            agv_buf.put(next_job)

            job_destination = next_job.cfp_wc_ma_result

            if job_destination is None:
                yield next_job.job_destination_set

                job_destination = next_job.cfp_wc_ma_result

            # Always reset the job destination set condition
            next_job.job_destination_set = simpy.Event(env)

            if not job_destination == "depot":
                job_shop.machine_buffer_per_wc[job_destination][1].put(next_job)

            driving_time = job_shop.travel_time_matrix[agv_location][job_destination]
            yield env.timeout(driving_time)

            # Change job and AGV location
            next_job.location = job_destination
            agv[1] = next_job.location
            agv_location = next_job.location

            #  Unload the job from the AGV and put in machine or machine buffer
            agv[0].items.remove(next_job)
            agv_buf.items.remove(next_job)

            if not agv_location == "depot":

                machine_buf = job_shop.machine_buffer_per_wc[agv_location][1]
                machine_buf.items.remove(next_job)
                machine_buf = job_shop.machine_buffer_per_wc[agv_location][0]
                machine_buf.put(next_job)

                #  Unload the job from the AGV and put job on machine, machine buffer or depot
                if not next_job.job_loaded_condition.triggered and next_job in machine_buf.items:
                    next_job.job_loaded_condition.succeed()

            else:
                job_shop.depot_queue.put(next_job)

            # Trigger all APA that there is an idle AGV
            for wc in noOfWC:

                if not job_shop.condition_flag_CFP_AGV[wc].triggered:
                    job_shop.condition_flag_CFP_AGV[wc].succeed()

        else:
            yield job_shop.condition_flag_agv[
                (relative_agv, currentWC - 1)]  # Used if there is currently no job in the agv queue
            job_shop.condition_flag_agv[(relative_agv, currentWC - 1)] = simpy.Event(env)  # Reset event if it is used


def machine_processing(job_shop, currentWC, machine_number, env, last_job, machine,
                       makespan, min_job, max_job, normalization, max_wip, machine_buf, JAFAMT):
    """This refers to a Machine Agent in the system. It checks which jobs it wants to process
    next and stores relevant information regarding it."""
    while True:

        relative_machine = machine_number_WC[currentWC - 1].index(machine_number)

        if machine.items:
            setup_time = []
            priority_list = []
            if not last_job[relative_machine]:  # Only for the first job
                ind_processing_job = 0
                setup_time.append(0)
            else:
                for job in machine.items:

                    if job in machine_buf[0].items:
                        job_present = 1
                    else:
                        job_present = 0

                    setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
                    job_queue_priority = choose_job_queue_ma(job_shop.weights, machine_number,
                                                             job.processingTime[job.currentOperation - 1],
                                                             job.dueDate[job.currentOperation], env, setuptime,
                                                             job.priority, normalization,
                                                             job_present)  # Calulate the job priorities
                    priority_list.append(job_queue_priority)
                    setup_time.append(setuptime)
                ind_processing_job = priority_list.index(max(priority_list))  # Get the job with the highest value

            next_job = machine.items[ind_processing_job]

            # If job is not at machine, machine will be yielded until AGVs loads job
            if next_job not in machine_buf[0].items:
                yield next_job.job_loaded_condition

            # Always reset the job loaded condition
            next_job.job_loaded_condition = simpy.Event(env)

            setuptime = setup_time[ind_processing_job]
            time_in_processing = next_job.processingTime[
                                     next_job.currentOperation - 1] + setuptime  # Total time the machine needs to process the job

            next_job.finished_job = False
            next_job.actual_finish_proc_time = time_in_processing + env.now

            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, env, setuptime)

            last_job[relative_machine] = next_job.type

            machine.items.remove(next_job)  # Remove job from queue

            next_job.cfp_wc_ma_result = None
            next_job.agv_requested = False


            # May not be higher than 2.0!!!!!!
            request_earlier_AGV_time = JAFAMT

            yield env.timeout(time_in_processing - request_earlier_AGV_time)

            if request_earlier_AGV_time != 0:
                next_job.agv_requested = True

                if next_job.currentOperation == next_job.numberOfOperations:
                    selected_WC = currentWC
                else:

                    selected_WC = operationOrder[next_job.type - 1][(next_job.currentOperation + 1) - 1]

                # Put job in APA
                AGVstore = job_shop.AGVstoreWC[selected_WC - 1]
                AGVstore.put(next_job)



                # Trigger the APA that there is a Job
                if not job_shop.condition_flag_CFP_AGV[selected_WC - 1].triggered:
                    job_shop.condition_flag_CFP_AGV[selected_WC - 1].succeed()

            yield env.timeout(request_earlier_AGV_time)





            next_job.finishing_time_machine = env.now

            next_job.finished_job = True



            if not next_job.job_in_progress.triggered:
                next_job.job_in_progress.succeed()



            next_workstation(next_job, job_shop, env, min_job, max_job, max_wip)  # Send the job to the next workstation

        else:
            yield job_shop.condition_flag_ma[
                (relative_machine, currentWC - 1)]  # Used if there is currently no job in the machines queue
            job_shop.condition_flag_ma[(relative_machine, currentWC - 1)] = simpy.Event(
                env)  # Reset event if it is used


def cfp_wc_ma(env, machine, store, job_shop, currentWC, normalization):
    """Sends out the Call-For-Proposals to the various machines.
    Represents the Job-Pool_agent"""

    while True:

        if store.items:
            job_shop.QueuesWC[currentWC - 1].append(
                {ii: len(job_shop.machine_queue_per_wc[(ii, currentWC - 1)].items) for ii in
                 range(machinesPerWC[currentWC - 1])})  # Stores the Queue length of the JPA

            c = bid_winner_ma(env, store.items, machinesPerWC[currentWC - 1], currentWC, job_shop,
                              machine, store, normalization)

            env.process(c)

        tib = 0.5  # Frequency of when CFPs are sent out
        yield env.timeout(tib)


def cfp_wc_agv(env, agvs, AGVstore, job_shop, currentWC, normalization, dispatch_rule_no, immediate_release, agvsPerWC,
               agv_number_WC):
    """Sends out the Call-For-Proposals to the various AGVs when in bidding control.
        Uses dispatching rules when in dispatch control
        Represents the AGV-Pool_agent"""

    while True:

        if AGVstore.items:

            job_list = AGVstore.items

            job_shop.AGVQueuesWC[currentWC - 1].append(
                {ii: len(job_shop.agv_queue_per_wc[(ii, currentWC - 1)][0].items) for ii in
                 range(agvsPerWC[currentWC - 1])})

            # Bidding control
            if dispatch_rule_no == 1:
                c = bid_winner_agv_per_WC(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop,
                                          agvs, AGVstore, normalization, agv_number_WC)

                env.process(c)

            # Bidding control - No AGV dedicated to WC
            if dispatch_rule_no == 2:
                c = bid_winner_agv_all_WC(env, job_list, agvsPerWC, currentWC, job_shop,
                                          agvs, AGVstore, normalization, agv_number_WC)

                env.process(c)

            # Dispatch control
            if dispatch_rule_no > 2:
                c = dispatch_control(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop, agvs, AGVstore,
                                     dispatch_rule_no, agvsPerWC, agv_number_WC)
                env.process(c)

        if immediate_release:
        #if immediate_release and not agvsPerWC[currentWC - 1] == 0:

            yield job_shop.condition_flag_CFP_AGV[currentWC - 1]  # Used if there is currently no job in the APA queue

            job_shop.condition_flag_CFP_AGV[currentWC - 1] = simpy.Event(env)  # Reset event if it is used

        else:
            tib = 0.5  # Frequency of when CFPs are sent out
            yield env.timeout(tib)


def source(env, number1, interval, job_shop, due_date_setting, min_job):
    """Reflects the Job Release Agent. Samples time and then "releases" a new
    job into the system."""
    while True:  # Needed for infinite case as True refers to "until".

        ii = number1

        number1 += 1
        job = New_Job('job%02d' % ii, env, ii, due_date_setting)

        if ii == min_job:
            job_shop.start_time = env.now  # Start counting when the minimum number of jobs have entered the system
        job_shop.tardiness.append(-1)
        job_shop.flowtime.append(0)
        job_shop.priority.append(0)
        job_shop.WIP += 1
        firstWC = operationOrder[job.type - 1][0]

        # Put job in depot
        depot = job_shop.depot_queue
        depot.put(job)

        # Put job in JPA
        store = job_shop.MAstoreWC[firstWC - 1]
        store.put(job)

        tib = random.expovariate(1.0 / interval)
        yield env.timeout(tib)


# %% Main Simulation function

def do_simulation_with_weights(mean_weight_new, std_weight_new, arrivalMean, due_date_tightness,
                               norm_range, norm_range_agv, min_job, max_job, wip_max, AGV_rule,
                               created_travel_time_matrix,
                               immediate_release_bool, JAFAMT_value, iii):
    eta_new = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))
    objective_new = np.zeros(2)
    mean_tard = np.zeros(2)
    max_tard = np.zeros(2)
    test_weights_pos = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))
    test_weights_min = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))

    for mm in range(sum(machinesPerWC)):
        for jj in range(totalAttributes):
            if (jj >= min(totalAttributes, noAttributesMA + noAttributesJobMA)) | (jj == noAttributesMA - 1):
                eta_new[mm][jj] = 0
                test_weights_pos[mm][jj] = 0
                test_weights_min[mm][jj] = 0
            else:
                eta_new[mm][jj] = random.gauss(0, np.exp(std_weight_new[mm][jj]))
                test_weights_pos[mm][jj] = mean_weight_new[mm][jj] + (eta_new[mm][jj])
                test_weights_min[mm][jj] = mean_weight_new[mm][jj] - (eta_new[mm][jj])

    for mm in range(sum(machinesPerWC), sum(machinesPerWC) + sum(agvsPerWC)):
        for jj in range(totalAttributes):
            if (jj >= min(totalAttributes, noAttributesAGV + noAttributesJobAGV)) | (jj == noAttributesAGV - 1) | (
                    jj == noAttributesAGV + noAttributesJobAGV - 1):
                eta_new[mm][jj] = 0
                test_weights_pos[mm][jj] = 0
                test_weights_min[mm][jj] = 0
            else:

                if AGV_rule <= 2:

                    if jj == 11 and JAFAMT_value == 0:
                        eta_new[mm][jj] = 0
                        test_weights_pos[mm][jj] = 0
                        test_weights_min[mm][jj] = 0
                    else:
                        eta_new[mm][jj] = random.gauss(0, np.exp(std_weight_new[mm][jj]))
                        test_weights_pos[mm][jj] = mean_weight_new[mm][jj] + (eta_new[mm][jj])
                        test_weights_min[mm][jj] = mean_weight_new[mm][jj] - (eta_new[mm][jj])

                else:
                    eta_new[mm][jj] = 0
                    test_weights_pos[mm][jj] = 0
                    test_weights_min[mm][jj] = 0


    env = Environment()
    job_shop = jobShop(env, test_weights_pos, created_travel_time_matrix, agvsPerWC,
                       agv_number_WC)  # Initiate the job shop
    env.process(source(env, number, arrivalMean, job_shop, due_date_tightness, min_job))

    for wc in range(len(machinesPerWC)):

        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        MAstoreWC = job_shop.MAstoreWC[wc]
        AGVstoreWC = job_shop.AGVstoreWC[wc]

        env.process(
            cfp_wc_ma(env, job_shop.machine_queue_per_wc, MAstoreWC, job_shop, wc + 1, norm_range))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_queue_per_wc[(ii, wc)]
            machine_buf = job_shop.machine_buffer_per_wc[(ii, wc)]

            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, last_job,
                                   machine, makespanWC, min_job, max_job, norm_range, wip_max,
                                   machine_buf, JAFAMT_value))

        env.process(
            cfp_wc_agv(env, job_shop.agv_queue_per_wc, AGVstoreWC, job_shop, wc + 1, norm_range_agv, AGV_rule,
                       immediate_release_bool, agvsPerWC, agv_number_WC))

        for ii in range(agvsPerWC[wc]):
            agv = job_shop.agv_queue_per_wc[(ii, wc)]
            agv_buf = job_shop.agv_buffer_per_wc[(ii, wc)]

            env.process(agv_processing(job_shop, wc + 1, agv_number_WC[wc][ii], env,
                                       agv, norm_range_agv, agv_buf, agv_number_WC, JAFAMT_value))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)

    if job_shop.early_termination == 1:

        if math.isnan(np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job]))):
            objective_new[0] = 20_000
            max_tard[0] = 1000
        else:
            objective_new[0] = np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job])) + maxTime - np.count_nonzero(
                job_shop.flowtime[min_job:max_job]) + 0.01 * max(job_shop.tardiness[min_job:max_job])

            max_tard[0] = np.nanmax(job_shop.tardiness[min_job:max_job])
    else:
        objective_new[0] = np.nanmean(job_shop.tardiness[min_job:max_job]) + 0.01 * max(
            job_shop.tardiness[min_job:max_job])
        max_tard[0] = np.nanmax(job_shop.tardiness[min_job:max_job])

    mean_tard[0] = np.nanmean(job_shop.tardiness[min_job:max_job])

    env = Environment()
    job_shop = jobShop(env, test_weights_min, created_travel_time_matrix, agvsPerWC,
                       agv_number_WC)  # Initiate the job shop
    env.process(source(env, number, arrivalMean, job_shop, due_date_tightness, min_job))

    for wc in range(len(machinesPerWC)):

        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        MAstoreWC = job_shop.MAstoreWC[wc]
        AGVstoreWC = job_shop.AGVstoreWC[wc]
        env.process(
            cfp_wc_ma(env, job_shop.machine_queue_per_wc, MAstoreWC, job_shop, wc + 1, norm_range))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_queue_per_wc[(ii, wc)]
            machine_buf = job_shop.machine_buffer_per_wc[(ii, wc)]
            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, last_job,
                                   machine, makespanWC, min_job, max_job, norm_range, wip_max,
                                   machine_buf, JAFAMT_value))

        env.process(
            cfp_wc_agv(env, job_shop.agv_queue_per_wc, AGVstoreWC, job_shop, wc + 1, norm_range_agv, AGV_rule,
                       immediate_release_bool, agvsPerWC, agv_number_WC))

        for ii in range(agvsPerWC[wc]):
            agv = job_shop.agv_queue_per_wc[(ii, wc)]
            agv_buf = job_shop.agv_buffer_per_wc[(ii, wc)]
            env.process(agv_processing(job_shop, wc + 1, agv_number_WC[wc][ii], env,
                                       agv, norm_range_agv, agv_buf, agv_number_WC, JAFAMT_value))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)

    if job_shop.early_termination == 1:

        if math.isnan(np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job]))):
            objective_new[1] = 20000
            max_tard[1] = 1000
        else:
            objective_new[1] = np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job])) + maxTime - np.count_nonzero(
                job_shop.flowtime[min_job:max_job]) + 0.01 * max(job_shop.tardiness[min_job:max_job])

            max_tard[1] = np.nanmax(job_shop.tardiness[min_job:max_job])
    else:
        objective_new[1] = np.nanmean(job_shop.tardiness[min_job:max_job]) + 0.01 * max(
            job_shop.tardiness[min_job:max_job])
        max_tard[1] = np.nanmax(job_shop.tardiness[min_job:max_job])

    mean_tard[1] = np.nanmean(job_shop.tardiness[min_job:max_job])

    return objective_new, eta_new, mean_tard, max_tard


def run_linear(filename1, filename2, arrival_time_mean, due_date_k, alpha, norm_range,
               norm_range_agv, min_job,
               max_job, wip_max, AGV_rule, created_travel_time_matrix, immediate_release_bool,
               JAFAMT_value):

    file1 = open(filename1, "w")
    mean_weight = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))
    std_weight = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))

    for m in range(sum(machinesPerWC)):
        for j in range(totalAttributes):
            if (j >= min(totalAttributes, noAttributesMA + noAttributesJobMA)) | (j == noAttributesMA - 1):
                std_weight[m][j] = 0
            else:
                std_weight[m][j] = std_weight[m][j] + np.log(0.3)

    for m in range(sum(machinesPerWC), sum(machinesPerWC) + sum(agvsPerWC)):
        for j in range(totalAttributes):
            if (j >= min(totalAttributes, noAttributesAGV + noAttributesJobAGV)) | (j == noAttributesAGV - 1) | (
                    j == noAttributesAGV + noAttributesJobAGV - 1):
                std_weight[m][j] = 0

            elif j == 11 and JAFAMT_value == 0:
                std_weight[m][j] = 0

            else:
                if AGV_rule <= 2:
                    std_weight[m][j] = std_weight[m][j] + np.log(0.3)
                else:
                    std_weight[m][j] = 0

    # population_size = noAttributesMA + noAttributesJobMA + noAttributesAGV + noAttributesJobAGV
    population_size = 16

    for i in range(sum(machinesPerWC)):
        mean_weight[i][6] = -3
        mean_weight[i][noAttributesMA] = -1
        mean_weight[i][noAttributesMA + 2] = -3

    if AGV_rule <= 2:
        for i in range(sum(machinesPerWC), sum(machinesPerWC) + sum(agvsPerWC)):
            mean_weight[i][3] = -3
            mean_weight[i][noAttributesAGV + 3] = -1

    jobshop_pool = Pool(processes=1)
    alpha_mean = 0.1
    alpha_std = 0.025
    beta_1 = 0.9
    beta_2 = 0.999
    m_t_mean = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))
    v_t_mean = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))

    m_t_std = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))
    v_t_std = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))

    objective = np.zeros((population_size, 2))

    eta = np.zeros((population_size, sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))
    mean_tardiness = np.zeros((population_size, 2))
    max_tardiness = np.zeros((population_size, 2))

    best_weights = mean_weight
    best_sim_number = 0
    old_tardiness = 999999
    for num_sim in range(no_generation):

        now = datetime.now()

        seeds = range(int(population_size))

        func1 = partial(do_simulation_with_weights, mean_weight, std_weight, arrival_time_mean, due_date_k, norm_range,
                        norm_range_agv, min_job, max_job, wip_max, AGV_rule,
                        created_travel_time_matrix,
                        immediate_release_bool, JAFAMT_value)

        makespan_per_seed = jobshop_pool.map(func1, seeds)

        for h, j in zip(range(int(population_size)), seeds):
            objective[j] = makespan_per_seed[h][0]
            eta[j] = makespan_per_seed[h][1]
            mean_tardiness[j] = makespan_per_seed[h][2]
            max_tardiness[j] = makespan_per_seed[h][3]

        objective_norm = np.zeros((population_size, 2))

        # Normalise the current populations performance
        for ii in range(population_size):
            objective_norm[ii][0] = (objective[ii][0] - np.mean(objective, axis=0)[0]) / np.std(objective, axis=0)[0]
            objective_norm[ii][1] = (objective[ii][1] - np.mean(objective, axis=0)[1]) / np.std(objective, axis=0)[1]

        delta_mean_final = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))
        delta_std_final = np.zeros((sum(machinesPerWC) + sum(agvsPerWC), totalAttributes))

        for m in range(sum(machinesPerWC) + sum(agvsPerWC)):
            for j in range(totalAttributes):
                delta_mean = 0
                delta_std = 0
                for ii in range(population_size):
                    delta_mean += ((objective_norm[ii][0] - objective_norm[ii][1]) / 2) * eta[ii][m][j] / np.exp(
                        std_weight[m][j])

                    delta_std += ((objective_norm[ii][0] + objective_norm[ii][1]) / 2) * (eta[ii][m][j] ** 2 - np.exp(
                        std_weight[m][j])) / (np.exp(std_weight[m][j]))

                delta_mean_final[m][j] = delta_mean / population_size
                delta_std_final[m][j] = delta_std / population_size

        # print(delta_std_final)
        t = num_sim + 1
        for m in range(sum(machinesPerWC) + sum(agvsPerWC)):
            for j in range(totalAttributes):
                m_t_mean[m][j] = (beta_1 * m_t_mean[m][j] + (1 - beta_1) * delta_mean_final[m][j])
                v_t_mean[m][j] = (beta_2 * v_t_mean[m][j] + (1 - beta_2) * delta_mean_final[m][j] ** 2)
                m_hat_t = (m_t_mean[m][j] / (1 - beta_1 ** t))
                v_hat_t = (v_t_mean[m][j] / (1 - beta_2 ** t))
                mean_weight[m][j] = mean_weight[m][j] - (alpha_mean * m_hat_t) / (np.sqrt(v_hat_t) + 10 ** -8)

                m_t_std[m][j] = (beta_1 * m_t_std[m][j] + (1 - beta_1) * delta_std_final[m][j])
                v_t_std[m][j] = (beta_2 * v_t_std[m][j] + (1 - beta_2) * delta_std_final[m][j] ** 2)
                m_hat_t_1 = (m_t_std[m][j] / (1 - beta_1 ** t))
                v_hat_t_1 = (v_t_std[m][j] / (1 - beta_2 ** t))
                std_weight[m][j] = std_weight[m][j] - (alpha_std * m_hat_t_1) / (np.sqrt(v_hat_t_1) + 10 ** -8)

        alpha_mean = 0.1 * np.exp(-(t - 1) / alpha)
        alpha_std = 0.025 * np.exp(-(t - 1) / alpha)

        objective1 = np.array(objective)

        # print(num_sim, objective1[objective1 < 5000].mean(), np.mean(np.mean(np.exp(std_weight))))
        print("Sim-number:", num_sim, "- Mean Tardiness", round(np.nanmean(mean_tardiness), 3), "- Max Tardiness:",
              round(np.nanmean(max_tardiness), 3), "- Objective:", round(np.mean(np.mean(objective, axis=0)), 3),
              "- Sim-time gen:", datetime.now() - now)

        # print(np.mean(np.exp(std_weight), axis=0))
        if SAVE:
            L = [str(num_sim) + " ", str(np.mean(np.mean(objective, axis=0))) + " ",
                 str(np.mean(np.exp(std_weight), axis=0)) + "\n"]
            file1.writelines(L)

        if np.nanmean(mean_tardiness) < old_tardiness and num_sim > (no_generation / 20):
            old_tardiness = np.nanmean(mean_tardiness)
            best_weights = mean_weight
            best_sim_number = num_sim

            if SAVE:
                file2 = open(filename2 + ".csv", 'w')
                writer = csv.writer(file2)
                writer.writerows(best_weights)
                file2.close()
            # print("Best intermediate simulation generation:", best_sim_number)

        if num_sim == no_generation - 1:

            if SAVE:
                file2 = open(filename2 + ".csv", 'w')
                writer = csv.writer(file2)
                writer.writerows(mean_weight)
                file2.close()
            print("Best simulation generation:", best_sim_number)


# %% Classes

class jobShop:
    """This class creates a job shop, along with everything that is needed to run the Simpy Environment."""

    def __init__(self, env, weights_new, travel_time_matrix, agvsPerWC, agv_number_WC):
        # TODO: Can we bundle the Machine queue, Machine buffer and Machine resource?
        # Virtual machine queue, phyical machine buffer capacity + jobs underway and machine resource
        self.machine_queue_per_wc = {(ii, jj): Store(env) for jj in noOfWC for ii in range(machinesPerWC[jj])}

        self.machine_buffer_per_wc = {(ii, jj): [Store(env, capacity=9999),
                                                 Store(env)] for jj in noOfWC for ii in
                                      range(machinesPerWC[jj])}

        # TODO: Can we bundle the AGV queue, AGV buffer and AGV resource?
        # Virtual agv queue and phyical resource + location
        self.agv_queue_per_wc = {(ii, jj): [Store(env), "depot"] for jj in noOfWC for ii in range(agvsPerWC[jj])}

        self.agv_buffer_per_wc = {(ii, jj): Store(env, capacity=1) for jj in noOfWC for ii in
                                  range(agvsPerWC[jj])}

        # Virtual queues JPA & APA
        self.MAstoreWC = {ii: FilterStore(env) for ii in noOfWC}  # Used to store jobs in the JPA
        self.AGVstoreWC = {ii: FilterStore(env) for ii in noOfWC}  # Used to store jobs in the APA

        # Travel distance matrix for calculating distances
        self.travel_time_matrix = travel_time_matrix

        # All the jobs objects at the depot
        self.depot_queue = Store(env)

        # Queue lengths
        self.QueuesWC = {jj: [] for jj in noOfWC}  # Can be used to keep track of Queue Lenghts JPA
        self.AGVQueuesWC = {jj: [] for jj in noOfWC}  # Can be used to keep track of Queue Lenghts APA

        self.scheduleWC = {ii: [] for ii in noOfWC}  # Used to keep track of the schedule
        self.makespanWC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                           noOfWC}  # Keeps track of the makespan of each machine
        self.last_job_WC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                            noOfWC}  # Keeps track of which job was last in the machine

        self.condition_flag_CFP_AGV = {ii: simpy.Event(env) for ii in noOfWC}
        # An event which keeps track if the APA has a job in the queue

        self.condition_flag_ma = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(machinesPerWC[jj])}
        # An event which keeps track if a machine has had a job inserted into it if it previously had no job

        self.condition_flag_agv = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(agvsPerWC[jj])}
        # An event which keeps track if an agv has had a job inserted into it if it previously had no job

        self.weights = weights_new
        self.flowtime = []  # Jobs flowtime
        self.tardiness = []  # Jobs tardiness
        self.makespan = []
        self.WIP = 0  # Current WIP of the system
        self.early_termination = 0  # Whether the simulation terminated earlier
        self.utilization = {(ii, jj): 0 for jj in noOfWC for ii in range(machinesPerWC[jj])}
        self.finish_time = 0  # Finishing time of the system
        self.totalWIP = []  # Keeps track of the total WIP of the system

        self.bids = []  # Keeps track of the bids
        self.priority = []  # Keeps track of the job priorities
        self.start_time = 0  # Starting time of the simulation

        self.gantt_list_ma = []  # List with machine job information for gantt chart
        self.gantt_list_agv = []  # List with agv job information for gantt chart

        self.QueuesMAs = {"MA" + str(ii): [0] for jj in noOfWC for ii in machine_number_WC[jj]}
        self.QueueTimes = [0]

        self.AGV_total_driving_time = {"AGV" + str(ii): [0] for jj in noOfWC for ii in agv_number_WC[jj]}
        self.AGV_total_waiting_time = {"AGV" + str(ii): [0] for jj in noOfWC for ii in agv_number_WC[jj]}
        self.AGV_remember_time = {"AGV" + str(ii): 0 for jj in noOfWC for ii in agv_number_WC[jj]}


class New_Job:
    """ This class is used to create a new job. It contains information
    such as processing time, due date, number of operations etc."""

    def __init__(self, name, env, number1, dueDateTightness):
        jobType = random.choices(range(len(operationOrder)), weights=demand, k=1)
        jobWeight = job_priority[jobType[0] - 1]
        self.type = jobType[0]
        self.priority = jobWeight
        self.number = number1
        self.name = name

        self.location = "depot"
        self.cfp_wc_ma_result = None
        self.agv_requested = False

        self.job_loaded_condition = simpy.Event(env)
        self.job_destination_set = simpy.Event(env)
        self.job_in_progress = simpy.Event(env)

        self.currentOperation = 1
        self.processingTime = np.zeros(numberOfOperations[self.type - 1])
        self.dueDate = np.zeros(numberOfOperations[self.type - 1] + 1)
        self.dueDate[0] = env.now

        self.actual_finish_proc_time = 0
        self.finished_job = False

        self.finishing_time_machine = 0
        self.average_waiting_time_pickup = 0

        self.arrival_time_system = env.now

        self.operationOrder = operationOrder[self.type - 1]
        self.numberOfOperations = numberOfOperations[self.type - 1]

        ddt = random.uniform(dueDateTightness, dueDateTightness + 4)
        for ii in range(self.numberOfOperations):
            meanPT = processingTimes[self.type - 1][ii]
            self.processingTime[ii] = meanPT
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * ddt


# %% Main


if __name__ == '__main__':

    simulation = [["scenario_4", False, 0.0], ["scenario_4", True, 0.0], ["scenario_4", False, 2.0], ["scenario_4", True, 2.0]]





    for sim in simulation:

        scenario = sim[0]

        max_workcenters = situations[scenario][0][0]
        min_workcenters = situations[scenario][0][1]
        no_of_jobs = situations[scenario][1]
        total_machines = situations[scenario][2]
        min_proc = situations[scenario][3][0]
        max_proc = situations[scenario][3][1]
        agvsPerWC_new = situations[scenario][4]
        max_wip = situations[scenario][5]
        maxTime = situations[scenario][6]
        seed = situations[scenario][7]
        uti = situations[scenario][8]
        setup_factor = 0.20

        processingTimes, operationOrder, machinesPerWC, setupTime, demand, job_priority, arrival_rate, machine_number_WC, CR, DDT = generate_scenario.new_scenario(
            max_workcenters, min_workcenters, no_of_jobs, total_machines, min_proc, max_proc, setup_factor, uti, seed)

        numberOfOperations = [len(i) for i in operationOrder]
        noOfWC = range(len(machinesPerWC))
        arrival_rate = [arrival_rate[0] - 0]
        created_travel_time_matrix, agvsPerWC, agv_number_WC = Travel_matrix.choose_distance_matrix(agvsPerWC_new,
                                                                                                    machinesPerWC, seed)

        # print(created_travel_time_matrix)

        max_job_priority = max(job_priority)
        max_setup_time = np.amax(setupTime)
        max_processing_time = 0
        max_remaining_processing_time = 0

        for idx, job_type in enumerate(processingTimes):
            max_pt = max(job_type)
            summed_pt = sum(processingTimes[idx])

            if max_pt > max_processing_time:
                max_processing_time = max_pt
            if summed_pt > max_remaining_processing_time:
                max_remaining_processing_time = summed_pt

        job_location_set = {"d": 1}
        for location in noOfWC:
            job_location_set[location] = location + 2






        # Simulation Parameter 1 - AGV scheduling control:
        # 1: Linear Bidding Auction - AGVs Dedicated to WC
        # 2: Linear Bidding Auction - AGVs Free For All
        # 3: Nearest Idle AGV Rule (AGV & JOB)
        # 4: Random AGV Rule (AGV) - Random Job Rule (Job)
        # 5: Longest Time In System Rule (JOB) - Minimal Distance Rule (AGV)
        # 6: Longest Waiting Time at Pickup Point (JOB) - Minimal Transfer Rule (AGV)
        # 7: Longest Average Waiting Time At Pickup Point (JOB) - Minimal Transfer Rule (AGV)
        # 8: Earliest Due Time (JOB) - Minimal Transfer Rule (AGV)
        # 9: Earliest Release Time (JOB) - Minimal Transfer Rule (AGV)
        simulation_parameter_1 = [2]

        # Simulation Parameter 2 - Job almost finished at machines trigger values
        simulation_parameter_2 = [sim[2]]

        # Simulation Parameter 4 - Direct or periodically job release APA (Direct = True)
        simulation_parameter_3 = [sim[1]]

        min_jobs = [499, 999, 1499]  # Minimum number of jobs in order te reach steady state
        max_jobs = [2499, 2999, 3499]  # Maximum number of jobs to collect information from
        wip_max = [150, max_wip, 300]  # Maximum WIP allowed in the system

        # arrival_time = [1.5429, 1.4572, 1.3804]
        arrival_time = [arrival_rate[0]]

        learning_decay_rate = [10, 100, 500, no_generation, 2500, 5000, 10000]

        utilization = [90]

        due_date_settings = [4, 4, 4]



        """normalization_MA_array = [[-112.5, 225.0, -21.0, 21.0, -112.5, 225.0, -10, 75],
                                  [-112.5, 225.0, -21.0, 21.0, -112.5, 225.0, -700, 250],
                                  [-112.5, 225.0, -21.0, 21.0, -112.5, 225.0, -10, 75]]


        normalization_AGV_array = [[-10, 25, -112.5, 225.0, -112.5, 225.0],
                                   [-10, 25, -112.5, 225.0, -250, 225.0],
                                   [-10, 25, -112.5, 225.0, -112.5, 225.0]]"""


        # ============================



        normalization_MA_array = [[-DDT*0.50, DDT, -CR, CR, -DDT*0.50, DDT, -DDT*0.50, DDT],
                                  [-DDT*0.50, DDT, -CR, CR, -DDT*0.50, DDT, -DDT*0.50, DDT],
                                  [-DDT*0.50, DDT, -CR, CR, -DDT*0.50, DDT, -DDT*0.50, DDT]]

        normalization_AGV_array = [[0, 0, -DDT*0.50, DDT, -DDT*0.50, DDT],
                                   [0, 0, -DDT*0.50, DDT, -DDT*0.50, DDT],
                                   [0, 0, -DDT*0.50, DDT, -DDT*0.50, DDT]]




        AGV_rule = simulation_parameter_1[0]
        JAFAMT_value = simulation_parameter_2[0]
        immediate_release_bool = simulation_parameter_3[0]

        print("Simulation:", "(" + str(AGV_rule) + "-" + str(JAFAMT_value) + "-" + str(immediate_release_bool) + ")")

        if AGV_rule == 2:
            sim_par_1_string = "AGV_ALL_WC"
        elif AGV_rule == 1:
            sim_par_1_string = "AGV_PER_WC"
        elif AGV_rule == 3:
            sim_par_1_string = "DISP_RULE_3"

        sim_par_2_string = "JAFAMT_" + str(JAFAMT_value) + "_" + str(immediate_release_bool)

        for util in range(len(utilization)):

            print("Current run is:" + str(utilization[util]) + "-" + str(due_date_settings[util]) + "-" + str(
                learning_decay_rate[3]))

            path = "Runs/" + "scenario_" + str(scenario) + "/" + sim_par_1_string + "/" + sim_par_2_string + "/" + str(
                agvsPerWC) + "/Attribute_Runs/" + str(utilization[util]) + "-" + str(
                due_date_settings[util])

            try:
                os.makedirs(path)
                # print("Directory ", path, " Created ")
            except FileExistsError:
                # print("Directory ", path, " already exists")
                pass

            str1 = "Runs/" + "scenario_" + str(scenario) + "/" + sim_par_1_string + "/" + sim_par_2_string + "/" + str(
                agvsPerWC) + "/Attribute_Runs/" + str(
                utilization[util]) + "-" + str(
                due_date_settings[util]) + "/Run-" + str(utilization[util]) + "-" + str(
                due_date_settings[util]) + "-" + str(str(learning_decay_rate[3])) + ".txt"

            str2 = "Runs/" + "scenario_" + str(scenario) + "/" + sim_par_1_string + "/" + sim_par_2_string + "/" + str(
                agvsPerWC) + "/Attribute_Runs/" + str(
                utilization[util]) + "-" + str(
                due_date_settings[util]) + "/Run-weights-" + str(utilization[util]) + "-" + str(
                due_date_settings[util])

            run_linear(str1, str2, arrival_time[util], due_date_settings[util], learning_decay_rate[3],
                       normalization_MA_array[util], normalization_AGV_array[util],
                       min_jobs[util + 1],
                       max_jobs[util + 1], wip_max[util + 1],
                       AGV_rule,
                       created_travel_time_matrix, immediate_release_bool, JAFAMT_value)
