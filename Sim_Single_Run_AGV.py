"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""

import itertools
import random
import time

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
maxTime = 10000.0  # Runtime limit

scenario = 1

if scenario == 1:  # ------------ Scenario 1:
    processingTimes, operationOrder, machinesPerWC, setupTime, demand, job_priority, arrival_rate, machine_number_WC, CR, DDT = generate_scenario.new_scenario(
        5, 2, 5, 16, 2, 9, 0.20, 0.90)

    numberOfOperations = [len(i) for i in operationOrder]
    noOfWC = range(len(machinesPerWC))

    agvsPerWC_new = [3, 3, 3, 3, 3]

    arrival_rate = [arrival_rate[0] - 0.10]

    created_travel_time_matrix, agvsPerWC, agv_number_WC = Travel_matrix.choose_distance_matrix(
        scenario, agvsPerWC_new)

else:  # ----------- Scenario 2:
    processingTimes, operationOrder, machinesPerWC, setupTime, demand, job_priority, arrival_rate, machine_number_WC, CR, DDT = generate_scenario.new_scenario(
        8, 4, 8, 26, 2, 9, 0.20, 0.70)
    numberOfOperations = [len(i) for i in operationOrder]
    noOfWC = range(len(machinesPerWC))

    agvsPerWC_new = [3, 3, 3, 3, 3, 3, 3, 3]

    created_travel_time_matrix, agvsPerWC, agv_number_WC = Travel_matrix.choose_distance_matrix(
        scenario, agvsPerWC_new)

# Virtual Machine QUEUE plotting
QUEUE = False

# DEBUG PRINTING
DEBUG = False

# PLOTTING
GANTT_Machine = False
GANTT_AGV = False
AGV_ROUTING = False

# GANTT GLOBAL VARS
GANTT_AGV_EMPTY_COUNTER = 0
GANTT_TRIGGER_MA = False
GANTT_TRIGGER_AGV = False
GANTT_BEGIN_TRIM = 1475  # Always minus 25
GANTT_END_TRIM = 1575  # Always plus 25

"Initial parameters of the GES"
noAttributesMA = 9
noAttributesAGV = 7

noAttributesJobMA = 4
noAttributesJobAGV = 6

totalAttributes = max(noAttributesMA + noAttributesJobMA, noAttributesAGV + noAttributesJobAGV)

count = 0

AGV_Queue = True
FIFO_agv_queue = False


# %% Simulation info functions

def debug(debug_type, env, job=None, WC=None, ma_number=None, agv_number=None, machine_loc=None, agv_loc=None,
          other_wc=None):
    if DEBUG:
        if debug_type == 1:
            print("CT:", round(env.now, 3), "-", job.name, "entered system ( type", job.type, ")",
                  "and will be processed first at WC:", WC)

        elif debug_type == 2:
            print("CT:", round(env.now, 3), "-", "JPA WC", WC, ": Sended CFPs to MAs!")

        elif debug_type == 3:
            print("CT:", round(env.now, 3), "-", "APA WC", WC, ": Sended CFPs to AGVs!")

        elif debug_type == 4:
            print("CT:", round(env.now, 3), "-", "JPA WC", WC, ": CFP done!", job.name,
                  "will be processed on MA", ma_number, "WC", WC, machine_loc)

        elif debug_type == 5:
            print("CT:", round(env.now, 3), "-", "JPA WC", WC, ": Job stored in APA", WC, "queue")

        elif debug_type == 6:
            print("CT:", round(env.now, 3), "-", "JPA WC", WC, ":", job.name, "removed from JPA", WC, "queue")

        elif debug_type == 7:
            print("CT:", round(env.now, 3), "-", "APA WC", WC, ": CFP done!", job.name, "linked to AGV",
                  agv_number, "WC", other_wc)

        elif debug_type == 8:
            print("CT:", round(env.now, 3), "-", "APA WC", WC, ":", job.name, "removed from APA", WC, "queue")

        elif debug_type == 9:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, "at", agv_loc,
                  ": I will pick", job.name, "which is at location", job.location)

        elif debug_type == 10:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, "at", agv_loc, ":", job.name,
                  "picked up!")

        elif debug_type == 11:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, "at", agv_loc,
                  ": I will bring", job.name, "to MA", ma_number, "WC", other_wc, job.cfp_wc_ma_result)

        elif debug_type == 12:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, "at", agv_loc,
                  ": I will bring", job.name,
                  "to depot")

        elif debug_type == 13:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, "at", agv_loc,
                  ":", job.name, "loaded on MA", ma_number, "WC", other_wc)

        elif debug_type == 14:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, "at", agv_loc, ":",
                  job.name, "dropped at depot")

        elif debug_type == 15:
            print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", WC, job.name,
                  "not at machine")

        elif debug_type == 16:
            print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", WC, job.name,
                  "at machine")

        elif debug_type == 17:
            print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", WC, ": Start processing",
                  job.name)

        elif debug_type == 18:
            print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", WC, ": Finished processing of",
                  job.name)

        elif debug_type == 19:
            print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", WC, ": Breakdown!")

        elif debug_type == 20:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, ": Breakdown!")

        elif debug_type == 21:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, ": Waiting on", job.name,
                  "to be finished by MA", ma_number, "WC", other_wc)

        elif debug_type == 22:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, ": Waiting for new destination", job.name)

        elif debug_type == 23:
            print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", WC, ": Repaired!")

        elif debug_type == 24:
            print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", WC, ": Repaired!")


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
        agv_res = job_shop.agv_process_per_wc[jj, currentWC - 1]

        if len(agv_res.users) > 1:
            print("ERROR")
            exit()

    if AGV_Queue:
        for jj in range(noOfAGVs):
            agv = job_shop.agv_queue_per_wc[jj, currentWC - 1]
            available_agvs.append(agv)
            relative_agv_numbers.append(relative_count)
            no_avail_agvs = len(available_agvs)
            relative_count += 1

    else:
        for jj in range(noOfAGVs):
            agv_res = job_shop.agv_process_per_wc[jj, currentWC - 1]

            if len(agv_res.users) == 0:
                agv = job_shop.agv_queue_per_wc[jj, currentWC - 1]
                available_agvs.append(agv)
                relative_agv_numbers.append(relative_count)

            relative_count += 1
        no_avail_agvs = len(available_agvs)

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
        agv_number = agv_number_WC[currentWC - 1][dispatched_agvs[ii]]
        debug(7, env, jobs[vv], currentWC, None, agv_number, None, None)
        put_job_in_agv_queue(currentWC, dispatched_agvs[ii], jobs[vv], job_shop, agvs)

    # Remove job from queue of the APA
    for ii in reversed(sorted(dispatched_jobs)):
        debug(8, env, jobs[ii], currentWC, None, None, None, None)
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

        debug(7, env, jobs[vv], currentWC, None, agv_number, None, None, dedicated_WC)
        put_job_in_agv_queue(dedicated_WC, dedicated_AGV, jobs[vv], job_shop, agvs)

    # Remove job from queue of the APA
    for ii in reversed(best_job):
        debug(8, env, jobs[ii], currentWC, None, None, None, None, None)
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
        agv_number = agv_number_WC[currentWC - 1][best_bid[ii]]
        debug(7, env, jobs[vv], currentWC, None, agv_number, None, None)
        put_job_in_agv_queue(currentWC, best_bid[ii], jobs[vv], job_shop, agvs)

    # Remove job from queue of the APA
    for ii in reversed(best_job):
        debug(8, env, jobs[ii], currentWC, None, None, None, None)
        yield AGVstore.get(lambda mm: mm == jobs[ii])


def bid_winner_ma(env, jobs, noOfMachines, currentWC, job_shop, machine, store,
                  normaliziation_range_ma):
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
                                            env.now,
                                            job.priority, queue_length, total_pt_ma_queue, normaliziation_range_ma)

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
        ma_number = machine_number_WC[currentWC - 1][best_bid[ii]]

        debug(4, env, jobs[vv], currentWC, ma_number, None, machine_loc, None)

        # If the job is not already planned on an AGV put job in APA
        if not jobs[vv].agv_requested:
            # Put job in APA
            AGVstore = job_shop.AGVstoreWC[currentWC - 1]
            AGVstore.put(jobs[vv])
            trigger = True
            debug(5, env, jobs[vv], currentWC, None, None, None, None)

    # If one of the enumerated jobs is triggered, trigger the APA
    if trigger:

        # Trigger the APA that there is a Job
        if not job_shop.condition_flag_CFP_AGV[currentWC - 1].triggered:
            job_shop.condition_flag_CFP_AGV[currentWC - 1].succeed()

    # Remove job from queue of the JPA
    for ii in reversed(best_job):
        debug(6, env, jobs[ii], currentWC, None, None, None, None)
        yield store.get(lambda mm: mm == jobs[ii])


def bid_calculation_agv(bid_weights, agvnumber, normalization, agv, job, job_shop, processing_time,
                        job_priority, total_rp, due_date, now, queue_distance):
    """Calulcates the bidding value of a job for AGVS."""

    attribute = [0] * noAttributesAGV
    attribute[0] = (queue_distance - normalization[0]) / (normalization[1] - normalization[0]) * \
                   bid_weights[sum(machinesPerWC) + agvnumber - 1][0]  # Total distance AGV queue
    attribute[1] = processing_time / 8.75 * bid_weights[sum(machinesPerWC) + agvnumber - 1][1]  # Processing time
    attribute[2] = (job_priority - 1) / (10 - 1) * bid_weights[sum(machinesPerWC) + agvnumber - 1][2]  # Job Priority
    attribute[3] = len(agv[0].items) / 25 * bid_weights[sum(machinesPerWC) + agvnumber - 1][3]  # AGV Queue length
    attribute[4] = total_rp / 25 * bid_weights[sum(machinesPerWC) + agvnumber - 1][4]  # Remaining processing time
    attribute[5] = (due_date - now - normalization[2]) / (normalization[3] - normalization[2]) * \
                   bid_weights[sum(machinesPerWC) + agvnumber - 1][5]  # Due date
    attribute[6] = 0

    """attribute1 = [0] * noAttributesAGV
    attribute1[0] = (queue_distance - normalization[0]) / (normalization[1] - normalization[0])
    attribute1[1] = processing_time / 8.75
    attribute1[2] = (job_priority - 1) / (10 - 1)
    attribute1[3] = len(agv[0].items) / 25
    attribute1[4] = total_rp / 25
    attribute1[5] = (due_date - now - normalization[2]) / (normalization[3] - normalization[2])
    attribute1 = np.around(attribute1, 2)

    if len([*filter(lambda x: x >= 1.01, attribute1)]) > 0:
        print(" ==== AGV Attributes ====")
        print(attribute1)

    if len([*filter(lambda x: x < 0, attribute1)]) > 0:
        print(" ==== AGV Attributes ====")
        print(attribute1)"""

    return sum(attribute)


def bid_calculation_ma(bid_weights, machinenumber, processing_time,
                       current, total_rp, due_date, now, job_priority, queue_length, total_pt_queue,
                       normalization):
    """Calulcates the bidding value of a job for MAs."""

    attribute = [0] * noAttributesMA
    attribute[0] = processing_time / 8.75 * bid_weights[machinenumber - 1][0]  # processing time
    attribute[1] = (current - 1) / (5 - 1) * bid_weights[machinenumber - 1][1]  # remaing operations
    attribute[2] = (due_date - now - normalization[0]) / (normalization[1] - normalization[0]) * \
                   bid_weights[machinenumber - 1][2]  # slack
    attribute[3] = total_rp / 25 * bid_weights[machinenumber - 1][3]  # remaining processing time
    attribute[4] = (((due_date - now) / total_rp) - normalization[2]) / (normalization[3] - normalization[2]) * \
                   bid_weights[machinenumber - 1][4]  # Critical Ratio
    attribute[5] = (job_priority - 1) / (10 - 1) * bid_weights[machinenumber - 1][5]  # Job Priority
    attribute[6] = queue_length / 25 * bid_weights[machinenumber - 1][6]  # Queue length
    attribute[7] = (total_pt_queue - normalization[4]) / (normalization[5] - normalization[4]) * \
                   bid_weights[machinenumber - 1][7]  # Total processing time queue

    """attribute[8] = 0

    attribute1 = [0] * noAttributesMA
    attribute1[0] = processing_time / 8.75
    attribute1[1] = (current - 1) / (5 - 1)
    attribute1[2] = (due_date - now - normalization[0]) / (normalization[1] - normalization[0])
    attribute1[3] = total_rp / 25
    attribute1[4] = (((due_date - now) / total_rp) - normalization[2]) / (normalization[3] - normalization[2])
    attribute1[5] = (job_priority - 1) / (10 - 1)
    attribute1[6] = queue_length / 25
    attribute1[7] = (total_pt_queue - normalization[4]) / (normalization[5] - normalization[4])
    attribute1 = np.around(attribute1, 2)

    if len([*filter(lambda x: x >= 1.01, attribute1)]) > 0:
        print(" ==== MA Attributes ====")
        print(attribute1)

    if len([*filter(lambda x: x < 0, attribute1)]) > 0:
        print(" ==== MA Attributes ====")
        print(attribute1)"""

    return sum(attribute)


def queue_total_distance(agv, job_shop):
    distance = 0

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

    global count

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
        job_shop.priority[job.number] = job.priority
        job_shop.flowtime[job.number] = finish_time - job.dueDate[0]

        if job.number > max_job:
            if np.count_nonzero(job_shop.flowtime[min_job:max_job]) == 2000:
                job_shop.finish_time = env.now
                job_shop.end_event.succeed()

        if (job_shop.WIP > max_wip) | (env.now > 10_000):

            if job_shop.WIP > max_wip:
                print("To much WIP")
            elif env.now > 10_000:
                print("Time eslaped")
            else:
                print("bad simulation")

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
    # global count

    """Calculates prioirities of jobs in a machines queue"""
    attribute_job = [0] * noAttributesJobMA

    attribute_job[0] = (due_date - processing_time - setup_time - env.now - normalization[6]) / (
            normalization[7] - normalization[6]) * \
                       job_weights[machinenumber - 1][noAttributesMA]
    attribute_job[1] = (job_priority - 1) / (10 - 1) * job_weights[machinenumber - 1][noAttributesMA + 1]
    attribute_job[2] = setup_time / 1.25 * job_weights[machinenumber - 1][noAttributesMA + 2]
    attribute_job[3] = job_present * job_weights[machinenumber - 1][noAttributesMA + 3]

    """attribute1 = [0] * noAttributesJobMA
    attribute1[0] = (due_date - processing_time - setup_time - env.now - normalization[6]) / (
            normalization[7] - normalization[6]) 
    attribute1[2] = (job_priority - 1) / (10 - 1) 
    attribute1[3] = job_present 
    attribute1 = np.around(attribute1, 2)

    if len([*filter(lambda x: x >= 1.01, attribute1)]) > 0:
        count += 1
        print(" ==== MA Sequencing ====")
        print(attribute1)

    if len([*filter(lambda x: x < 0, attribute1)]) > 0:
        count += 1
        print(" ==== MA Sequencing ====")
        print(attribute1)"""

    return sum(attribute_job)


def choose_job_queue_agv(job_weights, job, normalization, agv, agvnumber, env, due_date, job_priority, job_shop, remaining_pt, JAFAMT):
    """Calculates prioirities of jobs in an agv queue"""

    global count

    # job.location = [depot, WC1, WC2, WC3, WC4, WC5]
    job_location_set = {"d": 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6}

    attribute_job = [0] * noAttributesJobAGV
    attribute_job[0] = (job_priority - 1) / (10 - 1) * job_weights[sum(machinesPerWC) + agvnumber - 1][
        noAttributesAGV]  # Job priority
    attribute_job[1] = (job_shop.travel_time_matrix[agv[1]][job.location] / 1.5) * \
                       job_weights[sum(machinesPerWC) + agvnumber - 1][
                           noAttributesAGV + 1]  # Job travel distance
    attribute_job[2] = (job_location_set[job.location[0]] / 6) * job_weights[sum(machinesPerWC) + agvnumber - 1][
        noAttributesAGV + 2]  # Job location
    attribute_job[3] = ((due_date - env.now - normalization[4]) / (normalization[5] - normalization[4])) * \
                       job_weights[sum(machinesPerWC) + agvnumber - 1][
                           noAttributesAGV + 3]  # Due date
    attribute_job[4] = (remaining_pt / (JAFAMT + 0.00000001)) * job_weights[sum(machinesPerWC) + agvnumber - 1][
        noAttributesAGV + 4]  # remaining_pt
    attribute_job[5] = 0

    """attribute1 = [0] * noAttributesJobAGV
    attribute1[0] = (job_priority - 1) / (10 - 1) 
    attribute1[1] = (job_shop.travel_time_matrix[agv[1]][job.location] / 1.5) 
    attribute1[2] = (job_location_set[job.location[0]] / 6) 
    attribute1[3] = ((due_date - env.now - normalization[4]) / (normalization[5] - normalization[4])) 
    attribute1 = np.around(attribute1, 2)

    if len([*filter(lambda x: x >= 1.01, attribute1)]) > 0:
        count += 1
        print(" ==== AGV Sequencing ====")
        print(attribute1)

    if len([*filter(lambda x: x < 0, attribute1)]) > 0:
        count += 1
        print(" ==== AGV Sequencing ====")
        print(attribute1)"""

    return sum(attribute_job)


def agv_processing(job_shop, currentWC, agv_number, env, agv, normalization, agv_buf,
                   agv_number_WC, JAFAMT):
    """This refers to an AGV Agent in the system. It checks which jobs it wants to transfer
        next to machines and stores relevant information regarding it."""

    global GANTT_TRIGGER_AGV
    global GANTT_AGV_EMPTY_COUNTER

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
                debug(9, env, next_job, currentWC, None, agv_number, None, agv_location, None)

                if AGV_ROUTING:

                    if next_job.location == "depot":
                        job_shop.register_agv_routing(agv_number, None, next_job.location)

                    else:
                        ma_number = machine_number_WC[next_job.location[1]][next_job.location[0]]
                        job_shop.register_agv_routing(agv_number, ma_number, next_job.location)

                driving_time = job_shop.travel_time_matrix[agv_location][next_job.location]

                job_shop.agv_utilization[(relative_agv, currentWC - 1)] = job_shop.agv_utilization[(
                    relative_agv, currentWC - 1)] + driving_time

                job_shop.AGV_load_unloaded[(relative_agv, currentWC - 1)][1] += 1

                if GANTT_AGV:
                    if GANTT_BEGIN_TRIM < env.now < GANTT_END_TRIM:
                        GANTT_TRIGGER_AGV = True
                    else:
                        GANTT_TRIGGER_AGV = False

                    if GANTT_TRIGGER_AGV:
                        job_shop.update_gantt(driving_time, env.now + driving_time,
                                              "EMPTY" + str(GANTT_AGV_EMPTY_COUNTER), "AGV " + str(agv_number), env.now)

                        GANTT_AGV_EMPTY_COUNTER += 1

                yield env.timeout(driving_time)

                job_shop.AGV_total_driving_time["AGV" + str(agv_number)].append(driving_time)

                # Change AGV location
                agv[1] = next_job.location
                agv_location = agv[1]

            #  Load the job on the AGV and remove job from depot or machine
            if next_job in job_shop.depot_queue.items:
                job_shop.depot_queue.items.remove(next_job)

            elif next_job in job_shop.machine_buffer_per_wc[agv_location][0].items:

                if not next_job.job_in_progress.triggered:
                    ma_number = machine_number_WC[agv_location[1]][agv_location[0]]
                    debug(21, env, next_job, currentWC, ma_number, agv_number, None, None, agv_location[1] + 1)

                yield next_job.job_in_progress

                if QUEUE:
                    # Register queue length
                    ma_number = machine_number_WC[agv_location[1]][agv_location[0]]
                    job_shop.update_ma_queue(env.now, agv_location[1], ma_number, -1)

                next_job.average_waiting_time_pickup = (next_job.average_waiting_time_pickup + (
                        env.now - next_job.finishing_time_machine)) / next_job.currentOperation

                job_shop.machine_buffer_per_wc[agv_location][0].items.remove(next_job)

            # Always reset the job in progress condition
            next_job.job_in_progress = simpy.Event(env)

            # Put job on AGV buffer
            agv_buf.put(next_job)

            job_destination = next_job.cfp_wc_ma_result

            if job_destination is None:
                debug(22, env, next_job, currentWC, None, agv_number, None, None, None)

                yield next_job.job_destination_set

                job_destination = next_job.cfp_wc_ma_result

            # Always reset the job destination set condition
            next_job.job_destination_set = simpy.Event(env)

            # Put job in machine buffer "jobs underway"
            if not job_destination == "depot":
                job_shop.machine_buffer_per_wc[job_destination][1].put(next_job)

            if not next_job.cfp_wc_ma_result == "depot":

                ma_number = machine_number_WC[next_job.cfp_wc_ma_result[1]][next_job.cfp_wc_ma_result[0]]
                debug(10, env, next_job, currentWC, None, agv_number, None, agv_location, None)
                debug(11, env, next_job, currentWC, ma_number, agv_number, None, agv_location,
                      next_job.cfp_wc_ma_result[1] + 1)

                if AGV_ROUTING:
                    job_shop.register_agv_routing(agv_number, ma_number, next_job.cfp_wc_ma_result)

            else:

                debug(10, env, next_job, currentWC, None, agv_number, None, agv_location, None)
                debug(12, env, next_job, currentWC, None, agv_number, None, agv_location, None)

                if AGV_ROUTING:
                    job_shop.register_agv_routing(agv_number, None, next_job.cfp_wc_ma_result)

            driving_time = job_shop.travel_time_matrix[agv_location][job_destination]

            job_shop.agv_utilization[(relative_agv, currentWC - 1)] = job_shop.agv_utilization[(
                relative_agv, currentWC - 1)] + driving_time

            job_shop.AGV_load_unloaded[(relative_agv, currentWC - 1)][0] += 1

            if GANTT_AGV:
                if GANTT_BEGIN_TRIM < env.now < GANTT_END_TRIM:
                    GANTT_TRIGGER_AGV = True
                else:
                    GANTT_TRIGGER_AGV = False

                if GANTT_TRIGGER_AGV:
                    job_shop.update_gantt(driving_time, env.now + driving_time,
                                          next_job.name, "AGV " + str(agv_number), env.now)

            yield env.timeout(driving_time)

            job_shop.AGV_total_driving_time["AGV" + str(agv_number)].append(driving_time)

            # Change job and AGV location
            next_job.location = job_destination
            agv[1] = next_job.location
            agv_location = next_job.location

            #  Unload the job from the AGV and put in machine or machine buffer
            agv[0].items.remove(next_job)
            agv_buf.items.remove(next_job)

            if not agv_location == "depot":

                if QUEUE:
                    # Register queue length
                    ma_number = machine_number_WC[agv_location[1]][agv_location[0]]
                    job_shop.update_ma_queue(env.now, ma_number, 1)

                machine_buf = job_shop.machine_buffer_per_wc[agv_location][1]
                machine_buf.items.remove(next_job)
                machine_buf = job_shop.machine_buffer_per_wc[agv_location][0]
                machine_buf.put(next_job)

                #  Unload the job from the AGV and put job on machine, machine buffer or depot
                if not next_job.job_loaded_condition.triggered and next_job in machine_buf.items:
                    ma_number = machine_number_WC[agv_location[1]][agv_location[0]]
                    debug(13, env, next_job, currentWC, ma_number, agv_number, None, agv_location, agv_location[1] + 1)
                    next_job.job_loaded_condition.succeed()

            else:
                debug(14, env, next_job, currentWC, None, agv_number, None, agv_location, None)
                job_shop.depot_queue.put(next_job)

            # Trigger the APA that there is an idle AGV
            if not job_shop.condition_flag_CFP_AGV[currentWC - 1].triggered:
                job_shop.condition_flag_CFP_AGV[currentWC - 1].succeed()

        else:
            yield job_shop.condition_flag_agv[
                (relative_agv, currentWC - 1)]  # Used if there is currently no job in the agv queue
            job_shop.condition_flag_agv[(relative_agv, currentWC - 1)] = simpy.Event(env)  # Reset event if it is used


def machine_processing(job_shop, currentWC, machine_number, env, last_job, machine,
                       makespan, min_job, max_job, normalization, max_wip, machine_buf, JAFAMT):
    """This refers to a Machine Agent in the system. It checks which jobs it wants to process
    next and stores relevant information regarding it."""
    global GANTT_TRIGGER_MA

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
            ma_number = machine_number_WC[currentWC - 1][next_job.cfp_wc_ma_result[0]]

            # If job is not at machine, machine will be yielded until AGVs loads job
            if next_job not in machine_buf[0].items:
                debug(15, env, next_job, currentWC, ma_number, None, None, None, None)
                yield next_job.job_loaded_condition
                debug(16, env, next_job, currentWC, ma_number, None, None, None, None)

            # Always reset the job loaded condition
            next_job.job_loaded_condition = simpy.Event(env)

            setuptime = setup_time[ind_processing_job]
            time_in_processing = next_job.processingTime[
                                     next_job.currentOperation - 1] + setuptime  # Total time the machine needs to process the job

            next_job.finished_job = False
            next_job.actual_finish_proc_time = time_in_processing + env.now

            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, env, setuptime)

            job_shop.ma_utilization[(relative_machine, currentWC - 1)] = job_shop.ma_utilization[(
                relative_machine, currentWC - 1)] + setuptime + next_job.processingTime[
                                                                             next_job.currentOperation - 1]

            last_job[relative_machine] = next_job.type

            machine.items.remove(next_job)  # Remove job from queue

            debug(17, env, next_job, currentWC, ma_number, None, None, None, None)

            if GANTT_Machine:
                if GANTT_BEGIN_TRIM < env.now < GANTT_END_TRIM:
                    GANTT_TRIGGER_MA = True
                else:
                    GANTT_TRIGGER_MA = False

                if GANTT_TRIGGER_MA:
                    job_shop.update_gantt(time_in_processing, env.now + time_in_processing,
                                          next_job.name, "MA " + str(machine_number), env.now)

            next_job.cfp_wc_ma_result = None

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

            debug(18, env, next_job, currentWC, ma_number, None, None, None, None)

            next_job.finishing_time_machine = env.now

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
            debug(2, env, None, currentWC, None, None, None, None)

            job_shop.QueuesWC[currentWC - 1].append(
                {ii: len(job_shop.machine_queue_per_wc[(ii, currentWC - 1)].items) for ii in
                 range(machinesPerWC[currentWC - 1])})  # Stores the Queue length of the JPA

            c = bid_winner_ma(env, store.items, machinesPerWC[currentWC - 1], currentWC, job_shop,
                              machine, store, normalization)

            yield env.process(c)

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

            debug(3, env, None, currentWC, None, None, None, None, None)

            job_shop.AGVQueuesWC[currentWC - 1].append(
                {ii: len(job_shop.agv_queue_per_wc[(ii, currentWC - 1)][0].items) for ii in
                 range(agvsPerWC[currentWC - 1])})

            # Bidding control
            if dispatch_rule_no == 1:
                c = bid_winner_agv_per_WC(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop,
                                          agvs, AGVstore, normalization, agv_number_WC)

                yield env.process(c)

            # Bidding control - No AGV dedicated to WC
            if dispatch_rule_no == 2:
                c = bid_winner_agv_all_WC(env, job_list, agvsPerWC, currentWC, job_shop,
                                          agvs, AGVstore, normalization, agv_number_WC)

                yield env.process(c)

            # Dispatch control
            if dispatch_rule_no > 2:
                c = dispatch_control(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop, agvs, AGVstore,
                                     dispatch_rule_no, agvsPerWC, agv_number_WC)
                yield env.process(c)

        if immediate_release:

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

        debug(1, env, job, firstWC, None, None, None, None)

        tib = random.expovariate(1.0 / interval)

        yield env.timeout(tib)


# %% Plotting & Visualize functions

def AGV_routing(job_shop, agv_number_WC):
    dictionary = job_shop.AGV_routing_register
    df = DataFrame(dictionary).fillna(0).transpose()
    agv_trips = list([idx, *values] for idx, values in zip(df.index, df.values.astype(int).tolist()))
    agv_trips = np.delete(agv_trips, [0], 1)
    agv_trips = agv_trips.astype(np.int)

    locations_labels = [list(df.columns)]
    agv_labels = job_shop.AGV_routing_register.keys()

    y_axis_labels = agv_labels  # labels for x-axis
    x_axis_labels = locations_labels[0]  # labels for y-axis

    cmap = sb.cm.rocket_r
    sb.set(font_scale=1.5)
    heat_map = sb.heatmap(agv_trips, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap=cmap, linewidths=0.3,
                          annot=True, fmt="d")

    # plt.xlabel("Machine & depot locations")
    # plt.ylabel('AGVs')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)

    plt.gcf().set_size_inches(20, 8)
    plt.show()


def MA_queue_length_plot(machine_queues, time):
    """for machine in machine_queues:
        plt.plot(time, machine_queues[machine], label = machine)"""
    fig = plt.figure(figsize=(24, 10))
    plt.plot(time, machine_queues["MA1"], label="MA1")
    plt.plot(time, machine_queues["MA2"], label="MA2")
    plt.plot(time, machine_queues["MA3"], label="MA3")
    plt.plot(time, machine_queues["MA4"], label="MA4")
    plt.title("Work-Center 1")
    plt.legend()
    plt.xlim([0, time[-1]])
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(24, 10))
    plt.plot(time, machine_queues["MA5"], label="MA5")
    plt.plot(time, machine_queues["MA6"], label="MA6")
    plt.title("Work-Center 2")
    plt.legend()
    plt.xlim([0, time[-1]])
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(24, 10))
    plt.plot(time, machine_queues["MA7"], label="MA7")
    plt.plot(time, machine_queues["MA8"], label="MA8")
    plt.plot(time, machine_queues["MA9"], label="MA9")
    plt.plot(time, machine_queues["MA10"], label="MA10")
    plt.plot(time, machine_queues["MA11"], label="MA11")
    plt.title("Work-Center 3")
    plt.legend()
    plt.xlim([0, time[-1]])
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(24, 10))
    plt.plot(time, machine_queues["MA12"], label="MA12")
    plt.plot(time, machine_queues["MA13"], label="MA13")
    plt.plot(time, machine_queues["MA14"], label="MA14")
    plt.title("Work-Center 4")
    plt.legend()
    plt.xlim([0, time[-1]])
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(24, 10))
    plt.plot(time, machine_queues["MA15"], label="MA15")
    plt.plot(time, machine_queues["MA16"], label="MA16")
    plt.title("Work-Center 5")
    plt.legend()
    plt.xlim([0, time[-1]])
    fig.tight_layout()
    plt.show()


def visualize(gantt_list, GANTT_type):
    schedule = pd.DataFrame(gantt_list)
    JOBS = sorted(list(schedule['Job'].unique()))
    MACHINES = sorted(list(schedule['Machine'].unique()))
    makespan = schedule['Finish'].max()

    bar_style = {'alpha': 1.0, 'lw': 40, 'solid_capstyle': 'butt'}
    text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center'}
    colors = mpl.cm.Dark2.colors

    schedule.sort_values(by=['Job', 'Start'])
    schedule.set_index(['Job', 'Machine'], inplace=True)

    if GANTT_type == "Machine":
        # fig = plt.figure(figsize=(20, 5 + (len(JOBS) + len(MACHINES)) / 4))
        fig = plt.figure(figsize=(30, 18), dpi=150)

    if GANTT_type == "AGV":
        # fig = plt.figure(figsize=(20, 5 + (len(JOBS) + len(MACHINES)) / 4))
        fig = plt.figure(figsize=(30, 9), dpi=150)

    ax = fig.add_subplot()

    ax.autoscale(enable=True)

    for jdx, job in enumerate(JOBS, 1):
        for mdx, machine in enumerate(MACHINES, 1):
            if (job, machine) in schedule.index:

                xs = schedule.loc[(job, machine), 'Start']
                xf = schedule.loc[(job, machine), 'Finish']

                if GANTT_type == "AGV":
                    xs = xs.to_numpy()[0]
                    xf = xf.to_numpy()[0]
                if job[:5] == "EMPTY":
                    job = ""
                    ax.plot([xs, xf], [mdx] * 2, c="#636061", **bar_style)
                else:
                    ax.plot([xs, xf], [mdx] * 2, c=colors[jdx % 7], **bar_style)
                ax.text((xs + xf) / 2, mdx, job, **text_style, clip_on=True)

    if GANTT_type == "Machine":
        ax.set_title('Machine Schedule', fontsize=30, fontweight='bold', x=0.5, y=1.05)
        ax.set_ylabel('Machine', fontsize=30, fontweight='bold')

    if GANTT_type == "AGV":
        ax.set_title('AGV Schedule', fontsize=30, fontweight='bold', x=0.5, y=1.05)
        ax.set_ylabel('Job', fontsize=30, fontweight='bold')

    for idx, s in enumerate([JOBS, MACHINES]):

        if GANTT_type == "AGV":
            ax.set_xlim(GANTT_BEGIN_TRIM + 25, GANTT_END_TRIM - 25)
        else:
            ax.set_xlim(GANTT_BEGIN_TRIM + 25, GANTT_END_TRIM - 25)

        ax.set_ylim(0.5, len(s) + 0.5)
        ax.set_yticks(range(1, 1 + len(s)))
        ax.set_yticklabels(s)
        ax.text(makespan, ax.get_ylim()[0] - 0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        ax.plot([makespan] * 2, ax.get_ylim(), 'r--')
        ax.set_xlabel('Time', fontsize=40, fontweight='bold')
        ax.grid(True)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(24)
        label.set_fontweight('bold')

    # fig.tight_layout()
    plt.show()


# %% Result functions

def get_objectives(job_shop, min_job, max_job, early_termination):
    """This function gathers numerous results from a simulation run"""
    no_tardy_jobs_p1 = 0
    no_tardy_jobs_p2 = 0
    no_tardy_jobs_p3 = 0
    total_p1 = 0
    total_p2 = 0
    total_p3 = 0
    early_term = 0

    if early_termination == 1:
        early_term += 1
        makespan = job_shop.finish_time - job_shop.start_time
        flow_time = np.nanmean(job_shop.flowtime[min_job:max_job]) + 10_000 - np.count_nonzero(
            job_shop.flowtime[min_job:max_job])
        mean_tardiness = np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job])) + 10_000 - np.count_nonzero(
            job_shop.flowtime[min_job:max_job])
        max_tardiness = np.nanmax(job_shop.tardiness[min_job:max_job])
        for ii in range(min_job, len(job_shop.tardiness)):
            if job_shop.priority[ii] == 1:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p1 += 1
                total_p1 += 1
            elif job_shop.priority[ii] == 3:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p2 += 1
                total_p2 += 1
            elif job_shop.priority[ii] == 10:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p3 += 1
                total_p3 += 1
        # WIP Level
        mean_WIP = np.mean(job_shop.totalWIP)
    else:
        makespan = job_shop.finish_time - job_shop.start_time
        # Mean Flow Time
        flow_time = np.nanmean(job_shop.flowtime[min_job:max_job])
        # Mean Tardiness
        mean_tardiness = np.nanmean(job_shop.tardiness[min_job:max_job])
        # Max Tardiness
        max_tardiness = max(job_shop.tardiness[min_job:max_job])
        # print(len(job_shop.priority))
        # No of Tardy Jobs
        for ii in range(min_job, max_job):
            if job_shop.priority[ii] == 1:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p1 += 1
                total_p1 += 1
            elif job_shop.priority[ii] == 3:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p2 += 1
                total_p2 += 1
            elif job_shop.priority[ii] == 10:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p3 += 1
                total_p3 += 1
        # WIP Level
        mean_WIP = np.mean(job_shop.totalWIP)

    # Machines utilizations
    utilization_MAs = []
    for wc in range(len(machinesPerWC)):
        for ii in range(machinesPerWC[wc]):
            utilization_MAs.append(round(job_shop.ma_utilization[(ii, wc)] / job_shop.finish_time, 2))

    # AGV utilizations
    utilization_AGVs = []
    load_unload_AGVs = []
    for wc in range(len(agvsPerWC)):
        for ii in range(agvsPerWC[wc]):
            utilization_AGVs.append(round(job_shop.agv_utilization[(ii, wc)] / job_shop.finish_time, 2))

            loaded = job_shop.AGV_load_unloaded[(ii, wc)][0]
            unloaded = job_shop.AGV_load_unloaded[(ii, wc)][1]

            #ratio = (loaded + unloaded) / loaded
            ratio = 0
            load_unload_AGVs.append(ratio)

    # print("")
    # print("Early termination", job_shop.early_termination)
    # print("Tardy jobs prio 1", no_tardy_jobs_p1)
    # print("Tardy jobs prio 2", no_tardy_jobs_p2)
    # print("Tardy jobs prio 3", no_tardy_jobs_p3)
    # print("Flow time", flow_time)
    # print("Mean Tardiness", mean_tardiness)
    # print("")

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, \
           mean_WIP, early_term, utilization_MAs, utilization_AGVs, load_unload_AGVs


# %% Main Simulation function

def do_simulation_with_weights(mean_weights, arrivalMean, due_date_tightness, min_job, max_job,
                               normalization_ma, normalization_AGV, max_wip, AGV_rule_no, travel_time_matrix,
                               immediate_release, JAFAMT_value, iter1):
    """ This runs a single simulation"""



    random.seed(iter1)
    np.random.seed(iter1)

    env = Environment()  # Create Environment

    job_shop = jobShop(env, mean_weights, travel_time_matrix, agvsPerWC,
                       agv_number_WC)  # Initiate the job shop
    env.process(source(env, 0, arrivalMean, job_shop, due_date_tightness,
                       min_job))  # Starts the source (Job Release Agent)

    for wc in range(len(machinesPerWC)):

        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        MAstoreWC = job_shop.MAstoreWC[wc]
        AGVstoreWC = job_shop.AGVstoreWC[wc]

        env.process(
            cfp_wc_ma(env, job_shop.machine_queue_per_wc, MAstoreWC, job_shop, wc + 1, normalization_ma))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_queue_per_wc[(ii, wc)]
            machine_buf = job_shop.machine_buffer_per_wc[(ii, wc)]

            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, last_job,
                                   machine, makespanWC, min_job, max_job, normalization_ma, max_wip,
                                   machine_buf, JAFAMT_value))

        env.process(
            cfp_wc_agv(env, job_shop.agv_queue_per_wc, AGVstoreWC, job_shop, wc + 1, normalization_AGV, AGV_rule_no,
                       immediate_release, agvsPerWC, agv_number_WC))

        for ii in range(agvsPerWC[wc]):
            agv = job_shop.agv_queue_per_wc[(ii, wc)]
            agv_buf = job_shop.agv_buffer_per_wc[(ii, wc)]

            env.process(agv_processing(job_shop, wc + 1, agv_number_WC[wc][ii], env,
                                       agv, normalization_AGV, agv_buf, agv_number_WC, JAFAMT_value))



    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)  # Run the simulation until the end event gets triggered

    if AGV_ROUTING:
        AGV_routing(job_shop, agv_number_WC)

    if GANTT_Machine:
        visualize(job_shop.gantt_list_ma, 'Machine')

    if GANTT_AGV:
        visualize(job_shop.gantt_list_agv, "AGV")

    if QUEUE:
        MA_queue_length_plot(job_shop.QueuesMAs, job_shop.QueueTimes)

    makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, \
    mean_WIP, early_term, utilization_result_MA, utilization_result_AGV, load_unload_AGVs = get_objectives(job_shop,
                                                                                                           min_job,
                                                                                                           max_job,
                                                                                                           job_shop.early_termination)  # Gather all results

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, \
           mean_WIP, early_term, utilization_result_MA, utilization_result_AGV, load_unload_AGVs


# %% Classes

class jobShop:
    """This class creates a job shop, along with everything that is needed to run the Simpy Environment."""

    def __init__(self, env, stored_weights, travel_time_matrix, agvsPerWC, agv_number_WC):

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

        self.weights = stored_weights
        self.flowtime = []  # Jobs flowtime
        self.tardiness = []  # Jobs tardiness
        self.makespan = []
        self.WIP = 0  # Current WIP of the system
        self.early_termination = 0  # Whether the simulation terminated earlier
        self.finish_time = 0  # Finishing time of the system
        self.totalWIP = []  # Keeps track of the total WIP of the system

        self.ma_utilization = {(ii, jj): 0 for jj in noOfWC for ii in range(machinesPerWC[jj])}
        self.agv_utilization = {(ii, jj): 0 for jj in noOfWC for ii in range(agvsPerWC[jj])}

        self.AGV_load_unloaded = {(ii, jj): [0, 0] for jj in noOfWC for ii in range(agvsPerWC[jj])}

        self.bids = []  # Keeps track of the bids
        self.priority = []  # Keeps track of the job priorities
        self.start_time = 0  # Starting time of the simulation

        self.gantt_list_ma = []  # List with machine job information for gantt chart
        self.gantt_list_agv = []  # List with agv job information for gantt chart

        self.QueuesMAs = {"MA" + str(ii): [0] for jj in noOfWC for ii in machine_number_WC[jj]}
        self.QueueTimes = [0]

        self.AGV_total_driving_time = {"AGV" + str(ii): [0] for jj in noOfWC for ii in agv_number_WC[jj]}

        self.AGV_routing_register = {
            "AGV" + str(ii): {"MA" + str(ii): 0 for jj in noOfWC for ii in machine_number_WC[jj]} for jj in noOfWC for
            ii in agv_number_WC[jj]}

        for jj in noOfWC:
            for ii in agv_number_WC[jj]:
                self.AGV_routing_register["AGV" + str(ii)].update({"depot": 0})

    def register_agv_routing(self, agv_number, machine_number, location):

        if location == "depot":
            update_location = "depot"
        else:
            update_location = "MA" + str(machine_number)

        self.AGV_routing_register["AGV" + str(agv_number)][update_location] += 1

    def update_gantt(self, duration, finish_time, job, machine, start_time):

        if machine[:3] == "AGV":
            gantt_dict_agv = {'Duration': duration,
                              'Finish': finish_time,
                              'Job': job,
                              'Machine': machine,
                              'Start': start_time}

            self.gantt_list_agv.append(gantt_dict_agv)

        if machine[:2] == "MA":
            gantt_dict_ma = {'Duration': duration,
                             'Finish': finish_time,
                             'Job': job,
                             'Machine': machine,
                             'Start': start_time}

            self.gantt_list_ma.append(gantt_dict_ma)

    def update_ma_queue(self, time, machine_nr, amount):

        self.QueueTimes.append(time)
        self.QueueTimes.append(time)

        for machine in self.QueuesMAs:
            last_queue_length = self.QueuesMAs[machine][-1]
            self.QueuesMAs[machine].append(last_queue_length)

        for machine in self.QueuesMAs:
            if machine == "MA" + str(machine_nr):
                last_queue_length = self.QueuesMAs["MA" + str(machine_nr)][-1]
                self.QueuesMAs["MA" + str(machine_nr)].append(last_queue_length + amount)

            else:
                last_queue_length = self.QueuesMAs[machine][-1]
                self.QueuesMAs[machine].append(last_queue_length)


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

    no_runs = 60
    no_processes = 6  # Change dependent on number of threads computer has, be sure to leave 1 thread remaining

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
    simulation_parameter_1 = [1]

    # Simulation Parameter 2 - Number of AGVs per work center
    # 0: 6 AGV per WC - ZERO TRAVEL TIME!
    # 1: Manual number - ZERO TRAVEL TIME!
    # 2: Manual number
    simulation_parameter_2 = [scenario]

    # Simulation Parameter 3 - Job almost finished at machines trigger values
    simulation_parameter_3 = [0.0]

    # Simulation Parameter 4 - Direct or periodically job release APA (Direct = True)
    simulation_parameter_4 = [False]

    # Simulation Parameter 5 - Number of AGVs
    simulation_parameter_5 = [agvsPerWC]

    if simulation_parameter_2[0] == 1:  # scenario 1
        min_jobs = [499, 999, 1499]  # Minimum number of jobs in order te reach steady state
        max_jobs = [2499, 2999, 3499]  # Maximum number of jobs to collect information from
        wip_max = [150, 250, 300]  # Maximum WIP allowed in the system

    else:  # scenario 2
        min_jobs = [499, 999, 1499]  # Minimum number of jobs in order te reach steady state
        max_jobs = [2499, 2999, 3499]  # Maximum number of jobs to collect information from
        wip_max = [150, 400, 300]  # Maximum WIP allowed in the system"""

    arrival_time = [arrival_rate[0]]

    utilization = [90]
    due_date_settings = [4, 4, 4]

    normalization_MA_array = [[-112.5, 225.0, -21.0, 21.0, -112.5, 225.0, -10, 75],
                              [-112.5, 225.0, -21.0, 21.0, -112.5, 225.0, -700, 250],
                              [-112.5, 225.0, -21.0, 21.0, -112.5, 225.0, -10, 75]]

    normalization_AGV_array = [[-10, 25, -112.5, 225.0, -112.5, 225.0],
                               [-10, 25, -112.5, 225.0, -250, 225.0],
                               [-10, 25, -112.5, 225.0, -112.5, 225.0]]

    for (a, b, c, d, e) in itertools.product(simulation_parameter_1, simulation_parameter_2, simulation_parameter_3,
                                             simulation_parameter_4, simulation_parameter_5):

        AGV_rule = a
        AGV_selection = b
        JAFAMT_value = c
        immediate_release_bool = d
        agvsPerWC = e

        print("Simulation:", "(" + str(a) + "-" + str(b) + "-" + str(c) + "-" + str(d) + ")")

        if simulation_parameter_1[0] == 2:
            sim_par_1_string = "AGV_ALL_WC"
        elif simulation_parameter_1[0] == 1:
            sim_par_1_string = "AGV_PER_WC"
        elif simulation_parameter_1[0] == 3:
            sim_par_1_string = "DISP_RULE_3"

        sim_par_2_string = "JAFAMT_" + str(simulation_parameter_3[0]) + "_" + str(simulation_parameter_4[0])

        for i in range(len(utilization)):

            final_result = np.zeros((no_runs, 9))
            results = []

            ma_utilizations = []
            agv_utilizations = []
            average_ma_utilization = []
            average_agv_utilization = []
            total_average_machine_utilization = []
            total_average_avg_utilization = []

            load_unload_AGVs_list = []
            average_load_unload_AGVs_list = []

            utilization_per_sim = []

            str1 = "Runs/" + "scenario_" + str(
                scenario) + "/" + sim_par_1_string + "/" + sim_par_2_string + "/" + "/" + str(
                agvsPerWC) + "/Final_Runs/Run-weights-" + str(
                utilization[i]) + "-" + str(
                due_date_settings[i]) + ".csv"

            df = pd.read_csv(str1, header=None)
            weights = df.values.tolist()

            print("Current run is: " + str(utilization[i]) + "-" + str(due_date_settings[i]))
            obj = np.zeros(no_runs)

            for j in range(int(no_runs / no_processes)):

                jobshop_pool = Pool(processes=no_processes)
                seeds = range(j * no_processes, j * no_processes + no_processes)
                func1 = partial(do_simulation_with_weights, weights, arrival_time[i],
                                due_date_settings[i],
                                min_jobs[i + 1], max_jobs[i + 1], normalization_MA_array[i + 1],
                                normalization_AGV_array[i + 1],
                                wip_max[i + 1], AGV_rule,
                                created_travel_time_matrix, immediate_release_bool, JAFAMT_value)

                makespan_per_seed = jobshop_pool.map(func1, seeds)

                # Gather final results
                for h, o in itertools.product(range(no_processes), range(9)):
                    final_result[h + j * no_processes][o] = makespan_per_seed[h][o]

                    ma_utilizations.append(makespan_per_seed[h][9])
                    agv_utilizations.append(makespan_per_seed[h][10])

                    load_unload_AGVs_list.append(makespan_per_seed[h][11])

            results.append(list(np.nanmean(final_result, axis=0)))
            average_ma_utilization.append(np.nanmean(ma_utilizations, axis=0))
            average_agv_utilization.append(np.nanmean(agv_utilizations, axis=0))
            average_load_unload_AGVs_list.append(np.nanmean(load_unload_AGVs_list, axis=0))

            # Print utilizations Machine Agents and AGV agents
            for wc in range(len(machinesPerWC)):
                print("\n", " Work Center", wc)
                print("--------------------")
                for ma in machine_number_WC[wc]:
                    print("Utilization Machine", ma, "=", round(average_ma_utilization[0][ma - 1] * 100, 2), "%")
                    total_average_machine_utilization.append(round(average_ma_utilization[0][ma - 1] * 100, 2))
                if simulation_parameter_1[0] == 1:
                    for agv in agv_number_WC[wc]:
                        print("Utilization AGV", agv, "=", round(average_agv_utilization[0][agv - 1] * 100, 2), "%")
                        total_average_avg_utilization.append(round(average_agv_utilization[0][agv - 1] * 100, 2))

            if simulation_parameter_1[0] == 2:
                print("\n", "AGVs")
                print("--------------------")
                for wc in range(len(machinesPerWC)):
                    for agv in agv_number_WC[wc]:
                        print("Utilization AGV", agv, "=", round(average_agv_utilization[0][agv - 1] * 100, 2), "%")
                        total_average_avg_utilization.append(round(average_agv_utilization[0][agv - 1] * 100, 2))

            # Save the results
            results = pd.DataFrame(results, columns=['Makespan', 'Mean Flow Time', 'Mean Weighted Tardiness',
                                                     'Max Weighted Tardiness', 'No. Tardy Jobs P1', 'No. Tardy Jobs P2',
                                                     'No. Tardy Jobs P3', 'Mean WIP', 'Early_Term'])
            # Print mean tardiness
            print("\n=============== Job-Shop Results ===============")
            print("Mean Tardiness:", round(results['Mean Weighted Tardiness'][0], 2))
            print("Max Tardiness:", round(results['Max Weighted Tardiness'][0], 2))
            print("No. Tardy Jobs P1:", round(results['No. Tardy Jobs P1'][0], 2))
            print("No. Tardy Jobs P2:", round(results['No. Tardy Jobs P2'][0], 2))
            print("No. Tardy Jobs P3:", round(results['No. Tardy Jobs P3'][0], 2))
            print("Mean WIP:", round(results['Mean WIP'][0], 2))
            print("Total Average Machine Utilization:",
                  round(sum(total_average_machine_utilization) / len(total_average_machine_utilization), 2), "%")
            print("Total Average Avg Utilization:",
                  round(sum(total_average_avg_utilization) / len(total_average_avg_utilization), 2), "%")
            print("Average Load Unload Ratio", round(np.mean(average_load_unload_AGVs_list), 2))

            sensitivity_str = f"Result_analysis_AGV/Runs/Run-" + "(" + \
                              str(a) + "-" + \
                              str(b) + "-" + \
                              str(c) + "-" + \
                              str(d) + ")" + ".csv"

            path = "Results/" + sim_par_1_string + "/" + sim_par_2_string + "/" + str(agvsPerWC)

            try:
                os.makedirs(path)
                # print("Directory ", path, " Created ")
            except FileExistsError:
                # print("Directory ", path, " already exists")
                pass

            file_name = f"" + path + "/Results-" + str(
                utilization[i]) + "-" + str(
                due_date_settings[i]) + ".csv"

            results.to_csv(file_name)
