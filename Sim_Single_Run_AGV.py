"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import itertools
import random
from collections import defaultdict
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import pandas as pd
import simpy
from simpy import *

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# General Settings
number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit

# Machine shop settings
# processingTimes = [[8.75, 5.75, 4.5, 9.5], [5.75, 7.0, 9.5], [5.75, 4.5, 9.75, 7.0, 7.0]]  # Longer Processing Times
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]  # Processing Times
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]  # Workcenter per operations
numberOfOperations = [4, 3, 5]  # Number of operations per job type
setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]  # Setuptypes from one job type to another
demand = [0.2, 0.5, 0.3]

# Machine information
machinesPerWC = [4, 2, 5, 3, 2]  # Number of machines per workcenter
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]  # Index of machines

# buffer cap per machine
machine_buffers_cap_WC = [[99, 99, 99, 99], [99, 99], [99, 99, 99, 99, 99], [99, 99, 99], [99, 99]]
# machine_buffers_cap_WC = [[4, 4, 4, 4], [4, 4], [4, 4, 4, 4, 4], [4, 4, 4], [4, 4]]
# machine_buffers_cap_WC = [[3, 3, 3, 3], [3, 3], [3, 3, 3, 3, 3], [3, 3, 3], [3, 3]]
# machine_buffers_cap_WC = [[2, 2, 2, 2], [2, 2], [2, 2, 2, 2, 2], [2, 2, 2], [2, 2]]
# machine_buffers_cap_WC = [[1, 1, 1, 1], [1, 1], [1, 1, 1, 1, 1], [1, 1, 1], [1, 1]]
# machine_buffers_cap_WC = [[0, 0, 0, 0], [0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0]]

# AGV information
agvsPerWC = [10, 10, 10, 10, 10]  # Number of AGVs per workcenter
agv_number_WC = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                 [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]]  # Index of AGVs

# AGV information
agvsPerWC = [6, 6, 6, 6, 6]  # Number of AGVs per workcenter
agv_number_WC = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24],
                 [25, 26, 27, 28, 29, 30]]  # Index of AGVs

"""# AGV information
agvsPerWC = [3, 3, 3, 3, 3]  # Number of AGVs per workcenter
agv_number_WC = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]  # Index of AGVs"""

"""# AGV information (No_AGVS = Machines + 1)
agvsPerWC = [5, 3, 6, 4, 3]  # Number of AGVs per workcenter
agv_number_WC = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12, 13, 14], [15, 16, 17, 18], [19, 20, 21]]  # Index of AGVs"""

"""# AGV information (No_AGVS = Machines + 2)
agvsPerWC = [6, 4, 7, 5, 4]  # Number of AGVs per workcenter
agv_number_WC = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22], [23, 24, 25, 26]]  # Index of AGVs"""

# Central buffer information
noOfCbPerWC = [1, 1, 1, 1, 1]
central_buffers_cap_WC = [[1], [1], [1], [1], [1]]  # buffer cap per central buffer

noOfWC = range(len(machinesPerWC))

# Immediate CFP release APA
immediate_release = False

# Virtual Machine QUEUE plotting
QUEUE = False

# DEBUG PRINTING
DEBUG = False

# GANNT PLOTTING
GANTT_Job = False
GANTT_Machine = False
GANTT_AGV = False
GANTT = False
if GANTT_Job or GANTT_Machine:
    GANTT = True

"Initial parameters of the GES"
noAttributes = 8
noAttributesJob = 4
noAttributesAGV = 1
totalAttributes = noAttributes + noAttributesJob


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


def longest_time_in_system(env, jobs, noOfAGVs, currentWC, job_shop, agvs, AGVstore):
    """ AGV Dispatching rule that selects the job with the longest time in the system"""

    no_of_jobs = len(jobs)
    dispatched_jobs = []
    dispatched_agvs = []
    relative_agv_numbers = []
    relative_count = 0

    available_agvs = []
    for jj in range(noOfAGVs):

        agv_res = job_shop.agv_process_per_wc[jj, currentWC - 1]

        if len(agv_res.users) == 0:
            agv = job_shop.agv_queue_per_wc[jj, currentWC - 1]
            available_agvs.append(agv)
            relative_agv_numbers.append(relative_count)

        relative_count += 1

    no_avail_agvs = len(available_agvs)

    job_in_system_times = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):
        job_in_system_time = env.now - job.arrival_time_system
        job_in_system_times[job_idx] = job_in_system_time

    for agv_count, job in enumerate(jobs):

        if no_avail_agvs == 0:
            break

        best_job = np.argmax(job_in_system_times)
        no_avail_agvs -= 1
        dispatched_jobs.append(best_job)
        dispatched_agvs.append(relative_agv_numbers[agv_count])
        job_in_system_times[best_job] = -1

    # Put the winning jobs in the AGV queues
    for ii, vv in enumerate(dispatched_jobs):

        if DEBUG:
            agv_number = agv_number_WC[currentWC - 1][dispatched_agvs[ii]]
            print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ": CFP done!", jobs[vv].name, "linked to AGV",
                  agv_number, "WC", currentWC)

        put_job_in_agv_queue(currentWC, dispatched_agvs[ii], jobs[vv], job_shop, agvs)

    # Remove job from queue of the APA
    for ii in reversed(sorted(dispatched_jobs)):

        if DEBUG:
            print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ":", jobs[ii].name, "removed from APA",
                  currentWC, "queue")

        yield AGVstore.get(lambda mm: mm == jobs[ii])


def random_agv_rule(env, jobs, noOfAGVs, currentWC, job_shop, agvs, AGVstore):  # Get the bids for all AGVs
    """ AGV Dispatching rule that selects an available AGV per job at random"""

    dispatched_jobs = []
    dispatched_agvs = []
    relative_agv_numbers = []
    relative_count = 0

    available_agvs = []
    for jj in range(noOfAGVs):

        agv_res = job_shop.agv_process_per_wc[jj, currentWC - 1]

        if len(agv_res.users) == 0:
            agv = job_shop.agv_queue_per_wc[jj, currentWC - 1]
            available_agvs.append(agv)
            relative_agv_numbers.append(relative_count)

        relative_count += 1

    no_avail_agvs = len(available_agvs)

    for job_no, job in enumerate(jobs):

        if no_avail_agvs == 0:
            break

        best_agv = random.choice(relative_agv_numbers)
        relative_agv_numbers.remove(best_agv)

        no_avail_agvs -= 1

        dispatched_jobs.append(job_no)
        dispatched_agvs.append(best_agv)

    # Put the winning jobs in the AGV queues
    for ii, vv in enumerate(dispatched_jobs):

        if DEBUG:
            agv_number = agv_number_WC[currentWC - 1][dispatched_agvs[ii]]
            print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ": CFP done!", jobs[vv].name, "linked to AGV",
                  agv_number, "WC", currentWC)

        put_job_in_agv_queue(currentWC, dispatched_agvs[ii], jobs[vv], job_shop, agvs)

    # Remove job from queue of the APA
    for ii in reversed(sorted(dispatched_jobs)):

        if DEBUG:
            print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ":", jobs[ii].name, "removed from APA",
                  currentWC, "queue")

        yield AGVstore.get(lambda mm: mm == jobs[ii])


def nearest_idle_agv(env, jobs, noOfAGVs, currentWC, job_shop, agvs, AGVstore):  # Closest AGV dispatching rule
    """ AGV Dispatching rule that calculates the total time to pick and bring a job and then
        selects the best available AGV per job"""

    no_of_jobs = len(jobs)
    dispatched_jobs = []
    dispatched_agvs = []
    relative_agv_numbers = []
    relative_count = 0

    available_agvs = []
    for jj in range(noOfAGVs):

        agv_res = job_shop.agv_process_per_wc[jj, currentWC - 1]

        if len(agv_res.users) == 0:
            agv = job_shop.agv_queue_per_wc[jj, currentWC - 1]
            available_agvs.append(agv)
            relative_agv_numbers.append(relative_count)

        relative_count += 1

    no_avail_agvs = len(available_agvs)

    driving_times_per_job = [0] * no_of_jobs

    for job_idx, job in enumerate(jobs):

        driving_times_agvs = [0] * no_avail_agvs

        for agv_idx, agv in enumerate(available_agvs):
            pick_time = job_shop.travel_time_matrix[agv[1]][job.location]
            bring_time = job_shop.travel_time_matrix[job.location][job.cfp_wc_ma_result]
            driving_time = (pick_time + bring_time)

            driving_times_agvs[agv_idx] = driving_time

        driving_times_per_job[job_idx] = driving_times_agvs

    arr = np.array(driving_times_per_job)

    for job in jobs:

        if no_avail_agvs == 0:
            break

        result = np.where(arr == np.amin(arr))

        # TODO:  check which agv to dispatch when duplicate times

        best_job = result[0][0]
        best_agv = result[1][0]

        arr[:][best_job] = float("inf")

        for job_idx, _ in enumerate(jobs):
            arr[job_idx][best_agv] = float("inf")

        no_avail_agvs -= 1

        dispatched_jobs.append(best_job)
        dispatched_agvs.append(best_agv)

    # Put the winning jobs in the AGV queues
    for ii, vv in enumerate(dispatched_jobs):

        if DEBUG:
            agv_number = agv_number_WC[currentWC - 1][relative_agv_numbers[dispatched_agvs[ii]]]
            print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ": CFP done!", jobs[vv].name, "linked to AGV",
                  agv_number, "WC", currentWC)

        put_job_in_agv_queue(currentWC, relative_agv_numbers[dispatched_agvs[ii]], jobs[vv], job_shop, agvs)

    # Remove job from queue of the APA
    for ii in reversed(sorted(dispatched_jobs)):

        if DEBUG:
            print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ":", jobs[ii].name, "removed from APA",
                  currentWC, "queue")

        yield AGVstore.get(lambda mm: mm == jobs[ii])


def bid_winner_agv(env, jobs, noOfAGVs, currentWC, job_shop, agvs, AGVstore,
                   normaliziation_range):
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

        new_bid = [0] * no_of_jobs
        for ii, job in enumerate(jobs):
            attributes = bid_calculation_agv()
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

        if DEBUG:
            agv_number = agv_number_WC[currentWC - 1][best_bid[ii]]
            print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ": CFP done!", jobs[vv].name, "linked to AGV",
                  agv_number, "WC", currentWC)

        put_job_in_agv_queue(currentWC, best_bid[ii], jobs[vv], job_shop, agvs)

    # Remove job from queue of the APA
    for ii in reversed(best_job):

        if DEBUG:
            print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ":", jobs[ii].name, "removed from APA", currentWC,
                  "queue")

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
        for ii, job in enumerate(jobs):
            attributes = bid_calculation_ma(job_shop.test_weights, machine_number_WC[currentWC - 1][jj],
                                            job.processingTime[job.currentOperation - 1], job.currentOperation,
                                            total_rp[ii], job.dueDate[job.numberOfOperations],
                                            env.now,
                                            job.priority, queue_length, normaliziation_range)

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

    # Put the job in the AGV agent pool queue and link the winning machines with the jobs
    for ii, vv in enumerate(best_job):

        # Put job in APA
        AGVstore = job_shop.AGVstoreWC[currentWC - 1]
        AGVstore.put(jobs[vv])
        machine_loc = (best_bid[ii], currentWC - 1)
        jobs[vv].cfp_wc_ma_result = machine_loc

        put_job_in_ma_queue(currentWC, best_bid[ii], jobs[vv], job_shop, machine)

        if DEBUG:
            ma_number = machine_number_WC[currentWC - 1][best_bid[ii]]
            print("CT:", round(env.now, 3), "-", "JPA WC", currentWC, ": CFP done!", jobs[vv].name,
                  "will be processed on MA", ma_number, "WC",
                  currentWC, machine_loc)
            print("CT:", round(env.now, 3), "-", "JPA WC", currentWC, ": Job stored in APA", currentWC, "queue")

    # Trigger the APA that there is a Job
    if not job_shop.condition_flag_CFP_AGV[currentWC - 1].triggered:
        job_shop.condition_flag_CFP_AGV[currentWC - 1].succeed()

    # Remove job from queue of the JPA
    for ii in reversed(best_job):

        if DEBUG:
            print("CT:", round(env.now, 3), "-", "JPA WC", currentWC, ":", jobs[ii].name, "removed from JPA", currentWC,
                  "queue")

        yield store.get(lambda mm: mm == jobs[ii])


def bid_calculation_agv():
    """Calulcates the bidding value of a job for AGVS."""
    attribute = [0] * noAttributesAGV
    attribute[0] = 0
    #attribute[0] = random.uniform(0, 2)  # At this moment the choice of AGVs is random

    return sum(attribute)


def bid_calculation_ma(weights_new, machinenumber, processing_time,
                       current, total_rp, due_date, now, job_priority, queue_length,
                       normalization):
    """Calulcates the bidding value of a job for MAs."""
    attribute = [0] * noAttributes
    attribute[0] = processing_time / 8.75 * weights_new[machinenumber - 1][0]  # processing time
    attribute[1] = (current - 1) / (5 - 1) * weights_new[machinenumber - 1][1]  # remaing operations
    attribute[2] = (due_date - now - normalization[0]) / (normalization[1] - normalization[0]) * \
                   weights_new[machinenumber - 1][2]  # slack
    attribute[3] = total_rp / 21.25 * weights_new[machinenumber - 1][3]  # remaining processing time
    attribute[4] = (((due_date - now) / total_rp) - normalization[2]) / (normalization[3] - normalization[2]) * \
                   weights_new[machinenumber - 1][4]  # Critical Ratio
    attribute[5] = (job_priority - 1) / (10 - 1) * weights_new[machinenumber - 1][5]  # Job Priority
    attribute[6] = queue_length / 25 * weights_new[machinenumber - 1][6]  # Queue length
    attribute[7] = 0

    return sum(attribute)


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

        # finished_job += 1
        if job.number > max_job:
            if np.count_nonzero(job_shop.flowtime[min_job:max_job]) == 2000:

                job_shop.finish_time = env.now
                job_shop.end_event.succeed()

        if job_shop.WIP > max_wip:
            print("To much WIP")

        if env.now > 10_000:
            print("Time eslaped")


        if (job_shop.WIP > max_wip) | (env.now > 10_000):
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


def choose_job_queue_ma(weights_new_job, machinenumber, processing_time, due_date, env,
                        setup_time, job_priority, normalization):
    """Calculates prioirities of jobs in a machines queue"""
    attribute_job = [0] * noAttributesJob

    attribute_job[3] = 0
    attribute_job[2] = setup_time / 1.25 * weights_new_job[machinenumber - 1][noAttributes + 2]
    attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    attribute_job[0] = (due_date - processing_time - setup_time - env.now - normalization[4]) / (
            normalization[5] - normalization[4]) * \
                       weights_new_job[machinenumber - 1][noAttributes]

    return sum(attribute_job)


def choose_job_queue_agv(job):
    """Calculates prioirities of jobs in an agv queue"""

    attribute_job = [0] * 1

    if job.location != "depot":
        attribute_job[0] = 10

    elif job.location == "depot":
        attribute_job[0] = 1

    return sum(attribute_job)


def agv_processing(job_shop, currentWC, agv_number, env, weights_new, agv, normalization, agv_res, agv_buf):
    """This refers to a AGV Agent in the system. It checks which jobs it wants to transfer
        next to machines and stores relevant information regarding it."""

    while True:

        relative_agv = agv_number_WC[currentWC - 1].index(agv_number)

        if agv[0].items:

            agv_location = agv[1]
            priority_list = []

            for job in agv[0].items:
                job_queue_priority = choose_job_queue_agv(job)  # Calulate the job priorities
                priority_list.append(job_queue_priority)

            ind_processing_job = priority_list.index(max(priority_list))  # Get the job with the highest value

            # Remember job and job destination
            next_job = agv[0].items[ind_processing_job]
            job_destination = next_job.cfp_wc_ma_result

            if agv_location != next_job.location:

                if DEBUG:
                    print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", currentWC, "at", agv_location,
                          ": I will pick", next_job.name,
                          "which is at location", next_job.location)

                driving_time = job_shop.travel_time_matrix[agv_location][next_job.location]
                yield env.process(machine_proc_res(agv_res, driving_time, env))

                # Change AGV location
                agv[1] = next_job.location
                agv_location = agv[1]

            #  Load the job on the AGV and remove job from depot or machine
            if next_job in job_shop.depot_queue.items:
                job_shop.depot_queue.items.remove(next_job)

            elif next_job in job_shop.machine_buffer_per_wc[agv_location][0].items:

                if QUEUE:
                    # Register queue length
                    ma_number = machine_number_WC[agv_location[1]][agv_location[0]]
                    job_shop.update_ma_queue(env.now, agv_location[1], ma_number, -1)

                # TODO: Yield the machine if still busy

                job_shop.machine_buffer_per_wc[agv_location][0].items.remove(next_job)

            # Put job on AGV buffer
            agv_buf.put(next_job)

            if not next_job.cfp_wc_ma_result == "depot":
                job_shop.machine_buffer_per_wc[next_job.cfp_wc_ma_result][1].put(next_job)

            if DEBUG:
                if not next_job.cfp_wc_ma_result == "depot":
                    ma_number = machine_number_WC[next_job.cfp_wc_ma_result[1]][next_job.cfp_wc_ma_result[0]]
                    print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", currentWC, "at", agv_location, ":",
                          next_job.name, "picked up!")
                    print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", currentWC, "at", agv_location,
                          ": I will bring", next_job.name,
                          "to MA", ma_number, "WC", currentWC, next_job.cfp_wc_ma_result)

                else:
                    print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", currentWC, "at", agv_location, ":",
                          next_job.name, "picked up!")
                    print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", currentWC, "at", agv_location,
                          ": I will bring", next_job.name,
                          "to depot")

            driving_time = job_shop.travel_time_matrix[agv_location][job_destination]
            yield env.process(agv_proc_res(agv_res, driving_time, env))

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
                    job_shop.update_ma_queue(env.now, agv_location[1], ma_number, 1)

                machine_buf = job_shop.machine_buffer_per_wc[agv_location][1]
                machine_buf.items.remove(next_job)
                machine_buf = job_shop.machine_buffer_per_wc[agv_location][0]
                machine_buf.put(next_job)

                #  Unload the job from the AGV and put job on machine, machine buffer or depot
                if not next_job.job_loaded_condition.triggered and next_job in machine_buf.items:
                    ma_number = machine_number_WC[agv_location[1]][agv_location[0]]

                    if DEBUG:
                        print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", currentWC, "at", agv_location,
                              ":", next_job.name, "loaded on MA", ma_number, "WC", currentWC, agv_location)
                    next_job.job_loaded_condition.succeed()

            else:
                if DEBUG:
                    print("CT:", round(env.now, 3), "-", "AGV", agv_number, "WC", currentWC, "at", agv_location, ":",
                          next_job.name, "dropped at depot")
                job_shop.depot_queue.put(next_job)

            # Trigger the APA that there is an idle AGV
            if not job_shop.condition_flag_CFP_AGV[currentWC - 1].triggered:
                job_shop.condition_flag_CFP_AGV[currentWC - 1].succeed()


        else:
            yield job_shop.condition_flag_agv[
                (relative_agv, currentWC - 1)]  # Used if there is currently no job in the agv queue
            job_shop.condition_flag_agv[(relative_agv, currentWC - 1)] = simpy.Event(env)  # Reset event if it is used


def agv_proc_res(agv_resource, driving_time, env):
    with agv_resource.request() as req:
        yield req
        yield env.timeout(driving_time)


def machine_processing(job_shop, current_WC, machine_number, env, weights_new, last_job, machine,
                       makespan, min_job, max_job, normalization, max_wip, machine_res, machine_buf):
    """This refers to a Machine Agent in the system. It checks which jobs it wants to process
    next and stores relevant information regarding it."""
    while True:

        relative_machine = machine_number_WC[current_WC - 1].index(machine_number)

        if machine.items:
            setup_time = []
            priority_list = []
            if not last_job[relative_machine]:  # Only for the first job
                ind_processing_job = 0
                setup_time.append(0)
            else:
                for job in machine.items:
                    setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
                    job_queue_priority = choose_job_queue_ma(weights_new, machine_number,
                                                             job.processingTime[job.currentOperation - 1],
                                                             job.dueDate[job.currentOperation], env, setuptime,
                                                             job.priority, normalization)  # Calulate the job priorities
                    priority_list.append(job_queue_priority)
                    setup_time.append(setuptime)
                ind_processing_job = priority_list.index(max(priority_list))  # Get the job with the highest value

            next_job = machine.items[ind_processing_job]
            ma_number = machine_number_WC[current_WC - 1][next_job.cfp_wc_ma_result[0]]

            # If job is not at machine, machine will be yielded until AGVs loads job
            if next_job not in machine_buf[0].items:
                if DEBUG:
                    print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", current_WC, next_job.name,
                          "not at machine")

                yield next_job.job_loaded_condition

                if DEBUG:
                    print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", current_WC, next_job.name,
                          "at machine")

            # Always reset the job loaded condition
            next_job.job_loaded_condition = simpy.Event(env)

            setuptime = setup_time[ind_processing_job]
            time_in_processing = next_job.processingTime[
                                     next_job.currentOperation - 1] + setuptime  # Total time the machine needs to process the job

            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, env, setuptime)
            job_shop.utilization[(relative_machine, current_WC - 1)] = job_shop.utilization[(
                relative_machine, current_WC - 1)] + setuptime + next_job.processingTime[next_job.currentOperation - 1]
            last_job[relative_machine] = next_job.type

            machine.items.remove(next_job)  # Remove job from queue

            if DEBUG:
                print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", current_WC, ": Start processing",
                      next_job.name)

            if GANTT:
                job_shop.update_gantt(time_in_processing, env.now + time_in_processing,
                                      next_job.name, "MA_" + str(machine_number), env.now)

            yield env.process(machine_proc_res(machine_res, time_in_processing, env))

            if DEBUG:
                print("CT:", round(env.now, 3), "-", "MA", ma_number, "WC", current_WC, ": Finished processing of",
                      next_job.name)

            next_job.cfp_wc_ma_result = None
            next_job.cfp_wc_agv_result = None

            next_workstation(next_job, job_shop, env, min_job, max_job, max_wip)  # Send the job to the next workstation

        else:
            yield job_shop.condition_flag_ma[
                (relative_machine, current_WC - 1)]  # Used if there is currently no job in the machines queue
            job_shop.condition_flag_ma[(relative_machine, current_WC - 1)] = simpy.Event(
                env)  # Reset event if it is used


def machine_proc_res(machine_resource, time_in_processing, env):
    with machine_resource.request() as req:
        yield req
        yield env.timeout(time_in_processing)


def cfp_wc_ma(env, machine, store, job_shop, currentWC, normalization):
    """Sends out the Call-For-Proposals to the various machines.
    Represents the Job-Pool_agent"""
    while True:

        if store.items:

            if DEBUG:
                print("CT:", round(env.now, 3), "-", "JPA WC", currentWC, ": Sended CFPs to MAs!")

            job_shop.QueuesWC[currentWC - 1].append(
                {ii: len(job_shop.machine_queue_per_wc[(ii, currentWC - 1)].items) for ii in
                 range(machinesPerWC[currentWC - 1])})  # Stores the Queue length of the JPA

            c = bid_winner_ma(env, store.items, machinesPerWC[currentWC - 1], currentWC, job_shop,
                              machine, store, normalization)

            env.process(c)

        tib = 0.5  # Frequency of when CFPs are sent out
        yield env.timeout(tib)


def cfp_wc_agv(env, agvs, AGVstore, job_shop, currentWC, normalization, AGV_rule):
    """Sends out the Call-For-Proposals to the various AGVs.
        Represents the AGV-Pool_agent"""

    while True:

        if AGVstore.items:

            job_list = AGVstore.items

            if DEBUG:
                print("CT:", round(env.now, 3), "-", "APA WC", currentWC, ": Sended CFPs to AGVs!")

            job_shop.AGVQueuesWC[currentWC - 1].append(
                {ii: len(job_shop.agv_queue_per_wc[(ii, currentWC - 1)][0].items) for ii in
                 range(agvsPerWC[currentWC - 1])})

            if AGV_rule == 1:
                c = bid_winner_agv(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop,
                                   agvs, AGVstore, normalization)

                env.process(c)

            if AGV_rule == 2:
                c = nearest_idle_agv(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop, agvs, AGVstore)
                env.process(c)

            if AGV_rule == 3:
                c = random_agv_rule(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop, agvs, AGVstore)
                env.process(c)

            if AGV_rule == 4:
                c = longest_time_in_system(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop, agvs, AGVstore)
                env.process(c)



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

        if DEBUG:
            print("CT:", round(env.now, 3), "-", job.name, "entered system ( type", job.type, ")",
                  "and will be processed first at WC:", firstWC)

        tib = random.expovariate(1.0 / interval)
        yield env.timeout(tib)


def create_distance_matrix():
    """Creates distance matrix where distance can be requested by inputting:
    distance_maxtrix[actual location][destination]"""


    # All distances are in meters
    distance_matrix = {"depot": {(ii, jj): random.uniform(0, 2.0) for jj in noOfWC for ii in range(machinesPerWC[jj])}}

    distance_matrix["depot"].update(
        {(ii + machinesPerWC[jj], jj): random.uniform(0, 2.0) for jj in noOfWC for ii in range(noOfCbPerWC[jj])})

    distance_matrix["depot"].update({"depot": 0})

    # TODO: Random dinstance_matrix has to be placed before the simpy enviroment as a static dictionary

    for jj in noOfWC:
        for ii in range(machinesPerWC[jj]):
            distance_matrix[(ii, jj)] = {(ii, jj): random.uniform(0, 2.0) for jj in noOfWC for ii in
                                         range(machinesPerWC[jj])}

            distance_matrix[(ii, jj)].update(
                {(ii + machinesPerWC[jj], jj): 1 for jj in noOfWC for ii in range(noOfCbPerWC[jj])})
            distance_matrix[(ii, jj)].update({"depot": random.uniform(0, 2.0)})

            distance_matrix[ii, jj][ii, jj] = 0

    for jj in noOfWC:
        for ii in range(noOfCbPerWC[jj]):

            ii += machinesPerWC[jj]
            distance_matrix[(ii, jj)] = {(ii, jj): random.uniform(0, 2.0) for jj in noOfWC for ii in
                                         range(machinesPerWC[jj])}

            distance_matrix[(ii, jj)].update({"depot": random.uniform(0, 2.0)})

            distance_matrix[(ii, jj)].update({(ii + machinesPerWC[jj], jj): random.uniform(0, 2.0) for jj in
                                              noOfWC for ii in range(noOfCbPerWC[jj])})

            distance_matrix[ii, jj][ii, jj] = 0

    return distance_matrix


class jobShop:
    """This class creates a job shop, along with everything that is needed to run the Simpy Environment."""

    def __init__(self, env, weights_new, travel_time_matrix):

        # TODO: Can we bundle the Machine queue, Machine buffer and Machine resource?
        # Virtual machine queue, phyical machine buffer capacity + jobs underway and machine resource
        self.machine_queue_per_wc = {(ii, jj): Store(env) for jj in noOfWC for ii in range(machinesPerWC[jj])}

        self.machine_buffer_per_wc = {(ii, jj): [Store(env, capacity=1 + machine_buffers_cap_WC[jj][ii]),
                                                 Store(env)] for jj in noOfWC for ii in
                                      range(machinesPerWC[jj])}

        self.machine_process_per_wc = {(ii, jj): Resource(env, capacity=1) for jj in noOfWC
                                       for ii in range(machinesPerWC[jj])}

        # TODO: Can we bundle the AGV queue, AGV buffer and AGV resource?
        # Virtual agv queue and phyical resource + location
        self.agv_queue_per_wc = {(ii, jj): [Store(env), "depot"] for jj in noOfWC for ii in range(agvsPerWC[jj])}

        self.agv_buffer_per_wc = {(ii, jj): Store(env, capacity=1) for jj in noOfWC for ii in
                                  range(agvsPerWC[jj])}

        self.agv_process_per_wc = {(ii, jj): Resource(env, capacity=1) for jj in noOfWC for ii in
                                   range(agvsPerWC[jj])}

        # Central buffers physical buffer capacity
        self.central_buffers = {(ii + machinesPerWC[jj], jj): [Resource(env, capacity=central_buffers_cap_WC[jj][ii])]
                                for jj in noOfWC for ii in range(noOfCbPerWC[jj])}

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

        # TODO: Nog note toevoegen
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

        self.condition_flag_waiting_job = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(agvsPerWC[jj])}
        # An event which keeps track if a machine is waiting on a job to be brought by agv

        self.pending_agvs = {}

        self.test_weights = weights_new
        self.flowtime = []  # Jobs flowtime
        self.tardiness = []  # Jobs tardiness
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

    def update_gantt(self, duration, finish_time, job, machine, start_time):
        gantt_dict = {'Duration': duration,
                      'Finish': finish_time,
                      'Job': job,
                      'Machine': machine,
                      'Start': start_time}

        self.gantt_list_ma.append(gantt_dict)

    def update_ma_queue(self, time, current_wc, machine_nr, amount):

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

    bar_style = {'alpha': 1.0, 'lw': 25, 'solid_capstyle': 'butt'}
    text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center'}
    colors = mpl.cm.Dark2.colors

    schedule.sort_values(by=['Job', 'Start'])
    schedule.set_index(['Job', 'Machine'], inplace=True)

    fig = plt.figure(figsize=(24, 5 + (len(JOBS) + len(MACHINES)) / 4))
    # fig = plt.figure(figsize=(24, 20))
    ax = fig.add_subplot()

    for jdx, j in enumerate(JOBS, 1):
        for mdx, m in enumerate(MACHINES, 1):
            if (j, m) in schedule.index:
                xs = schedule.loc[(j, m), 'Start']
                xf = schedule.loc[(j, m), 'Finish']
                ax.plot([xs, xf], [mdx] * 2, c=colors[jdx % 7], **bar_style)
                ax.text((xs + xf) / 2, mdx, j, **text_style)

    if GANTT_type == "Machine":
        ax.set_title('Machine Schedule')
        ax.set_ylabel('Machine')

    if GANTT_type == "Job":
        ax.set_title('Job Schedule')
        ax.set_ylabel('Job')

    if GANTT_type == "AGV":
        ax.set_title('AGV Schedule')
        ax.set_ylabel('Job')

    for idx, s in enumerate([JOBS, MACHINES]):
        ax.set_ylim(0.5, len(s) + 0.5)
        ax.set_yticks(range(1, 1 + len(s)))
        ax.set_yticklabels(s)
        ax.text(makespan, ax.get_ylim()[0] - 0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        ax.plot([makespan] * 2, ax.get_ylim(), 'r--')
        ax.set_xlabel('Time')
        ax.grid(True)

    # fig.tight_layout()
    plt.show()


class New_Job:
    """ This class is used to create a new job. It contains information
    such as processing time, due date, number of operations etc."""

    def __init__(self, name, env, number1, dueDateTightness):
        jobType = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)
        jobWeight = random.choices([1, 3, 10], weights=[0.5, 0.3, 0.2], k=1)
        self.type = jobType[0]
        self.priority = jobWeight[0]
        self.number = number1
        self.name = name

        self.location = "depot"
        self.cfp_wc_ma_result = None
        self.cfp_wc_agv_result = None

        self.job_loaded_condition = simpy.Event(env)

        self.currentOperation = 1
        self.processingTime = np.zeros(numberOfOperations[self.type - 1])
        self.dueDate = np.zeros(numberOfOperations[self.type - 1] + 1)
        self.dueDate[0] = env.now

        self.arrival_time_system = env.now

        self.operationOrder = operationOrder[self.type - 1]
        self.numberOfOperations = numberOfOperations[self.type - 1]
        ddt = random.uniform(dueDateTightness, dueDateTightness + 5)
        for ii in range(self.numberOfOperations):
            meanPT = processingTimes[self.type - 1][ii]
            self.processingTime[ii] = meanPT
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * ddt


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

    print("Early termination", job_shop.early_termination)
    print("Tardy jobs prio 1", no_tardy_jobs_p1)
    print("Tardy jobs prio 2", no_tardy_jobs_p2)
    print("Tardy jobs prio 3", no_tardy_jobs_p3)
    print("Flow time", flow_time)
    print("Mean Tardiness", mean_tardiness)

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1 / total_p1, no_tardy_jobs_p2 / total_p2, no_tardy_jobs_p3 / total_p2, mean_WIP, early_term


def do_simulation_with_weights(mean_weight_new, arrivalMean, due_date_tightness, min_job, max_job,
                               normalization, normalization_AGV, max_wip, AGV_rule, iter1):
    """ This runs a single simulation"""

    travel_time_matrix = create_distance_matrix()

    random.seed(iter1)

    print(random.uniform(1,2))


    env = Environment()  # Create Environment
    job_shop = jobShop(env, mean_weight_new, travel_time_matrix)  # Initiate the job shop
    env.process(source(env, 0, arrivalMean, job_shop, due_date_tightness,
                       min_job))  # Starts the source (Job Release Agent)

    for wc in range(len(machinesPerWC)):

        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        MAstoreWC = job_shop.MAstoreWC[wc]
        AGVstoreWC = job_shop.AGVstoreWC[wc]

        env.process(
            cfp_wc_ma(env, job_shop.machine_queue_per_wc, MAstoreWC, job_shop, wc + 1, normalization))

        env.process(
            cfp_wc_agv(env, job_shop.agv_queue_per_wc, AGVstoreWC, job_shop, wc + 1, normalization_AGV, AGV_rule))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_queue_per_wc[(ii, wc)]
            machine_buf = job_shop.machine_buffer_per_wc[(ii, wc)]
            machine_res = job_shop.machine_process_per_wc[(ii, wc)]
            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
                                   machine, makespanWC, min_job, max_job, normalization, max_wip, machine_res,
                                   machine_buf))

        for ii in range(agvsPerWC[wc]):
            agv = job_shop.agv_queue_per_wc[(ii, wc)]
            agv_buf = job_shop.agv_buffer_per_wc[(ii, wc)]
            agv_res = job_shop.agv_process_per_wc[(ii, wc)]
            # TODO: weight for AGVs must still be learned
            env.process(agv_processing(job_shop, wc + 1, agv_number_WC[wc][ii], env, mean_weight_new,
                                       agv, normalization_AGV, agv_res, agv_buf))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)  # Run the simulation until the end event gets triggered

    if GANTT_Machine:
        visualize(job_shop.gantt_list_ma, 'Machine')

    if GANTT_Job:
        visualize(job_shop.gantt_list_ma, 'Job')

    if GANTT_AGV:
        visualize(job_shop.gantt_list_agv, "AGV")

    if QUEUE:
        MA_queue_length_plot(job_shop.QueuesMAs, job_shop.QueueTimes)

    makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, mean_WIP, early_term = get_objectives(
        job_shop, min_job, max_job, job_shop.early_termination)  # Gather all results

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, mean_WIP, early_term


if __name__ == '__main__':

    min_jobs = [499, 499, 999, 999, 1499, 1499]  # Minimum number of jobs in order te reach steady state
    max_jobs = [2499, 2499, 2999, 2999, 3499, 3499]  # Maximum number of jobs to collect information from
    wip_max = [150, 150, 200, 200, 300, 300]  # Maximum WIP allowed in the system


    arrival_time = [1.5429, 1.5429, 1.4572, 1.4572, 1.3804, 1.3804]
    utilization = [85, 85, 90, 90, 95, 95]
    #utilization = [85]

    due_date_settings = [4, 6, 4, 6, 4, 6]

    normaliziation = [[-75, 150, -8, 12, -75, 150],
                      [-30, 150, -3, 12, -30, 150],
                      [-200, 150, -15, 12, -200, 150],
                      [-75, 150, -6, 12, -75, 150],
                      [-300, 150, -35, 12, -300, 150],
                      [-150, 150, -15, 12, -150, 150]]  # Normalization ranges needed for the bidding

    normalization_AGV = [[],
                         [],
                         [],
                         [],
                         [],
                         []]

    final_obj = []
    final_std = []

    no_runs = 1
    no_processes = 1  # Change dependent on number of threads computer has, be sure to leave 1 thread remaining
    final_result = np.zeros((no_runs, 9))
    results = []

    # AGV rules:
    # 1: Linear bidding auction
    # 2: Nearest idle AGV dispatching rule
    # 3: Random AGV dispatching rule
    # 4: Longest Time in System dispatching rule
    AGV_rule = 2

    for i in range(len(utilization)):
        str1 = "Runs/Final_runs/Run-weights-" + str(utilization[i]) + "-" + str(due_date_settings[i]) + ".csv"
        df = pd.read_csv(str1, header=None)
        weights = df.values.tolist()

        print("Current run is: " + str(utilization[i]) + "-" + str(due_date_settings[i]))
        obj = np.zeros(no_runs)
        for j in range(int(no_runs / no_processes)):

            jobshop_pool = Pool(processes=no_processes)
            seeds = range(j * no_processes, j * no_processes + no_processes)
            func1 = partial(do_simulation_with_weights, weights, arrival_time[i], due_date_settings[i],
                            min_jobs[i], max_jobs[i], normaliziation[i], normalization_AGV[i], wip_max[i], AGV_rule)
            makespan_per_seed = jobshop_pool.map(func1, seeds)
            # print(makespan_per_seed)
            for h, o in itertools.product(range(no_processes), range(9)):
                final_result[h + j * no_processes][o] = makespan_per_seed[h][o]
                # print(final_result)

        results.append(list(np.mean(final_result, axis=0)))

    results = pd.DataFrame(results,
                           columns=['Makespan', 'Mean Flow Time', 'Mean Weighted Tardiness', 'Max Weighted Tardiness',
                                    'No. Tardy Jobs P1', 'No. Tardy Jobs P2', 'No. Tardy Jobs P3', 'Mean WIP',
                                    'Early_Term'])
    file_name = f"Results/Custom_1.csv"
    results.to_csv(file_name)
