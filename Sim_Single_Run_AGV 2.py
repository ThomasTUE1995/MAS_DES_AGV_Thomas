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
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]  # Processing Times
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]  # Workcenter per operations
numberOfOperations = [4, 3, 5]  # Number of operations per job type

setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]  # Setuptypes from one job type to another
demand = [0.2, 0.5, 0.3]

# Machine information
machinesPerWC = [4, 2, 5, 3, 2]  # Number of machines per workcenter
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]  # Index of machines

machine_buffers_cap_WC = [[5, 5, 5, 5], [5, 5], [5, 5, 5, 5, 5], [5, 5, 5], [5, 5]]  # buffer cap per machine
#machine_buffers_cap_WC = [[4, 4, 4, 4], [4, 4], [4, 4, 4, 4, 4], [4, 4, 4], [4, 4]]  # buffer cap per machine
#machine_buffers_cap_WC = [[3, 3, 3, 3], [3, 3], [3, 3, 3, 3, 3], [3, 3, 3], [3, 3]]  # buffer cap per machine
machine_buffers_cap_WC = [[2, 2, 2, 2], [2, 2], [2, 2, 2, 2, 2], [2, 2, 2], [2, 2]]  # buffer cap per machine
#machine_buffers_cap_WC = [[1, 1, 1, 1], [1, 1], [1, 1, 1, 1, 1], [1, 1, 1], [1, 1]]  # buffer cap per machine
#machine_buffers_cap_WC = [[0, 0, 0, 0], [0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0]]  # buffer cap per machine

# AGV information
agvsPerWC = [5, 5, 5, 5, 5]  # Number of AGVs per workcenter
agv_number_WC = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
                 [21, 22, 23, 24, 25]]  # Index of AGVs

#agvsPerWC = [10, 10, 10, 10, 10]  # Number of AGVs per workcenter
#agv_number_WC = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25,
#                  26 ,27 ,28 ,29, 30], [31, 32, 33, 34, 35, 36, 37, 38, 39, 40], [41, 42, 43, 44, 45,
#                 46, 47, 48, 49 ,50]]  # Index of AGVs

# Central buffer information
noOfCbPerWC = [1, 1, 1, 1, 1]
central_buffers_cap_WC = [[0], [0], [0], [0], [4]]  # buffer cap per central buffer

noOfWC = range(len(machinesPerWC))

DEBUG = False
DEBUG2 = False

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


def bid_winner_agv(env, jobs, noOfAGVs, currentWC, job_shop, agv, AGVstore,
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

    # Get the bids for all AGVs
    for jj in range(noOfAGVs):

        # TODO: Are we going to use a queue length for AGVs?
        # At this moment AGVs have a dedicated queue length.
        # If we want to work without a AGV queue we have to exclude AGVs from the bidding

        new_bid = [0] * no_of_jobs

        # TODO: Finish AGV dispatching rules
        AGV_rule = 0

        # Closest AGV rule
        if AGV_rule == 1:
            pass

        # Random AGV rule
        elif AGV_rule == 2:
            pass

        # Linear AGV rule
        elif AGV_rule == 3:
            pass

        for ii, job in enumerate(jobs):
            attributes = bid_calculation_agv()

            # TODO: Finish attributes

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

    # Put the job in the AGV agent pool queue and link the winning agvs with the jobs
    for ii, vv in enumerate(best_job):

        jobs[vv].cfp_wc_agv_result = (best_bid[ii], currentWC)  # AGV Agent is linked with the job
        agv_number = agv_number_WC[currentWC - 1][best_bid[ii]]

        if DEBUG:
            print("APA WC", currentWC, ": CFP done!", jobs[vv].name, "linked to AGV", agv_number, "WC", currentWC)

        put_job_in_agv_queue(currentWC, best_bid[ii], jobs[vv], job_shop, agv)

    # Remove job from queue of the APA
    for ii in reversed(best_job):

        if DEBUG:
            print("APA WC", currentWC, ":", jobs[ii].name, "removed from APA", currentWC, "queue")

        yield AGVstore.get(lambda mm: mm == jobs[ii])


def bid_winner(env, jobs, noOfMachines, currentWC, job_shop, machine, store,
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
        queue_length = len(machine[(jj, currentWC - 1)][0].items)
        new_bid = [0] * no_of_jobs
        for ii, job in enumerate(jobs):
            attributes = bid_calculation(job_shop.test_weights, machine_number_WC[currentWC - 1][jj],
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

        jobs[vv].cfp_wc_ma_result = (best_bid[ii], currentWC - 1)  # Machine Agent is linked with the job

        if DEBUG:
            ma_number = machine_number_WC[currentWC - 1][best_bid[ii]]
            print("JPA WC", currentWC, ": CFP done!", jobs[vv].name, "will be processed on MA", ma_number, "WC",
                  currentWC, jobs[vv].cfp_wc_ma_result)
            print("JPA WC", currentWC, ": Job stored in APA", currentWC, "queue")

        AGVstore = job_shop.AGVstoreWC[currentWC - 1]
        AGVstore.put(jobs[vv])

    # Remove job from queue of the JPA
    for ii in reversed(best_job):

        if DEBUG:
            print("JPA WC", currentWC, ":", jobs[ii].name, "removed from JPA", currentWC, "queue")

        yield store.get(lambda mm: mm == jobs[ii])

        # TODO: What if the job(s) is/are not picked up anymore?


def bid_calculation_agv():
    attribute = [0] * noAttributesAGV
    attribute[0] = random.uniform(0, 2)  # At this moment the choice of AGVs is random

    return sum(attribute)


def bid_calculation(weights_new, machinenumber, processing_time,
                    current, total_rp, due_date, now, job_priority, queue_length,
                    normalization):
    """Calulcates the bidding value of a job."""
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


        store = job_shop.storeWC[nextWC - 1]
        store.put(job)

    else:

        currentWC = operationOrder[job.type - 1][job.currentOperation - 1]
        AGVstore = job_shop.AGVstoreWC[currentWC - 1]
        job.cfp_wc_ma_result = "depot"
        AGVstore.put(job)

        #jkds
        #exit()

        # TODO: AGVs must still pick up the finished job

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

        if (job_shop.WIP > max_wip) | (env.now > 100):

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

    """if len(agvs[(choice, currentWC - 1)][0].items) > 5:
        print("TERMINATION: to much jobs at AGV", choice, "WC", currentWC)
        exit()"""

    if not job_shop.condition_flag_agv[(choice, currentWC - 1)].triggered:
        job_shop.condition_flag_agv[(choice, currentWC - 1)].succeed()


def put_job_in_ma_queue(currentWC, choice, job, job_shop, machines):
    """Puts a job in a machine queue. Also checks if the machine is currently active
    or has a job in its queue. If not, it succeeds an event to tell the machine
    that a new job has been added to the queue."""
    machines[(choice, currentWC - 1)][0].put(job)

    # TODO: Compare with actual buffer capacity

    """if len(machines[(choice, currentWC - 1)][0].items) > 1:
        print("TERMINATION: to much jobs at MA", choice, "WC", currentWC)
        exit()"""

    if not job_shop.condition_flag[(choice, currentWC - 1)].triggered:
        job_shop.condition_flag[(choice, currentWC - 1)].succeed()


def choose_job_queue(weights_new_job, machinenumber, processing_time, due_date, env,
                     setup_time,
                     job_priority, normalization):
    """Calculates prioirities of jobs in a machines queue"""
    attribute_job = [0] * noAttributesJob

    attribute_job[3] = 0
    attribute_job[2] = setup_time / 1.25 * weights_new_job[machinenumber - 1][noAttributes + 2]
    attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    attribute_job[0] = (due_date - processing_time - setup_time - env.now - normalization[4]) / (
            normalization[5] - normalization[4]) * \
                       weights_new_job[machinenumber - 1][noAttributes]

    return sum(attribute_job)


def agv_processing(job_shop, currentWC, agv_number, env, agv, agv_location):
    """This refers to a AGV Agent in the system. It checks which jobs it wants to transfer
        next to machines and stores relevant information regarding it."""

    while True:

        relative_agv = agv_number_WC[currentWC - 1].index(agv_number)

        if agv.items:

            agv_location = job_shop.agv_per_wc[(relative_agv, currentWC - 1)][1]
            job = agv.items[0]  # First job in queue (FIFO)
            job_location = job.location
            job_destination = job.cfp_wc_ma_result


            if job.loaded_on_agv and job_destination == "depot":


                if DEBUG:

                    print("AGV", agv_number, "WC", currentWC, "at", agv_location, ": I will bring finished", job.name, "to depot")

                driving_time = job_shop.travel_time_matrix[agv_location]["depot"]

                yield env.timeout(driving_time)

                job.loaded_on_agv = False

                job.location = "depot"
                job_shop.agv_per_wc[(relative_agv, currentWC - 1)][1] = job_destination
                agv_location = job_shop.agv_per_wc[(relative_agv, currentWC - 1)][1]

                if DEBUG:

                    print("AGV", agv_number, "WC", currentWC, "at", agv_location, ":", job.name, "deliverd at depot")


                agv.items.remove(job)





            elif job.loaded_on_agv and not job_destination == "depot":  # If AGV is already loaded with job bring to destination

                if DEBUG:
                    ma_number = machine_number_WC[currentWC - 1][job.cfp_wc_ma_result[0]]
                    print("AGV", agv_number, "WC", currentWC, "at", agv_location, ": I will bring", job.name, "to MA",
                          ma_number, "WC", currentWC, job.cfp_wc_ma_result)

                driving_time = job_shop.travel_time_matrix[agv_location][job_destination]

                yield env.timeout(driving_time)



                machine = job_shop.machine_per_wc[job_destination]


                test = False
                while len(machine[2]) >= machine[1] and len(machine[3]) != 0:
                    #print("AGV can not load", job.name, "on machine")

                    test = True
                    yield env.timeout(1)

                if test:
                    test = False
                    #print("AGV heeft", job.name, "op machine geladen")

                if len(machine[2]) < machine[1]:
                    #print("Ruimte in de buffer")
                    machine[2].append(job)


                elif len(machine[3]) == 0:
                    #print("Ruimte op de machine")
                    machine[3].append(job)



                job.loaded_on_agv = False

                job.location = job_destination
                job_shop.agv_per_wc[(relative_agv, currentWC - 1)][1] = job_destination
                agv_location = job_shop.agv_per_wc[(relative_agv, currentWC - 1)][1]

                if DEBUG:
                    ma_number = machine_number_WC[currentWC - 1][job.cfp_wc_ma_result[0]]
                    print("AGV", agv_number, "WC", currentWC, "at", agv_location, ":", job.name, "loaded on MA",
                          ma_number, "WC", currentWC, job.location)

                machine_Nr = job.cfp_wc_ma_result[0]
                put_job_in_ma_queue(currentWC, machine_Nr, job, job_shop, job_shop.machine_per_wc)
                agv.items.remove(job)

            else:  # When not loaded drive to pick up location

                if DEBUG:
                    print("AGV", agv_number, "WC", currentWC, "at", agv_location, ": I will pick", job.name,
                          "which is at location", job.location)

                driving_time = job_shop.travel_time_matrix[agv_location][job.location]

                yield env.timeout(driving_time)

                # TODO: Unload and load time?

                if job_location != "depot":

                    machine = job_shop.machine_per_wc[job_location]

                    if job in machine[3]:

                        machine[3].remove(job)

                    elif job in machine[2]:

                        machine[2].remove(job)

                job.loaded_on_agv = True
                job_shop.agv_per_wc[(relative_agv, currentWC - 1)][1] = job_location
                agv_location = job_shop.agv_per_wc[(relative_agv, currentWC - 1)][1]

                if DEBUG:
                    print("AGV", agv_number, "WC", currentWC, "at", agv_location, ":", job.name, "picked up!")



                # TODO: If job has no next machine yet, AGV has to be yielded!!!!


        else:
            yield job_shop.condition_flag_agv[
                (relative_agv, currentWC - 1)]  # Used if there is currently no job in the agv queue
            job_shop.condition_flag_agv[(relative_agv, currentWC - 1)] = simpy.Event(env)  # Reset event if it is used


def machine_processing(job_shop, current_WC, machine_number, env, weights_new, last_job, machine,
                       makespan, min_job, max_job, normalization, max_wip):
    """This refers to a Machine Agent in the system. It checks which jobs it wants to process
    next and stores relevant information regarding it."""
    while True:

        relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
        if machine[0].items:

            setup_time = []
            priority_list = []
            if not last_job[relative_machine]:  # Only for the first job
                ind_processing_job = 0
                setup_time.append(0)
            else:
                for job in machine[0].items:
                    setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
                    job_queue_priority = choose_job_queue(weights_new, machine_number,
                                                          job.processingTime[job.currentOperation - 1],
                                                          job.dueDate[job.currentOperation], env, setuptime,
                                                          job.priority, normalization)  # Calulate the job priorities
                    priority_list.append(job_queue_priority)
                    setup_time.append(setuptime)
                ind_processing_job = priority_list.index(max(priority_list))  # Get the job with the highest value

            next_job = machine[0].items[ind_processing_job]
            ma_number = machine_number_WC[current_WC - 1][relative_machine]

            if DEBUG:
                print("MA", ma_number, "WC", current_WC, next_job.location, ": Start processing", next_job.name)

            setuptime = setup_time[ind_processing_job]
            time_in_processing = next_job.processingTime[
                                     next_job.currentOperation - 1] + setuptime  # Total time the machine needs to process the job

            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, env, setuptime)
            job_shop.utilization[(relative_machine, current_WC - 1)] = job_shop.utilization[(
                relative_machine, current_WC - 1)] + setuptime + next_job.processingTime[next_job.currentOperation - 1]
            last_job[relative_machine] = next_job.type

            machine[0].items.remove(next_job)  # Remove job from queue

            gantt = update_gantt(time_in_processing, env.now + time_in_processing, next_job.name,
                                       "MA_" + str(ma_number), env.now)

            job_shop.gantt_list_ma.append(gantt)

            if next_job in machine[3]:  # If job is already on machine
                #print("Do nothing")
                pass

            else:  # If choosen job is in buffer
                if not len(machine[3]) == 0:

                    move_job = machine[3][0]
                    machine[2].append(move_job)
                    machine[3].remove(move_job)

                machine[2].remove(next_job)
                machine[3].append(next_job)

            if len(machine[3]) > 1:
                print(machine)
                print("ERROR: Buffer limit exceeded")
                exit()

            yield env.timeout(time_in_processing)

            if DEBUG:
                print("MA", ma_number, "WC", current_WC, ": Finished processing of", next_job.name)



            next_workstation(next_job, job_shop, env, min_job, max_job, max_wip)  # Send the job to the next workstation

            test = False



            while len(machine[2]) >= machine[1] and next_job in machine[3]:
                #print("---------")
                #print("Machine can not load", next_job.name, "on buffer, waiting for AGV...")


                test = True
                yield env.timeout(0.25)

            if test:
                test = False
                #print("AGV gekomen voor ", next_job.name)


            if len(machine[2]) < machine[1] and next_job in machine[3]:
                #print("Machine loaded", next_job.name, "on buffer, waiting for AGV...")
                machine[3].remove(next_job)
                machine[2].append(next_job)

        else:
            yield job_shop.condition_flag[
                (relative_machine, current_WC - 1)]  # Used if there is currently no job in the machines queue
            job_shop.condition_flag[(relative_machine, current_WC - 1)] = simpy.Event(env)  # Reset event if it is used


def cfp_wc_ma(env, machine, store, job_shop, currentWC, normalization):
    """Sends out the Call-For-Proposals to the various machines.
    Represents the Job-Pool_agent"""
    while True:

        if DEBUG2:
            print("JPA WC", currentWC, ": I have", len(store.items), "jobs in the pool")

        if store.items:

            if DEBUG:
                print("JPA WC", currentWC, ": Sended CFPs to MAs!")

            job_shop.QueuesWC[currentWC - 1].append(
                {ii: len(job_shop.machine_per_wc[(ii, currentWC - 1)][0].items) for ii in
                 range(machinesPerWC[currentWC - 1])})  # Stores the Queue length of the JPA

            c = bid_winner(env, store.items, machinesPerWC[currentWC - 1], currentWC, job_shop,
                           machine, store, normalization)

            env.process(c)

        tib = 0.5  # Frequency of when CFPs are sent out
        yield env.timeout(tib)


def cfp_wc_agv(env, agv, AGVstore, job_shop, agv_manager, currentWC, normalization):
    """Sends out the Call-For-Proposals to the various AGVs.
        Represents the AGV-Pool_agent"""
    while True:

        # TODO: Implement the use of buffers

        if DEBUG2:
            print("APA WC", currentWC, ": I have", len(AGVstore.items), "jobs in the pool")

        if AGVstore.items:

            job_list = []
            for job in AGVstore.items:
                duplicates = False

                if job.cfp_wc_ma_result == "depot":
                    job_list.append(job)
                    #print("------------------ JOB HELEMAAL KLAAR ")

                    break

                elif job.cfp_wc_ma_result != "depot" and job.location != "depot":
                    job_list.append(job)
                    #print("------------------ JOB RICHTING NIEUWE MACHINE")
                    break

                else:
                    machine = job_shop.machine_per_wc[job.cfp_wc_ma_result]

                #print("----------- NIEUWE JOB VANUIT DEPOT")

                for job_dub in job_list:
                    if job_dub.cfp_wc_ma_result == job.cfp_wc_ma_result:
                        duplicates = True

                if not duplicates:
                    if len(machine[2]) + len(machine[3]) + len(machine[4]) < machine[1]:

                        job_list.append(job)

            #print(job_list)





            if len(job_list) > 0:

                if DEBUG:
                    print("APA WC", currentWC, ": Sended CFPs to AGVs!")

                job_shop.AGVQueuesWC[currentWC - 1].append(
                    {ii: len(job_shop.agv_per_wc[(ii, currentWC - 1)][0].items) for ii in
                     range(agvsPerWC[currentWC - 1])})

                c = bid_winner_agv(env, job_list, agvsPerWC[currentWC - 1], currentWC, job_shop,
                                   agv, AGVstore, normalization)

                env.process(c)

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

        # Put job in JPA
        store = job_shop.storeWC[firstWC - 1]
        store.put(job)

        if DEBUG:
            print(job.name, "entered system ( type", job.type, ")", "and will be processed first at WC:", firstWC)

        tib = random.expovariate(1.0 / interval)
        yield env.timeout(tib)


def create_distance_matrix():
    """Creates distance matrix where distance can be requested by inputting:
    distance_maxtrix[actual location][destination]"""

    # All distances are in meters
    distance_matrix = {"depot": {(ii, jj): random.uniform(0, 2.5) for jj in noOfWC for ii in range(machinesPerWC[jj])}}
    distance_matrix["depot"].update(
        {(ii + machinesPerWC[jj], jj): random.uniform(0, 2.5) for jj in noOfWC for ii in range(noOfCbPerWC[jj])})
    distance_matrix["depot"].update({"depot": 0})

    # TODO: Random dinstance_matrix has to be placed before the simpy enviroment as a static dictionary

    for jj in noOfWC:
        for ii in range(machinesPerWC[jj]):
            distance_matrix[(ii, jj)] = {(ii, jj): random.uniform(0, 2.5) for jj in noOfWC for ii in
                                         range(machinesPerWC[jj])}
            distance_matrix[(ii, jj)].update(
                {(ii + machinesPerWC[jj], jj): 1 for jj in noOfWC for ii in range(noOfCbPerWC[jj])})
            distance_matrix[(ii, jj)].update({"depot": random.uniform(0, 2.5)})
            distance_matrix[ii, jj][ii, jj] = 0

    for jj in noOfWC:
        for ii in range(noOfCbPerWC[jj]):
            ii += machinesPerWC[jj]
            distance_matrix[(ii, jj)] = {(ii, jj): random.uniform(0, 2.5) for jj in noOfWC for ii in
                                         range(machinesPerWC[jj])}
            distance_matrix[(ii, jj)].update({"depot": random.uniform(0, 2.5)})
            distance_matrix[(ii, jj)].update({(ii + machinesPerWC[jj], jj): random.uniform(0, 2.5) for jj in
                                              noOfWC for ii in range(noOfCbPerWC[jj])})
            distance_matrix[ii, jj][ii, jj] = 0

    return distance_matrix


class AGV_Manager:
    """This class creates the AGV manager which holds information about the shopfloor layout."""

    def __init__(self, env):
        pass


class jobShop:
    """This class creates a job shop, along with everything that is needed to run the Simpy Environment."""

    def __init__(self, env, weights_new):
        # NOTE: [Store, buffer capacity, jobs in buffer, job on machine, jobs under way]
        self.machine_per_wc = {(ii, jj): [Store(env), machine_buffers_cap_WC[jj][ii], [], [], []]
                               for jj in noOfWC for ii in range(machinesPerWC[jj])}
        # Used to store jobs in a machine together witch capacity and status info

        self.agv_per_wc = {(ii, jj): [Store(env), "depot"] for jj in noOfWC for ii in
                           range(agvsPerWC[jj])}  # Used to store jobs on AGVS and to remember location and status

        self.central_buffers = {(ii + machinesPerWC[jj], jj): [Store(env), central_buffers_cap_WC[jj][ii]]
                                for jj in noOfWC for ii in range(noOfCbPerWC[jj])}

        # Used to store jobs on central buffers if available

        self.storeWC = {ii: FilterStore(env) for ii in noOfWC}  # Used to store jobs in a JPA
        self.AGVstoreWC = {ii: FilterStore(env) for ii in noOfWC}  # Used to store jobs in a APA

        self.travel_time_matrix = create_distance_matrix()

        self.QueuesWC = {jj: [] for jj in noOfWC}  # Can be used to keep track of Queue Lenghts JPA
        self.AGVQueuesWC = {jj: [] for jj in noOfWC}  # Can be used to keep track of Queue Lenghts AGV requests

        self.scheduleWC = {ii: [] for ii in noOfWC}  # Used to keep track of the schedule
        self.makespanWC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                           noOfWC}  # Keeps track of the makespan of each machine
        self.last_job_WC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                            noOfWC}  # Keeps track of which job was last in the machine

        self.condition_flag = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(machinesPerWC[jj])}
        # An event which keeps track if a machine has had a job inserted into it if it previously had no job

        self.condition_flag_agv = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(agvsPerWC[jj])}
        # An event which keeps track if an agv has had a job inserted into it if it previously had no job

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

        self.gantt_list_ma = []


def update_gantt(duration, finish_time, job, machine, start_time):
    gantt_dict = {'Duration': duration,
                  'Finish': finish_time,
                  'Job': job,
                  'Machine': machine,
                  'Start': start_time}

    return gantt_dict


def visualize(results):
    schedule = pd.DataFrame(results)
    JOBS = sorted(list(schedule['Job'].unique()))
    MACHINES = sorted(list(schedule['Machine'].unique()))
    makespan = schedule['Finish'].max()

    bar_style = {'alpha': 1.0, 'lw': 25, 'solid_capstyle': 'butt'}
    text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center'}
    colors = mpl.cm.Dark2.colors

    schedule.sort_values(by=['Job', 'Start'])
    schedule.set_index(['Job', 'Machine'], inplace=True)

    fig, ax = plt.subplots(2, 1, figsize=(12, 5 + (len(JOBS) + len(MACHINES)) / 4))

    for jdx, j in enumerate(JOBS, 1):
        for mdx, m in enumerate(MACHINES, 1):
            if (j, m) in schedule.index:
                xs = schedule.loc[(j, m), 'Start']
                xf = schedule.loc[(j, m), 'Finish']
                ax[0].plot([xs, xf], [jdx] * 2, c=colors[mdx % 7], **bar_style)
                ax[0].text((xs + xf) / 2, jdx, m, **text_style)
                ax[1].plot([xs, xf], [mdx] * 2, c=colors[jdx % 7], **bar_style)
                ax[1].text((xs + xf) / 2, mdx, j, **text_style)

    ax[0].set_title('Job Schedule')
    ax[0].set_ylabel('Job')
    ax[1].set_title('Machine Schedule')
    ax[1].set_ylabel('Machine')

    for idx, s in enumerate([JOBS, MACHINES]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        ax[idx].set_yticklabels(s)
        ax[idx].text(makespan, ax[idx].get_ylim()[0] - 0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        ax[idx].plot([makespan] * 2, ax[idx].get_ylim(), 'r--')
        ax[idx].set_xlabel('Time')
        ax[idx].grid(True)

    fig.tight_layout()

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
        self.loaded_on_agv = False
        self.location = "depot"
        self.cfp_wc_ma_result = None
        self.cfp_wc_agv_result = None
        self.currentOperation = 1
        self.processingTime = np.zeros(numberOfOperations[self.type - 1])
        self.dueDate = np.zeros(numberOfOperations[self.type - 1] + 1)
        self.dueDate[0] = env.now
        self.operationOrder = operationOrder[self.type - 1]
        self.numberOfOperations = numberOfOperations[self.type - 1]
        for ii in range(self.numberOfOperations):
            meanPT = processingTimes[self.type - 1][ii]
            self.processingTime[ii] = meanPT
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * dueDateTightness


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

    # print(no_tardy_jobs_p1)
    # print(no_tardy_jobs_p2)
    # print(no_tardy_jobs_p3)

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1 / total_p1, no_tardy_jobs_p2 / total_p2, no_tardy_jobs_p3 / total_p2, mean_WIP, early_term


def do_simulation_with_weights(mean_weight_new, arrivalMean, due_date_tightness, min_job, max_job,
                               normalization, max_wip, iter1):
    print("Start simulation")

    """ This runs a single simulation"""
    random.seed(iter1)

    env = Environment()  # Create Environment
    job_shop = jobShop(env, mean_weight_new)  # Initiate the job shop
    agv_manager = AGV_Manager(env)  # Initiate the AGV manager
    env.process(source(env, 0, arrivalMean, job_shop, due_date_tightness,
                       min_job))  # Starts the source (Job Release Agent)

    for wc in range(len(machinesPerWC)):

        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        store = job_shop.storeWC[wc]

        AGVstore = job_shop.AGVstoreWC[wc]

        env.process(
            cfp_wc_ma(env, job_shop.machine_per_wc, store, job_shop, wc + 1, normalization))

        env.process(
            cfp_wc_agv(env, job_shop.agv_per_wc, AGVstore, job_shop, agv_manager, wc + 1, normalization_AGV))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_per_wc[(ii, wc)]
            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
                                   machine, makespanWC, min_job, max_job, normalization, max_wip))

        for ii in range(agvsPerWC[wc]):
            agv = job_shop.agv_per_wc[(ii, wc)][0]
            agv_location = job_shop.agv_per_wc[(ii, wc)][1]
            env.process(agv_processing(job_shop, wc + 1, agv_number_WC[wc][ii], env, agv, agv_location))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)  # Run the simulation until the end event gets triggered

    visualize(job_shop.gantt_list_ma)

    makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, mean_WIP, early_term = get_objectives(
        job_shop, min_job, max_job, job_shop.early_termination)  # Gather all results

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, mean_WIP, early_term


if __name__ == '__main__':

    min_jobs = [1, 499, 999, 999, 1499, 1499]  # Minimum number of jobs in order te reach steady state
    max_jobs = [2499, 2499, 2999, 2999, 3499, 3499]  # Maximum number of jobs to collect information from
    wip_max = [150, 150, 200, 200, 300, 300]  # Maximum WIP allowed in the system

    arrival_time = [1.5429, 1.5429, 1.4572, 1.4572, 1.3804, 1.3804]
    # utilization = [85, 85, 90, 90, 95, 95]
    utilization = [85]
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

    for i in range(len(utilization)):
        str1 = "Runs/Final_Runs/Run-weights-" + str(utilization[i]) + "-" + str(due_date_settings[i]) + ".csv"
        df = pd.read_csv(str1, header=None)
        weights = df.values.tolist()

        print("Current run is: " + str(utilization[i]) + "-" + str(due_date_settings[i]))
        obj = np.zeros(no_runs)
        for j in range(int(no_runs / no_processes)):

            jobshop_pool = Pool(processes=no_processes)
            seeds = range(j * no_processes, j * no_processes + no_processes)
            func1 = partial(do_simulation_with_weights, weights, arrival_time[i], due_date_settings[i],
                            min_jobs[i], max_jobs[i], normaliziation[i], wip_max[i])
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
