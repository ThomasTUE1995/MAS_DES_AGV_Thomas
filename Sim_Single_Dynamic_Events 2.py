"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import csv
import math
import random
from collections import defaultdict
from functools import partial

from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
# import matplotlib.cbook
import pandas as pd
import simpy
from simpy import *

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# General Settings
number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit
# Machine shop settings
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]
numberOfOperations = [4, 3, 5]
machinesPerWC = [4, 2, 5, 3, 2]
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]
setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]
mean_setup = [0.515, 0.306, 0.515, 0.429, 0.306]
demand = [0.2, 0.5, 0.3]
noOfWC = range(len(machinesPerWC))

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 8
noAttributesJob = 4
totalAttributes = noAttributes + noAttributesJob

no_generation = 500


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


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
        queue_length = len(machine[(jj, currentWC - 1)].items)
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

    # Put the job in the queue of the winning machine
    for ii, vv in enumerate(best_job):
        put_job_in_queue(currentWC, best_bid[ii], jobs[vv], job_shop, machine)

    # Remove job from queue of the JPA
    for ii in reversed(best_job):
        yield store.get(lambda mm: mm == jobs[ii])


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

        if (job_shop.WIP > max_wip) | (env.now > 7_000):
            job_shop.end_event.succeed()
            job_shop.early_termination = 1
            job_shop.finish_time = env.now


def set_makespan(current_makespan, job, env, setup_time):
    """Sets the makespan of a machine"""
    add = current_makespan + job.processingTime[job.currentOperation - 1] + setup_time

    new = env.now + job.processingTime[job.currentOperation - 1] + setup_time

    return max(add, new)


def put_job_in_queue(currentWC, choice, job, job_shop, machines):
    """Puts a job in a machine queue. Also checks if the machine is currently active
    or has a job in its queue. If not, it succeeds an event to tell the machine
    that a new job has been added to the queue."""
    machines[(choice, currentWC - 1)].put(job)
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


def machine_processing(job_shop, current_WC, machine_number, env, weights_new, last_job, machine,
                       makespan, min_job, max_job, normalization, max_wip):
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
                    job_queue_priority = choose_job_queue(weights_new, machine_number,
                                                          job.processingTime[job.currentOperation - 1],
                                                          job.dueDate[job.currentOperation], env, setuptime,
                                                          job.priority, normalization)  # Calulate the job priorities
                    priority_list.append(job_queue_priority)
                    setup_time.append(setuptime)
                ind_processing_job = priority_list.index(max(priority_list))  # Get the job with the highest value

            next_job = machine.items[ind_processing_job]

            setuptime = setup_time[ind_processing_job]

            time_in_processing = next_job.processingTime[
                                     next_job.currentOperation - 1] + setuptime  # Total time the machine needs to process the job

            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, env, setuptime)

            job_shop.utilization[(relative_machine, current_WC - 1)] = job_shop.utilization[(
                relative_machine, current_WC - 1)] + setuptime + next_job.processingTime[next_job.currentOperation - 1]

            last_job[relative_machine] = next_job.type

            machine.items.remove(next_job)  # Remove job from queue

            job_shop.scheduleMachine[(relative_machine, current_WC - 1)].append([next_job.processingTime[
                                                                                next_job.currentOperation - 1],
                                                                            setuptime])
            job_shop.job_per_machine[next_job.number][next_job.currentOperation - 1] = machine_number_WC[current_WC - 1][relative_machine]
            yield env.timeout(time_in_processing)
            next_workstation(next_job, job_shop, env, min_job, max_job, max_wip)  # Send the job to the next workstation
        else:
            yield job_shop.condition_flag[
                (relative_machine, current_WC - 1)]  # Used if there is currently no job in the machines queue
            job_shop.condition_flag[(relative_machine, current_WC - 1)] = simpy.Event(env)  # Reset event if it is used


def cfp_wc(env, machine, store, job_shop, currentWC, normalization):
    """Sends out the Call-For-Proposals to the various machines.
    Represents the Job-Pool_agent"""
    while True:
        if store.items:
            job_shop.QueuesWC[currentWC].append(
                {ii: len(job_shop.machine_per_wc[(ii, currentWC)].items) for ii in
                 range(machinesPerWC[currentWC])})  # Stores the Queue length of the JPA
            c = bid_winner(env, store.items, machinesPerWC[currentWC], currentWC + 1, job_shop,
                           machine, store, normalization)
            env.process(c)
        tib = 0.5  # Frequency of when CFPs are sent out
        yield env.timeout(tib)


def source(env, number1, interval, job_shop, due_date_setting, min_job, rush_job_perc):
    """Reflects the Job Release Agent. Samples time and then "releases" a new
    job into the system."""
    while True:  # Needed for infinite case as True refers to "until".
        ii = number1
        number1 += 1
        job = New_Job('job%02d' % ii, env, ii, due_date_setting, job_shop, rush_job_perc)
        if ii == min_job:
            job_shop.start_time = env.now  # Start counting when the minimum number of jobs have entered the system
        job_shop.tardiness.append(-1)
        job_shop.flowtime.append(0)
        job_shop.priority.append(0)
        job_shop.job_per_machine.append(np.zeros(job.numberOfOperations))
        job_shop.WIP += 1
        firstWC = operationOrder[job.type - 1][0]
        store = job_shop.storeWC[firstWC - 1]
        store.put(job)
        tib = random.expovariate(1.0 / interval)
        yield env.timeout(tib)


class jobShop:
    """This class creates a job shop, along with everything that is needed to run the Simpy Environment."""

    def __init__(self, env, weights_new):
        self.machine_per_wc = {(ii, jj): Store(env) for jj in noOfWC for ii in
                               range(machinesPerWC[jj])}  # Used to store jobs in a machine
        self.storeWC = {ii: FilterStore(env) for ii in noOfWC}  # Used to store jobs in a JPA
        self.QueuesWC = {jj: [] for jj in noOfWC}  # Can be used to keep track of Queue Lenghts
        self.scheduleWC = {ii: [] for ii in noOfWC}  # Used to keep track of the schedule
        self.makespanWC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                           noOfWC}  # Keeps track of the makespan of each machine
        self.last_job_WC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                            noOfWC}  # Keeps track of which job was last in the machine
        self.condition_flag = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(machinesPerWC[
                                                                                               jj])}  # An event which keeps track if a machine has had a job inserted into it if it previously had no job
        self.scheduleMachine = {(ii, jj): [] for jj in noOfWC for ii in
                                range(machinesPerWC[jj])}
        self.job_per_machine = []

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


class New_Job:
    def __init__(self, name, env, number1, dueDateTightness, job_shop, rush_job_perc):
        jobType = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)
        jobWeight = random.choices([1, 3, 10], weights=[0.5, 0.3, 0.2], k=1)
        self.type = jobType[0]
        self.priority = jobWeight[0]
        self.number = number1
        self.name = name
        self.currentOperation = 1
        self.processingTime = processingTimes[self.type - 1]
        self.dueDate = np.zeros(numberOfOperations[self.type - 1] + 1)
        self.dueDate[0] = env.now
        self.operationOrder = operationOrder[self.type - 1]
        self.numberOfOperations = numberOfOperations[self.type - 1]
        p = random.choices([1, 0], weights=[rush_job_perc, 1 - rush_job_perc], k=1)

        flowtime = np.mean([job_shop.flowtime[i] for i in range(self.number) if job_shop.flowtime[i] > 0])
        if math.isnan(flowtime):
            flowtime = 0

        due_date_reduction = p[0] * flowtime * 0.3
        reduction_factor = (sum(self.processingTime) * dueDateTightness - due_date_reduction) / (
                sum(self.processingTime) * dueDateTightness)
        for ii in range(self.numberOfOperations):
            self.dueDate[ii + 1] = self.dueDate[ii] + (self.processingTime[ii] * dueDateTightness) * reduction_factor


def do_simulation_with_weights(mean_weight_new, arrivalMean, due_date_tightness, normalization,
                               min_job, max_job, max_wip, rush_job_perc, iter):
    random.seed(iter)

    env = Environment()  # Create Environment
    job_shop = jobShop(env, mean_weight_new)  # Initiate the job shop
    env.process(source(env, 0, arrivalMean, job_shop, due_date_tightness,
                       min_job, 0))  # Starts the source (Job Release Agent)

    for wc in range(len(machinesPerWC)):
        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        store = job_shop.storeWC[wc]

        env.process(
            cfp_wc(env, job_shop.machine_per_wc, store, job_shop, wc, normalization))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_per_wc[(ii, wc)]

            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
                                   machine, makespanWC, min_job, max_job, normalization, max_wip))
    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)  # Run the simulation until the end event gets triggered
    objective = np.nanmean(job_shop.tardiness[min_job:max_job]) + 0.01 * max(job_shop.tardiness[min_job:max_job])
    tard = job_shop.tardiness[min_job:max_job]

    random.seed(iter)
    env = Environment()  # Create Environment
    job_shop = jobShop(env, mean_weight_new)  # Initiate the job shop
    env.process(source(env, 0, arrivalMean, job_shop, due_date_tightness,
                       min_job, rush_job_perc))  # Starts the source (Job Release Agent)

    for wc in range(len(machinesPerWC)):
        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        store = job_shop.storeWC[wc]

        env.process(
            cfp_wc(env, job_shop.machine_per_wc, store, job_shop, wc, normalization))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_per_wc[(ii, wc)]

            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
                                   machine, makespanWC, min_job, max_job, normalization, max_wip))
    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)  # Run the simulation until the end event gets triggered
    objective1 = np.nanmean(job_shop.tardiness[min_job:max_job]) + 0.01 * max(job_shop.tardiness[min_job:max_job])
    tard1 = job_shop.tardiness[min_job:max_job]

    stability = np.zeros(3)
    robustness = 1 - (np.abs(objective - objective1) / objective)
    for i in range(len(tard)):
        stability[0] += np.abs(tard1[i] - tard[i])
    for w in noOfWC:
        for m in range(machinesPerWC[w]):
            print(np.sum(job_shop.scheduleMachine[(m, w)], axis=0))

    print(job_shop.job_per_machine)
    # print(a)
    # print(tard)
    # print(tard1)
    return robustness, stability


if __name__ == '__main__':

    arrival_time = [1.5429, 1.5429, 1.4572, 1.4572, 1.3804, 1.3804]
    utilization = [85, 85, 90, 90, 95, 95]
    due_date_settings = [4, 6, 4, 6, 4, 6]

    min_jobs = [499, 499, 999, 999, 1499, 1499]
    max_jobs = [2499, 2499, 2999, 2999, 3499, 3499]
    wip_max = [150, 150, 200, 200, 300, 300]

    normaliziation = [[-75, 150, -8, 12, -75, 150], [-30, 150, -3, 12, -30, 150], [-200, 150, -15, 12, -200, 150],
                      [-75, 150, -8, 12, -75, 150], [-300, 150, -50, 12, -300, 150], [-150, 150, -15, 12, -150, 150]]

    final_obj = []
    final_std = []

    no_runs = 1
    # final_result = np.zeros((no_runs, len(skip_seq)))

    for j in range(1):
        str1 = "Runs/Final_Runs/Run-weights-" + str(utilization[j]) + "-" + str(due_date_settings[j]) + ".csv"
        df = pd.read_csv(str1, header=None)
        weights = df.values.tolist()
        obj = np.zeros(no_runs)
        jobshop_pool = Pool(processes=no_runs)
        seeds = range(no_runs)
        func1 = partial(do_simulation_with_weights, weights, arrival_time[j], due_date_settings[j],
                        normaliziation[j], min_jobs[j], max_jobs[j], wip_max[j], 0.05)
        makespan_per_seed = jobshop_pool.map(func1, seeds)
        # print(makespan_per_seed)
        for h in range(no_runs):
            print(makespan_per_seed)
            # final_result.append(makespan_per_seed[h])
        #     final_result[h][i] = makespan_per_seed[h][0]
        #     stability[]
        #
        # print(np.mean(final_result))

        # filename2 = 'Results/Tardiness_85_4.csv'
        # filename2 = "Results/Attributes_Final_" + str(utilization[j]) + "-" + str(due_date_settings[j]) + ".csv"
        # with open(filename2, 'w') as file2:
        #     writer = csv.writer(file2)
        #     writer.writerows(final_result)

        # arrival_time = [1.5429, 1.5429, 1.5429, 1.4572, 1.4572, 1.4572, 1.3804, 1.3804, 1.3804]
