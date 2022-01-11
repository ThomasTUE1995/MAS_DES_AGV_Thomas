"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import csv
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


def bid_winner(env, job, noOfMachines, currentWC, job_shop, last_job, makespan_currentWC, machine, store, bid_skip, normalization):
    # test_weights_new = job_shop.test_weights
    current_bid = [0] * noOfMachines
    current_job = [0] * noOfMachines
    best_bid = []
    best_job = []
    no_of_jobs = len(job)
    # total_rp = {j: (remain_processing_time(job[j])) for j in range(no_of_jobs)}
    # print(total_rp)
    total_rp = [0] * no_of_jobs
    for jj in range(no_of_jobs):
        total_rp[jj] = (remain_processing_time(job[jj]))

    for jj in range(noOfMachines):
        # expected_start = expected_start_time(jj, currentWC, machine)
        # start_time = max(env.now, makespan_currentWC[jj] + expected_start)
        queue_length = len(machine[(jj, currentWC - 1)].items)
        new_bid = [0] * no_of_jobs
        for ii, j in enumerate(job):
            attributes = bid_calculation(job_shop.test_weights, machine_number_WC[currentWC - 1][jj],
                                         j.processingTime[j.currentOperation - 1], j.currentOperation,
                                         total_rp[ii], j.dueDate[j.numberOfOperations],
                                         env.now,
                                         j.priority, queue_length, normalization)
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

    for ii, vv in enumerate(best_job):
        put_job_in_queue(currentWC, best_bid[ii], job[vv], job_shop, env, machine)

    for ii in reversed(best_job):
        yield store.get(lambda mm: mm == job[ii])


def bid_calculation(weights_new, machinenumber, processing_time,
                    current, total_rp, due_date, now, job_priority, queue_length,
                    normalization):
    attribute = [0] * noAttributes
    attribute[0] = processing_time / 8.75 * weights_new[machinenumber - 1][0]
    attribute[1] = (current - 1) / (5 - 1) * weights_new[machinenumber - 1][1]
    attribute[2] = (due_date - now - normalization[0]) / (normalization[1] - normalization[0]) * \
                   weights_new[machinenumber - 1][2]
    attribute[3] = total_rp / 21.25 * weights_new[machinenumber - 1][3]
    attribute[4] = (((due_date - now) / total_rp) - normalization[2]) / (normalization[3] - normalization[2]) * \
                   weights_new[machinenumber - 1][4]  # Critical Ratio
    attribute[5] = (job_priority - 1) / (10 - 1) * weights_new[machinenumber - 1][5]  # Job Weight
    attribute[6] = queue_length / 25 * weights_new[machinenumber - 1][6]  # Queue length
    attribute[7] = 0

    return sum(attribute)


def expected_start_time(jj, currentWC, machine):
    extra_start_time = 0
    for current_job in machine[jj].items:
        extra_start_time += (current_job.processingTime[current_job.currentOperation - 1] + mean_setup[currentWC - 1])

    return extra_start_time


def remain_processing_time(job):
    total_rp = 0
    # total_rp = sum((job.processingTime[ii]) for ii in range(job.currentOperation - 1, job.numberOfOperations))
    for ii in range(job.currentOperation - 1, job.numberOfOperations):
        total_rp += job.processingTime[ii]

    return total_rp


def next_workstation(job, job_shop, env, min_job, max_job, max_wip):
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        store = job_shop.storeWC[nextWC - 1]
        store.put(job)
    else:
        finish_time = env.now
        job_shop.tardiness[job.number] = max(job.priority * (finish_time - job.dueDate[job.numberOfOperations]), 0)
        job_shop.mean_WIP.append(job_shop.WIP)
        job_shop.WIP -= 1
        job_shop.makespan[job.number] = finish_time - job.dueDate[0]

        if job.number > max_job:
            if np.count_nonzero(job_shop.makespan[min_job:max_job]) == 2000:
                job_shop.finishtime = env.now
                job_shop.end_event.succeed()

        if (job_shop.WIP > max_wip) | (env.now > 7_000):
            print("Terminated")
            job_shop.end_event.succeed()
            job_shop.early_termination = 1


def normalize(value, max_value, min_value):
    return (value - min_value) / (max_value - min_value)


def expected_setup_time(new_job, job_shop, list_jobs):
    set_time = []

    for f in list_jobs:
        set_time.append(setupTime[f.type - 1][new_job.type - 1])

    return max(set_time)


def set_makespan(current_makespan, job, last_job, env, setup_time):
    # if last_job != 0:
    #     setup_time = setupTime[job.type - 1][int(last_job) - 1]
    # else:
    #     setup_time = 0
    add = current_makespan + job.processingTime[job.currentOperation - 1] + setup_time

    new = env.now + job.processingTime[job.currentOperation - 1] + setup_time

    return max(add, new)


def put_job_in_queue(currentWC, choice, job, job_shop, env, machines):
    machines[(choice, currentWC - 1)].put(job)
    if not job_shop.condition_flag[(choice, currentWC - 1)].triggered:
        job_shop.condition_flag[(choice, currentWC - 1)].succeed()


def choose_job_queue(weights_new_job, machinenumber, processing_time, due_date, env,
                     setup_time,
                     job_priority, seq_skip, normalization):
    attribute_job = [0] * noAttributesJob

    attribute_job[3] = 0
    attribute_job[2] = setup_time / 1.25 * weights_new_job[machinenumber - 1][noAttributes + 2]
    attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    attribute_job[0] = (due_date - processing_time - setup_time - env.now - normalization[4]) / (
            normalization[5] - normalization[4]) * \
                       weights_new_job[machinenumber - 1][noAttributes]

    return sum(attribute_job)


def machine_processing(job_shop, current_WC, machine_number, env, weights_new, relative_machine, last_job, machine,
                       makespan, seq_skip, utilization, normalization, min_job, max_job, max_wip):
    while True:
        relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
        if machine.items:
            setup_time = []
            priority_list = []
            if not last_job[relative_machine]:
                ind_processing_job = 0
                setup_time.append(0)
            else:
                for job in machine.items:
                    setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
                    # priority_list.append(job.dueDate[job.numberOfOperations])
                    job_queue_priority = choose_job_queue(weights_new, machine_number,
                                                          job.processingTime[job.currentOperation - 1],
                                                          job.dueDate[job.currentOperation], env, setuptime,
                                                          job.priority, seq_skip, normalization)
                    priority_list.append(job_queue_priority)
                    setup_time.append(setuptime)
                ind_processing_job = priority_list.index(max(priority_list))

            # ind_processing_job = 0
            next_job = machine.items[ind_processing_job]
            setuptime = setup_time[ind_processing_job]
            tip = next_job.processingTime[next_job.currentOperation - 1] + setuptime
            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, last_job[relative_machine],
                                                      env, setuptime)
            utilization = utilization + setuptime + next_job.processingTime[next_job.currentOperation - 1]
            last_job[relative_machine] = next_job.type
            machine.items.remove(next_job)
            yield env.timeout(tip)
            next_workstation(next_job, job_shop, env, min_job, max_job, max_wip)
        else:
            yield job_shop.condition_flag[(relative_machine, current_WC - 1)]
            job_shop.condition_flag[(relative_machine, current_WC - 1)] = simpy.Event(env)


def cfp_wc(env, last_job, machine, makespan, store, job_shop, currentWC, bid_skip, normalization):
    while True:
        if store.items:
            c = bid_winner(env, store.items, machinesPerWC[currentWC], currentWC + 1, job_shop, last_job, makespan,
                           machine, store, bid_skip, normalization)
            env.process(c)
        tib = 0.5
        yield env.timeout(tib)


def no_in_system(R):
    """Total number of jobs in the resource R"""
    return len(R.put_queue) + len(R.users)


def source(env, number1, interval, job_shop, due_date_setting):
    if not noJobCap:  # If there is a limit on the number of jobs
        for ii in range(number1):
            job = New_Job('job%02d' % ii, env, ii, due_date_setting)
            firstWC = operationOrder[job.type - 1][0]
            store = eval('job_shop.storeWC' + str(firstWC))
            store.put(job)
            tib = random.expovariate(1.0 / interval)
            yield env.timeout(tib)
    else:
        while True:  # Needed for infinite case as True refers to "until".
            ii = number1
            number1 += 1
            job = New_Job('job%02d' % ii, env, ii, due_date_setting)
            job_shop.tardiness.append(-1)
            job_shop.makespan.append(0)
            job_shop.WIP += 1
            firstWC = operationOrder[job.type - 1][0]
            store = job_shop.storeWC[firstWC - 1]
            store.put(job)
            tib = random.expovariate(1.0 / interval)
            yield env.timeout(tib)


class jobShop:
    def __init__(self, env, weights):
        self.machine_per_wc = {(ii, jj): Store(env) for jj in noOfWC for ii in range(machinesPerWC[jj])}
        self.storeWC = {ii: FilterStore(env) for ii in noOfWC}
        self.QueuesWC = {(ii, jj): [] for jj in noOfWC for ii in range(machinesPerWC[jj])}
        self.scheduleWC = {ii: [] for ii in noOfWC}
        self.makespanWC = {ii: np.zeros(machinesPerWC[ii]) for ii in noOfWC}
        self.last_job_WC = {ii: np.zeros(machinesPerWC[ii]) for ii in noOfWC}
        self.condition_flag = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(machinesPerWC[jj])}

        self.test_weights = weights
        self.makespan = []
        self.tardiness = []
        self.WIP = 0
        self.early_termination = 0
        self.utilization = {(ii, jj): 0 for jj in noOfWC for ii in range(machinesPerWC[jj])}

        self.bids = []
        self.mean_WIP = []


class New_Job:
    def __init__(self, name, env, number1, dueDateTightness):
        jobType = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)
        jobWeight = random.choices([1, 3, 10], weights=[0.5, 0.3, 0.2], k=1)
        self.type = jobType[0]
        self.priority = jobWeight[0]
        self.number = number1
        self.name = name
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


def do_simulation_with_weights(mean_weight_new, arrivalMean, due_date_tightness, bid_skip, seq_skip, normalization, min_job, max_job, max_wip, iter):
    random.seed(iter)
    for mm in range(sum(machinesPerWC)):
        for jj in range(totalAttributes):
            if (jj == noAttributes - 1) | (jj == noAttributesJob + noAttributes - 1) | (jj in bid_skip) | (
                    jj in [x + noAttributes for x in seq_skip]):
                mean_weight_new[mm][jj] = 0

    # print(mean_weight_new)

    env = Environment()
    job_shop = jobShop(env, mean_weight_new)
    env.process(source(env, number, arrivalMean, job_shop, due_date_tightness))

    for wc in range(len(machinesPerWC)):
        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        store = job_shop.storeWC[wc]

        env.process(cfp_wc(env, last_job, job_shop.machine_per_wc, makespanWC, store, job_shop, wc, bid_skip, normalization))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_per_wc[(ii, wc)]
            utilization = job_shop.utilization[(ii, wc)]

            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, ii, last_job,
                                   machine, makespanWC, seq_skip, utilization, normalization, min_job, max_job, max_wip))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)
    objective = np.nanmean(job_shop.tardiness[min_job:max_job]) + 0.01 * max(job_shop.tardiness[min_job:max_job])
    return objective


if __name__ == '__main__':
    # df = pd.read_csv('Runs/Attribute_Runs/Run-Weights-NoDD-85-4-5000.csv', header=None)
    # weights = df.values.tolist()
    #

    arrival_time = [1.5429, 1.5429, 1.4572, 1.4572, 1.3804, 1.3804]
    utilization = [85, 85, 90, 90, 95, 95]
    due_date_settings = [4, 6, 4, 6, 4, 6]

    min_jobs = [499, 499, 999, 999, 1499, 1499]
    max_jobs = [2499, 2499, 2999, 2999, 3499, 3499]
    wip_max = [150, 150, 200, 200, 300, 300]

    # skip_bid = [[7, 7], [0, 7], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [7, 7], [7, 7], [7, 7]]
    # skip_seq = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [0, 3], [1, 3], [2, 3]]

    skip_bid = [[7, 7]]
    skip_seq = [[3, 3]]

    # skip_bid = [[7, 7], [2, 7], [4, 7]]
    # skip_seq = [[3, 3], [3, 3], [3, 3]]

    # normaliziation = [[-250, 150, -25, 12, -250, 150], [-275, 150, -28, 12, -275, 150], [-325, 150, -32, 12, -325, 150], [-350, 150, -35, 12, -350, 150]]

    normaliziation = [[-75, 150, -8, 12, -75, 150], [-30, 150, -3, 12, -30, 150], [-200, 150, -15, 12, -200, 150],
                      [-75, 150, -8, 12, -75, 150], [-300, 150, -50, 12, -300, 150], [-150, 150, -15, 12, -150, 150]]

    final_obj = []
    final_std = []

    no_runs = 100
    # final_result = np.zeros((no_runs, len(skip_seq)))

    for j in range(4, 5):
        final_result = np.zeros((no_runs, len(skip_seq)))
        for i, (skip_b, skip_s) in enumerate(zip(skip_bid, skip_seq)):
            str1 = "Runs/Attribute_Runs/" + str(utilization[j]) + "-" + str(due_date_settings[j]) + "/Run-weights-" + str(utilization[j]) + "-" + str(due_date_settings[j]) + "-1000" + "-" + str(skip_b) + "-" + str(
                skip_s) + ".csv"
            df = pd.read_csv(str1, header=None)
            weights = df.values.tolist()
            # print(weights)
            # print(weights)
            obj = np.zeros(no_runs)
            jobshop_pool = Pool(processes=no_runs)
            seeds = range(no_runs)
            func1 = partial(do_simulation_with_weights, weights, arrival_time[j], due_date_settings[j], skip_b, skip_s, normaliziation[j], min_jobs[j], max_jobs[j], wip_max[j])
            makespan_per_seed = jobshop_pool.map(func1, seeds)
            # print(makespan_per_seed)
            for h in range(no_runs):
                # final_result.append(makespan_per_seed[h])
                final_result[h][i] = makespan_per_seed[h]

            print(np.mean(final_result))

        # filename2 = 'Results/Tardiness_85_4.csv'
        # filename2 = "Results/Attributes_Final_" + str(utilization[j]) + "-" + str(due_date_settings[j]) + ".csv"
        # with open(filename2, 'w') as file2:
        #     writer = csv.writer(file2)
        #     writer.writerows(final_result)

        # arrival_time = [1.5429, 1.5429, 1.5429, 1.4572, 1.4572, 1.4572, 1.3804, 1.3804, 1.3804]
