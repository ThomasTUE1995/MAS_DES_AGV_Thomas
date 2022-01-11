"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import csv
import gc
import random
from collections import defaultdict
from functools import partial

import pandas
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool

# import matplotlib.cbook
import simpy
import numpy as np
from matplotlib import pyplot as plt
from simpy import *

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]
numberOfOperations = [4, 3, 5]
machinesPerWC = [4, 2, 5, 3, 2]
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]
setupTime = [[0.01, 0.125, 0.25], [0.125, 0.01, 0.16], [0.25, 0.16, 0.01]]
arrivalMean = 1.53  # Mean of arrival process

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 6
noAttributesJob = 3
totalAttributes = noAttributes

no_generation = 500


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


def bid_winner(env, job, noOfMachines, currentWC, job_shop):
    test_weights_new = job_shop.test_weights
    list_jobs = job
    no_jobs = len(list_jobs)
    current_bid = np.zeros(noOfMachines)
    current_job = np.zeros(noOfMachines)
    current_time = np.zeros(noOfMachines)
    best_bid = []
    best_job = []
    best_time = []
    removed_job = []
    queue = eval('job_shop.machinesWC' + str(currentWC))
    last_job = eval('job_shop.last_job_WC' + str(currentWC))
    makespan_currentWC = eval('job_shop.makespanWC' + str(currentWC))

    meanProcessingTime = 0

    dueDateVector = np.zeros(no_jobs)
    setup_time_mean = np.zeros(noOfMachines)
    for ii in range(no_jobs):
        dueDateVector[ii] = job[ii].dueDate[job[ii].currentOperation]
        meanProcessingTime += job[ii].processingTime[job[ii].currentOperation - 1]
        for jj in range(noOfMachines):
            if last_job[jj] != 0:
                setup_time_mean[jj] += setupTime[job[ii].type - 1][int(last_job[jj]) - 1]
            else:
                setup_time_mean[jj] += 0

    meanDueDate = np.mean(dueDateVector)
    maxDueDate = max(dueDateVector)
    minDueDate = min(dueDateVector)
    setup_time_mean = setup_time_mean / no_jobs
    meanProcessingTime = meanProcessingTime / no_jobs

    for ii in range(len(job)):
        new_job = job[ii]
        due_date_operation = new_job.dueDate[new_job.currentOperation]
        due_date = new_job.dueDate[new_job.numberOfOperations]
        currentOperation = new_job.currentOperation
        totalOperations = new_job.numberOfOperations

        proc_time = new_job.processingTime[currentOperation - 1]
        for jj in range(noOfMachines):
            if last_job[jj] != 0:
                setup_time = setupTime[new_job.type - 1][int(last_job[jj]) - 1]
            else:
                setup_time = 0
            #
            # new_bid = (-no_in_system(queue[jj]))
            # queue_length = no_in_system(queue[jj])
            machine = eval('job_shop.machinesWC' + str(currentWC))
            queue_length = len(machine[jj].items)
            # print(queue_length)
            total_rp = remain_processing_time(new_job)
            expected_start = expected_start_time(jj, currentWC, job_shop, last_job)
            # expected_start = 0
            start_time = max(env.now, makespan_currentWC[jj] + expected_start)
            # print(start_time)

            #
            new_bid = bid_calculation(test_weights_new, machine_number_WC[currentWC - 1][jj], no_jobs,
                                      proc_time, setup_time,
                                      queue_length, start_time, currentOperation, totalOperations,
                                      due_date_operation, total_rp, expected_start, due_date, env.now)

            # new_bid = bid_calulculation_other(no_jobs, noOfMachines, setup_time_mean[jj], meanProcessingTime,
            #                                   meanDueDate,
            #                                   maxDueDate, minDueDate, due_date, proc_time, start_time, setup_time,
            #                                   list_jobs)

            if ii == 0:
                current_bid[jj] = new_bid
                current_job[jj] = ii
                current_time[jj] = new_job.processingTime[new_job.currentOperation - 1] + setup_time
            elif new_bid >= current_bid[jj]:
                current_bid[jj] = new_bid
                current_job[jj] = ii
                current_time[jj] = new_job.processingTime[new_job.currentOperation - 1] + setup_time

    # Determine the winning bids
    sorted_list = sorted(list_duplicates(current_job))
    for dup in sorted_list:
        bestmachine = dup[1][0]
        bestbid = current_bid[bestmachine]
        bestime = current_time[bestmachine]
        for ii in dup[1]:
            if bestbid < current_bid[ii]:
                bestbid = current_bid[ii]
                bestmachine = ii
                bestime = current_time[ii]

        best_bid.append(bestmachine)  # Machine winner
        best_time.append(bestime)
        # print(bestbid, bestmachine)
        best_job.append(int(dup[0]))  # Job to be processed

    for ii in range(len(best_job)):
        put_job_in_queue(currentWC, best_bid[ii], job[best_job[ii]], job_shop)
        # c = globals()['WC' + str(currentWC)](env, job[best_job[ii]], job_shop, best_bid[ii], best_time[ii])
        # env.process(c)
        removed_job.append(best_job[ii])

    for ii in reversed(removed_job):
        func = eval('job_shop.storeWC' + str(currentWC))
        yield func.get(lambda mm: mm == list_jobs[ii])


def expected_start_time(jj, currentWC, job_shop, last_job):
    machine = eval('job_shop.machinesWC' + str(currentWC))
    extra_start_time = 0
    for kk in range(len(machine[jj].items)):
        current_job = machine[jj].items[kk]
        extra_start_time += current_job.processingTime[current_job.currentOperation - 1]

    return extra_start_time
    #
    # total_setup = 0
    #
    # if (last_job[jj] != 0) & (len(machine[jj].items) > 0):
    #     first_job = machine[jj].items[0]
    #     total_setup += setupTime[int(last_job[jj]) - 1][first_job.type - 1]
    #
    # for f in range(len(machine[jj].items) - 1):
    #     for ll in range(len(machine[jj].items)):
    #         first_job = machine[jj].items[f]
    #         second_job = machine[jj].items[ll]
    #         total_setup += setupTime[first_job.type - 1][second_job.type - 1]

    # return extra_start_time + total_setup


def bid_calulculation_other(pool, noMachines, meanSetup, meanProcessingTime, meanDueDate, maxDueDate, minDueDate,
                            due_date, processingTime, current_time, setup_time, job):
    if meanSetup == 0:
        meanSetup = 0.01

    mu = pool / noMachines
    eta = meanSetup / meanProcessingTime
    beta = 0.4 - 10 / (mu ** 2) - eta / 7
    tau = 1 - meanDueDate / (beta * meanSetup + meanProcessingTime) * mu
    R = (maxDueDate - minDueDate) / (beta * meanSetup + meanProcessingTime) * mu

    if (tau < 0.5) | ((eta < 0.5) & (mu > 5)):
        k_1 = 1.2 * np.log(mu) - R - 0.5
    else:
        k_1 = 1.2 * np.log(mu) - R

    if tau < 0.8:
        k_2 = tau / (1.8 * np.sqrt(eta))
    else:
        k_2 = tau / (2 * np.sqrt(eta))

    bid = 1 / processingTime * np.exp(
        -max(0, (due_date - processingTime - current_time)) / (k_1 * meanProcessingTime)) * np.exp(
        -setup_time / (k_2 * meanSetup))

    # if bid > 10 ** 9:
    #     print(eta, mu, tau)

    return bid


def remain_processing_time(job):
    total_rp = 0
    for ii in range(job.currentOperation, job.numberOfOperations):
        total_rp += job.processingTime[ii]

    return total_rp


def next_workstation(job, job_shop, env):
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        c = job_pool_agent(job, nextWC, job_shop)
        env.process(c)
    else:
        finish_time = env.now
        job_shop.tardiness[job.number] = max((finish_time - job.dueDate[job.numberOfOperations]), 0)
        # print(job.number)
        # print(np.count_nonzero(job_shop.tardiness[499:2500]))
        # job_shop.tardiness.append(max((finish_time - job.dueDate[job.numberOfOperations]), 0))
        job_shop.makespan[job.number] = finish_time - job.dueDate[0]

        if (job.number > 2499) & (np.count_nonzero(job_shop.makespan[499:2499]) == 2000):
            job_shop.finishtime = env.now
            job_shop.end_event.succeed()
        #
        #
        if job.number > 2499:
            print(np.count_nonzero(job_shop.makespan[499:2500]) == 2000)


def bid_calculation(weights_new, machinenumber, pool, processing_time, setup_time, queue, proc_time,
                    current, total, due_date_operation, total_rp, expected_start, due_date, now):
    attribute = [0] * noAttributes
    attribute[0] = normalize(processing_time, 8.75, 0) * weights_new[machinenumber - 1][0]
    attribute[1] = normalize(queue, 25, 0) * weights_new[machinenumber - 1][1]
    attribute[2] = normalize((total - current), total - 1, 1) * weights_new[machinenumber - 1][2]
    attribute[3] = normalize(setup_time, 0.25, 0) * weights_new[machinenumber - 1][3]
    attribute[4] = normalize((due_date_operation - (processing_time + setup_time) - proc_time), 60, -100) * \
                   weights_new[machinenumber - 1][4]
    print(normalize((due_date_operation - (processing_time + setup_time) - proc_time), 60, -200))
    attribute[5] = normalize(total_rp, 21.25, 0) * weights_new[machinenumber - 1][5]
    # attribute[6] = normalize(expected_start, 100, 0) * weights_new[machinenumber - 1][6]  # Current Workload
    # attribute[7] = normalize((due_date - now), 100, -100) * \
    #                weights_new[machinenumber - 1][7]  # Slack

    total_bid = sum(attribute)
    return total_bid


def normalize(value, max_value, min_value):
    return (value - min_value) / (max_value - min_value)


def set_makespan(current_makespan, job, last_job, env):
    if last_job != 0:
        setup_time = setupTime[job.type - 1][int(last_job) - 1]
    else:
        setup_time = 0
    add = current_makespan + job.processingTime[job.currentOperation - 1] + setup_time

    new = env.now + job.processingTime[job.currentOperation - 1] + setup_time

    return max(add, new)


def put_job_in_queue(currentWC, choice, job, job_shop):
    machines = eval('job_shop.machinesWC' + str(currentWC))
    machines[choice].put(job)
    if not job_shop.condition_flag[currentWC - 1][choice].triggered:
        job_shop.condition_flag[currentWC - 1][choice].succeed()
    # print(machines[choice].items, choice, currentWC)


def choose_job_queue(weights_new, machinenumber, processing_time, current, total, due_date, total_rp):
    attribute = [0] * noAttributesJob
    attribute[0] = normalize(processing_time, 15, 0) * weights_new[machinenumber - 1][noAttributes]
    # attribute[1] = normalize((total - current), total - 1, 1) * weights_new[machinenumber - 1][7]
    attribute[1] = normalize(due_date, 100, -100) * weights_new[machinenumber - 1][noAttributes + 1]
    attribute[2] = normalize(total_rp, 45, 0) * weights_new[machinenumber - 1][noAttributes + 2]
    # attribute[10] = normalize(setup_time, 0.25, 0) * weights_new[machinenumber - 1][10]

    total_bid = sum(attribute)
    return total_bid


def machine_processing(job_shop, current_WC, machine_number, env, weights_new):
    while True:
        WC = eval('job_shop.QueuesWC' + str(current_WC))
        last_job = eval('job_shop.last_job_WC' + str(current_WC))
        machine = eval('job_shop.machinesWC' + str(current_WC))
        makespan = eval('job_shop.makespanWC' + str(current_WC))
        schedule = eval('job_shop.scheduleWC' + str(current_WC))
        timeWC = eval('job_shop.timeWC' + str(current_WC))
        current_items = machine[machine_number].items
        if current_items:
            WC.append(len(current_items))
            priority_list = []
            for job in current_items:
                priority_list.append(job.dueDate[job.numberOfOperations])
                # priority = choose_job_queue(weights_new, machine_number, job.processingTime[job.currentOperation - 1],
                #                             job.currentOperation, job.numberOfOperations,
                #                             job.dueDate[job.numberOfOperations],
                #                             remain_processing_time(job))
                # priority_list.append(priority)
            ind_processing_job = priority_list.index(min(priority_list))
            # ind_processing_job = 0
            next_job = current_items[ind_processing_job]
            if last_job[machine_number] != 0:
                setup_time = setupTime[next_job.type - 1][int(last_job[machine_number]) - 1]
            else:
                setup_time = 0
            tip = next_job.processingTime[next_job.currentOperation - 1] + setup_time
            timeWC[machine_number].append(tip)
            makespan[machine_number] = set_makespan(makespan[machine_number], next_job, last_job[machine_number],
                                                    env)
            last_job[machine_number] = next_job.type
            schedule.append(
                [machine_number, next_job.name, next_job.type, makespan[machine_number], tip, setup_time, env.now])
            current_items.remove(next_job)
            yield env.timeout(tip)
            next_workstation(next_job, job_shop, env)
        else:
            yield job_shop.condition_flag[current_WC - 1][machine_number]
            job_shop.condition_flag[current_WC - 1][machine_number] = simpy.Event(env)


def job_pool_agent(job, currentWC, job_shop):
    func = eval('job_shop.storeWC' + str(currentWC))
    yield func.put(job)


def Enquiry(lis1):
    if not lis1:
        return 1
    else:
        return 0


def cfp_wc1(env, job_shop):
    while True:
        if job_shop.storeWC1.items:
            c = bid_winner(env, job_shop.storeWC1.items, machinesPerWC[0], 1, job_shop)
            env.process(c)
        job_shop.poolLengthWC1.append(len(job_shop.storeWC1.items))
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc2(env, job_shop):
    while True:
        if not Enquiry(job_shop.storeWC2.items):
            c = bid_winner(env, job_shop.storeWC2.items, machinesPerWC[1], 2, job_shop)
            env.process(c)
        job_shop.poolLengthWC2.append(len(job_shop.storeWC2.items))
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc3(env, job_shop):
    while True:
        if not Enquiry(job_shop.storeWC3.items):
            c = bid_winner(env, job_shop.storeWC3.items, machinesPerWC[2], 3, job_shop)
            env.process(c)
        job_shop.poolLengthWC3.append(len(job_shop.storeWC3.items))
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc4(env, job_shop):
    while True:
        if not Enquiry(job_shop.storeWC4.items):
            c = bid_winner(env, job_shop.storeWC4.items, machinesPerWC[3], 4, job_shop)
            env.process(c)
        job_shop.poolLengthWC4.append(len(job_shop.storeWC4.items))
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc5(env, job_shop):
    while True:
        if not Enquiry(job_shop.storeWC5.items):
            c = bid_winner(env, job_shop.storeWC5.items, machinesPerWC[4], 5, job_shop)
            env.process(c)
        job_shop.poolLengthWC5.append(len(job_shop.storeWC5.items))
        tib = 1.0
        yield env.timeout(tib)


def no_in_system(R):
    """Total number of jobs in the resource R"""
    return len(R.put_queue) + len(R.users)


def source(env, number1, interval, job_shop):
    if not noJobCap:  # If there is a limit on the number of jobs
        for ii in range(number1):
            job = New_Job('job%02d' % ii, env, ii)
            job_shop.tardiness.append(0)
            firstWC = operationOrder[job.type - 1][0]
            d = job_pool_agent(job, firstWC, job_shop)
            env.process(d)
            tib = random.expovariate(1.0 / interval)
            yield env.timeout(tib)
    else:
        while True:  # Needed for infinite case as True refers to "until".
            ii = number1
            number1 += 1
            job = New_Job('job%02d' % ii, env, ii)
            job_shop.tardiness.append(-1)
            job_shop.makespan.append(0)
            firstWC = operationOrder[job.type - 1][0]
            d = job_pool_agent(job, firstWC, job_shop)
            env.process(d)
            tib = random.expovariate(1.0 / interval)
            yield env.timeout(tib)


class jobShop:
    def __init__(self, env, weights):
        machine_wc1 = {ii: Store(env) for ii in range(machinesPerWC[0])}
        machine_wc2 = {ii: Store(env) for ii in range(machinesPerWC[1])}
        machine_wc3 = {ii: Store(env) for ii in range(machinesPerWC[2])}
        machine_wc4 = {ii: Store(env) for ii in range(machinesPerWC[3])}
        machine_wc5 = {ii: Store(env) for ii in range(machinesPerWC[4])}

        job_poolwc1 = simpy.FilterStore(env)
        job_poolwc2 = simpy.FilterStore(env)
        job_poolwc3 = simpy.FilterStore(env)
        job_poolwc4 = simpy.FilterStore(env)
        job_poolwc5 = simpy.FilterStore(env)

        self.machinesWC1 = machine_wc1
        self.machinesWC2 = machine_wc2
        self.machinesWC3 = machine_wc3
        self.machinesWC4 = machine_wc4
        self.machinesWC5 = machine_wc5

        self.QueuesWC1 = []
        self.QueuesWC2 = []
        self.QueuesWC3 = []
        self.QueuesWC4 = []
        self.QueuesWC5 = []

        self.timeWC1 = []
        self.timeWC2 = []
        self.timeWC3 = []
        self.timeWC4 = []
        self.timeWC5 = []

        self.scheduleWC1 = []
        self.scheduleWC2 = []
        self.scheduleWC3 = []
        self.scheduleWC4 = []
        self.scheduleWC5 = []

        self.poolLengthWC1 = []
        self.poolLengthWC2 = []
        self.poolLengthWC3 = []
        self.poolLengthWC4 = []
        self.poolLengthWC5 = []

        self.condition_flag = []
        for wc in range(len(machinesPerWC)):
            Q = eval('self.QueuesWC' + str(wc + 1))
            P = eval('self.timeWC' + str(wc + 1))
            self.condition_flag.append([])
            for ii in range(machinesPerWC[wc]):
                Q.append([])
                P.append([])
                self.condition_flag[wc].append(simpy.Event(env))

        self.makespanWC1 = np.zeros(machinesPerWC[0])
        self.makespanWC2 = np.zeros(machinesPerWC[1])
        self.makespanWC3 = np.zeros(machinesPerWC[2])
        self.makespanWC4 = np.zeros(machinesPerWC[3])
        self.makespanWC5 = np.zeros(machinesPerWC[4])

        self.last_job_WC1 = np.zeros(machinesPerWC[0])
        self.last_job_WC2 = np.zeros(machinesPerWC[1])
        self.last_job_WC3 = np.zeros(machinesPerWC[2])
        self.last_job_WC4 = np.zeros(machinesPerWC[3])
        self.last_job_WC5 = np.zeros(machinesPerWC[4])

        self.storeWC1 = job_poolwc1
        self.storeWC2 = job_poolwc2
        self.storeWC3 = job_poolwc3
        self.storeWC4 = job_poolwc4
        self.storeWC5 = job_poolwc5

        self.test_weights = weights
        self.makespan = []
        self.tardiness = []
        self.finishtime = 0


class New_Job:
    def __init__(self, name, env, number1):
        jobType = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)
        self.type = jobType[0]
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
            # self.processingTime[ii] = random.gammavariate(5, meanPT / 5)
            dueDateTightness = 4
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * dueDateTightness


def do_simulation_with_weights(mean_weight_new, iter):
    env = Environment()
    job_shop = jobShop(env, mean_weight_new)
    env.process(source(env, number, arrivalMean, job_shop))

    env.process(cfp_wc1(env, job_shop))
    env.process(cfp_wc2(env, job_shop))
    env.process(cfp_wc3(env, job_shop))
    env.process(cfp_wc4(env, job_shop))
    env.process(cfp_wc5(env, job_shop))
    for wc in range(len(machinesPerWC)):
        for ii in range(machinesPerWC[wc]):
            env.process(machine_processing(job_shop, wc + 1, ii, env, mean_weight_new))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)
    objective = job_shop.tardiness[499:2499]
    # print(job_shop.tardiness)
    print(job_shop.tardiness[499:2499], len(job_shop.tardiness))
    #
    # if i == 1:
    #     fig, axes = plt.subplots(1, 1)
    #     # plt.title("Individual Queue Loads WC1")
    #
    #     axes = df1.plot(ax=axes)
    #     axes.set_ylabel('Queue length', fontsize=20)
    #     axes.set_xlabel('No. of jobs', fontsize=20)
    #     plt.show()

    # print(job_shop.tardiness)
    # print(job_shop.scheduleWC1)
    utilization = np.zeros((5, 5))
    for wc in range(5):
        Q = eval('job_shop.timeWC' + str(wc + 1))
        for ma in range(machinesPerWC[wc]):
            utilization[wc][ma] = sum(Q[ma]) / job_shop.finishtime

    print(job_shop.QueuesWC1)

    # print(utilization)

    return np.mean(objective)


if __name__ == '__main__':
    df = pandas.read_csv('Runs/Run_weights_EDD_85_4.csv', header=None)
    weights = df.values.tolist()
    # weights = np.zeros((sum(machinesPerWC), 6))

    # for i in range(sum(machinesPerWC)):
    #     for j in range(6):
    #         weights[i][j] = random.uniform(-1, 1)

    # random.seed(1)
    no_runs = 1
    obj = np.zeros(no_runs)
    jobshop_pool = Pool(processes=no_runs, maxtasksperchild=10)
    seeds = range(no_runs)
    func1 = partial(do_simulation_with_weights, weights)
    makespan_per_seed = jobshop_pool.map(func1, seeds)
    print(makespan_per_seed)
    for h in range(no_runs):
        obj[h] = makespan_per_seed[h]
        print(obj[h])

    filename2 = "Results/Custom_85_1.txt"
    file2 = open(filename2, 'w')
    for i in range(len(makespan_per_seed)):
        file2.writelines(str(makespan_per_seed[i]) + "\n")
    # writer = csv.writer(file2)
    # writer.writerow(makespan_per_seed)
    # for ii in range(sum(machinesPerWC)):
    #     writer.writerow(mean_weight[ii])
    file2.close()

    print(np.mean(obj), np.std(obj))

    # # to run GUI event loop
    # dat = [0, 1]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # Ln, = ax.plot(dat)
    # ax.set_xlim([0, 1000])
    # ax.set_ylim([0, 150])
    # plt.ion()
    # plt.show()
    #
    # # setting title
    # plt.title("Mean objective function", fontsize=20)
    #
    # # setting x-axis label and y-axis label
    # plt.xlabel("No. of iterations")
    # plt.ylabel("Obejctive Function")
