"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import random
import sys
import time
from collections import defaultdict
import simpy
from numba import jit
from simpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

number = 10000  # Max number of jobs if infinite is false
noJobCap = False  # For infinite
maxTime = 1000.0  # Runtime limit
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]
numberOfOperations = [4, 3, 5]
machinesPerWC = [4, 2, 5, 3, 2]
setupTime = [[0, 0.125, 0.25], [0.125, 0, 0.16], [0.25, 0.16, 0]]
arrivalMean = 5 / 3  # Mean of arrival process
seed = 204209  # Seed for RNG
dueDateTightness = 2
parallelism = 5  # Number of queues
doPrint = False  # True => print every arrival and wait time

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 9
# weights = np.zeros((16, noAttributes))
mean_weight = np.zeros((sum(machinesPerWC), noAttributes))
std_weight = np.zeros((sum(machinesPerWC), noAttributes))
std_weight = std_weight + np.log(0.3)
alpha_mean = 0.1
alpha_std = 0.025
beta_1 = 0.9
beta_2 = 0.999
# eta = 10 ** -8
no_generation = 500
population_size = 24
for i in range(sum(machinesPerWC)):
    mean_weight[i][0] = -0.5

QueuesWC1 = []
QueuesWC2 = []
QueuesWC3 = []
QueuesWC4 = []
QueuesWC5 = []

currentTardiness = []

last_job_WC1 = np.zeros(machinesPerWC[0])
last_job_WC2 = np.zeros(machinesPerWC[1])
last_job_WC3 = np.zeros(machinesPerWC[2])
last_job_WC4 = np.zeros(machinesPerWC[3])
last_job_WC5 = np.zeros(machinesPerWC[4])

makespan = []
currentCycleTime = []


def bid_winner(env, job, noOfMachines, currentWC, job_shop):
    test_weights_new = job_shop.test_weights
    # print(test_weights_new)
    list_jobs = job
    no_jobs = len(list_jobs)
    # bids = np.zeros((len(job), noOfMachines))
    current_bid = np.zeros(noOfMachines)
    current_job = np.zeros(noOfMachines)
    current_time = np.zeros(noOfMachines)
    best_bid = []
    best_job = []
    best_time = []
    removed_job = []
    queue = eval('job_shop.machinesWC' + str(currentWC))
    max_due_date = [job[n].dueDate[job[n].currentOperation] for n in range(len(job))]
    total_machines = machinesPerWC[currentWC - 1]
    last_job = eval('last_job_WC' + str(currentWC))

    maxduedate = min(max_due_date)
    for ii in range(len(job)):
        new_job = job[ii]
        due_date = new_job.dueDate[new_job.currentOperation - 1]
        currentOperation = new_job.currentOperation
        totalOperations = new_job.numberOfOperations
        job_weight = 1
        proc_time = new_job.processingTime[currentOperation - 1]
        for jj in range(noOfMachines):
            if last_job[jj] != 0:
                setup_time = setupTime[new_job.type - 1][int(last_job[jj]) - 1]
            else:
                setup_time = 0

            # new_bid = 1 / (no_in_system(queue[jj]) + new_job.processingTime[new_job.currentOperation - 1])
            queue_length = no_in_system(queue[jj])
            total_rp = remain_processing_time(new_job)

            new_bid = bid_calculation(test_weights_new, jj + 1, no_jobs,
                                      proc_time, setup_time,
                                      queue_length, currentWC, env.now, maxduedate, currentOperation, totalOperations,
                                      due_date, job_weight, total_machines, total_rp)

            if ii == 0:
                current_bid[jj] = new_bid
                current_job[jj] = ii
                current_time[jj] = new_job.processingTime[new_job.currentOperation - 1] + setup_time
            elif new_bid >= current_bid[jj]:
                current_bid[jj] = new_bid
                current_job[jj] = ii
                current_time[jj] = new_job.processingTime[new_job.currentOperation - 1] + setup_time

    # Determine the winning bids
    new_list = sorted(list_duplicates(current_job))
    for dup in new_list:
        bestbid = -1
        bestmachine = 0
        bestime = 0
        for ii in dup[1]:
            if ii == 0:
                bestbid = current_bid[ii]
                bestmachine = ii
                bestime = current_time[ii]
            if bestbid < current_bid[ii]:
                bestbid = current_bid[ii]
                bestmachine = ii
                bestime = current_time[ii]
        best_bid.append(bestmachine)  # Machine winner
        best_time.append(bestime)
        best_job.append(int(dup[0]))  # Job to be processed

    for ii in range(len(best_job)):
        c = globals()['WC' + str(currentWC)](env, job[best_job[ii]], job_shop, best_bid[ii], best_time[ii])
        env.process(c)
        removed_job.append(best_job[ii])

    for ii in reversed(removed_job):
        func = eval('job_shop.storeWC' + str(currentWC))
        yield func.get(lambda mm: mm == list_jobs[ii])


def remain_processing_time(job):
    total_rp = 0
    for ii in range(job.currentOperation, job.numberOfOperations):
        total_rp += job.processingTime[ii]

    return total_rp


def next_workstation(job, job_shop, env):
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        c = job_pool_agent(job, job_shop, nextWC)
        env.process(c)
    else:
        finish_time = env.now
        job_shop.tardiness.append(finish_time - job.dueDate[job.numberOfOperations])
        # print(finish_time - job.dueDate[0], job.currentOperation)
        # currentCycleTime.append(finish_time - job.dueDate[0])
        job_shop.makespan.append(finish_time - job.dueDate[0])


# @jit(nopython=True)
def bid_calculation(weights_new, machinenumber, pool, processing_time, setup_time, queue, wc, proc_time, max_due_date,
                    current, total, due_date, job_weight, machines, total_rp):
    attribute = [0] * noAttributes
    attribute[0] = normalize(processing_time, 8.75, 0) * weights_new[machinenumber - 1][0]
    attribute[1] = normalize(queue, 100, 0) * weights_new[machinenumber - 1][1]
    attribute[2] = normalize((total - current), total, 1) * weights_new[machinenumber - 1][2]
    attribute[3] = normalize(setup_time, 0.25, -1) * weights_new[machinenumber - 1][3]
    attribute[4] = normalize((due_date - (processing_time + setup_time) - proc_time), 100, -100) * \
                   weights_new[machinenumber - 1][4]
    attribute[5] = normalize(pool, 100, 0) * weights_new[machinenumber - 1][5]
    attribute[6] = normalize(machines, 5, 1) * weights_new[machinenumber - 1][6]
    attribute[7] = normalize(max_due_date, 100, 0) * weights_new[machinenumber - 1][7]
    attribute[8] = normalize(total_rp, 40, 0) * weights_new[machinenumber - 1][8]

    total_bid = sum(attribute)

    # total_bid = 0
    # for jj in range(noAttributes):
    #     total_bid += attribute[jj]

    # print(attribute[4])
    return total_bid


def normalize(value, max_value, min_value):
    return (value - min_value) / (max_value - min_value)


def WC1(env, job, job_shop, choice, tib):
    QueuesWC1.append({ii: len(job_shop.machinesWC1[ii].put_queue) for ii in range(len(job_shop.machinesWC1))})
    last_job_WC1[choice] = job.type
    with job_shop.machinesWC1[choice].request() as req:
        # Wait in queue
        yield req
        yield env.timeout(tib)
        QueuesWC1.append({ii: len(job_shop.machinesWC1[ii].put_queue) for ii in range(len(job_shop.machinesWC1))})
        next_workstation(job, job_shop, env)


def WC2(env, job, job_shop, choice, tib):
    QueuesWC2.append({ii: len(job_shop.machinesWC2[ii].put_queue) for ii in range(len(job_shop.machinesWC2))})
    last_job_WC2[choice] = job.type
    with job_shop.machinesWC2[choice].request() as req2:
        # Wait in queue
        yield req2
        yield env.timeout(tib)
        QueuesWC2.append({ii: len(job_shop.machinesWC2[ii].put_queue) for ii in range(len(job_shop.machinesWC2))})
        next_workstation(job, job_shop, env)


def WC3(env, job, job_shop, choice, tib):
    QueuesWC3.append({ii: len(job_shop.machinesWC3[ii].put_queue) for ii in range(len(job_shop.machinesWC3))})
    last_job_WC3[choice] = job.type
    with job_shop.machinesWC3[choice].request() as req2:
        # Wait in queue
        yield req2
        yield env.timeout(tib)
        QueuesWC3.append({ii: len(job_shop.machinesWC3[ii].put_queue) for ii in range(len(job_shop.machinesWC3))})
        next_workstation(job, job_shop, env)


def WC4(env, job, job_shop, choice, tib):
    QueuesWC4.append({ii: len(job_shop.machinesWC4[ii].put_queue) for ii in range(len(job_shop.machinesWC4))})
    last_job_WC4[choice] = job.type
    with job_shop.machinesWC4[choice].request() as req2:
        # Wait in queue
        yield req2
        yield env.timeout(tib)
        QueuesWC4.append({ii: len(job_shop.machinesWC4[ii].put_queue) for ii in range(len(job_shop.machinesWC4))})
        next_workstation(job, job_shop, env)


def WC5(env, job, job_shop, choice, tib):
    QueuesWC5.append({ii: len(job_shop.machinesWC5[ii].put_queue) for ii in range(len(job_shop.machinesWC5))})
    last_job_WC5[choice] = job.type
    with job_shop.machinesWC5[choice].request() as req2:
        # Wait in queue
        yield req2
        yield env.timeout(tib)
        QueuesWC5.append({ii: len(job_shop.machinesWC5[ii].put_queue) for ii in range(len(job_shop.machinesWC5))})
        next_workstation(job, job_shop, env)


def job_pool_agent(job, job_shop, currentWC):
    func = eval('job_shop.storeWC' + str(currentWC))
    yield func.put(job)


def cfp_wc1(env, job_shop):
    while True:
        if len(job_shop.storeWC1.items) > 0:
            c = bid_winner(env, job_shop.storeWC1.items, machinesPerWC[0], 1, job_shop)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc2(env, job_shop):
    while True:
        if len(job_shop.storeWC2.items) > 0:
            c = bid_winner(env, job_shop.storeWC2.items, machinesPerWC[1], 2, job_shop)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc3(env, job_shop):
    while True:
        if len(job_shop.storeWC3.items) > 0:
            c = bid_winner(env, job_shop.storeWC3.items, machinesPerWC[2], 3, job_shop)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc4(env, job_shop):
    while True:
        if len(job_shop.storeWC4.items) > 0:
            c = bid_winner(env, job_shop.storeWC4.items, machinesPerWC[3], 4, job_shop)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc5(env, job_shop):
    while True:
        if len(job_shop.storeWC5.items) > 0:
            c = bid_winner(env, job_shop.storeWC5.items, machinesPerWC[4], 5, job_shop)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)


def no_in_system(R):
    """Total number of jobs in the resource R"""
    return len(R.put_queue) + len(R.users)


def source(env, number1, interval, job_shop):
    if not noJobCap:  # If there is a limit on the number of jobs
        for ii in range(number1):
            job = New_Job('job%02d' % ii, env)
            firstWC = operationOrder[job.type - 1][0]
            d = job_pool_agent(job, job_shop, firstWC)
            env.process(d)
            tib = random.expovariate(1.0 / interval)
            yield env.timeout(tib)
    else:
        while True:  # Needed for infinite case as True refers to "until".
            ii = number1
            number1 += 1
            job = New_Job('job%02d' % ii, env)
            firstWC = operationOrder[job.type - 1][0]
            d = job_pool_agent(job, job_shop, firstWC)
            env.process(d)
            tib = random.expovariate(1.0 / interval)
            yield env.timeout(tib)


class jobShop:
    def __init__(self, env, weights):
        machine_wc1 = {ii: Resource(env) for ii in range(machinesPerWC[0])}
        machine_wc2 = {ii: Resource(env) for ii in range(machinesPerWC[1])}
        machine_wc3 = {ii: Resource(env) for ii in range(machinesPerWC[2])}
        machine_wc4 = {ii: Resource(env) for ii in range(machinesPerWC[3])}
        machine_wc5 = {ii: Resource(env) for ii in range(machinesPerWC[4])}

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

        self.storeWC1 = job_poolwc1
        self.storeWC2 = job_poolwc2
        self.storeWC3 = job_poolwc3
        self.storeWC4 = job_poolwc4
        self.storeWC5 = job_poolwc5

        self.test_weights = weights
        self.makespan = []
        self.makespan.append(30)
        self.tardiness = []


class New_Job:
    def __init__(self, name, env):
        jobType = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)
        # self.type = 2
        self.type = jobType[0]
        self.name = name
        self.currentOperation = 1
        self.processingTime = np.zeros(numberOfOperations[self.type - 1])
        self.dueDate = np.zeros(numberOfOperations[self.type - 1] + 1)
        self.dueDate[0] = env.now
        # self.processingTime = processingTimes[self.type - 1]
        self.operationOrder = operationOrder[self.type - 1]
        self.numberOfOperations = numberOfOperations[self.type - 1]
        # self.dueDate = env.now
        for ii in range(self.numberOfOperations):
            meanPT = processingTimes[self.type - 1][ii]
            # self.processingTime[ii] = meanPT
            self.processingTime[ii] = random.gammavariate(2, meanPT / 2)
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * dueDateTightness


# Setup and start the simulation
weights = np.random.rand(sum(machinesPerWC), 9)
# print(weights)
seed = random.randrange(sys.maxsize)
rng = random.Random(seed)
# print("Seed was:", seed)
env = Environment()

job_shop = jobShop(env, weights)
# job_pool_wc1 = simpy.FilterStore(env)
env.process(source(env, number, arrivalMean, job_shop))
env.process(cfp_wc1(env, job_shop))
env.process(cfp_wc2(env, job_shop))
env.process(cfp_wc3(env, job_shop))
env.process(cfp_wc4(env, job_shop))
env.process(cfp_wc5(env, job_shop))
# print(f'\n Running simulation with seed {seed}... \n')
env.run(until=maxTime)
# print('\n Done \n')

# print("Total tardiness is: ", sum(currentTardiness))
print("Total makespan is: ", max(job_shop.makespan))
pos_count = len(list(filter(lambda x: (x >= 0), currentTardiness)))
print("Number of late jobs is: ", pos_count)

#
# df1 = pd.DataFrame(QueuesWC1)
# df2 = pd.DataFrame(QueuesWC2)
# df3 = pd.DataFrame(QueuesWC3)
# df4 = pd.DataFrame(QueuesWC4)
# df5 = pd.DataFrame(QueuesWC5)
#
# nrow = 2
# ncol = 3
#
# fig, axes = plt.subplots(nrow, ncol)
# # plt.title("Individual Queue Loads WC1")
#
# axes1 = df1.plot(ax=axes[0, 0])
# axes2 = df2.plot(ax=axes[0, 1])
# axes3 = df3.plot(ax=axes[0, 2])
# axes4 = df4.plot(ax=axes[1, 0])
# axes5 = df5.plot(ax=axes[1, 1])
#
# axes1.set_ylabel('Queue Length')
# axes1.set_xlabel('Job')
#
# axes2.set_ylabel('Queue Length')
# axes2.set_xlabel('Job')
#
# axes3.set_ylabel('Queue Length')
# axes3.set_xlabel('Job')
#
# axes4.set_ylabel('Queue Length')
# axes4.set_xlabel('Job')
#
# axes5.set_ylabel('Queue Length')
# axes5.set_xlabel('Job')
#
# plt.show()

# # # plot counter
# count = 0
# for r in range(nrow):
#     for c in range(ncol):
#         df_list[count].plot(ax=axes[r])
#         count += 1

# min_objective = []
# max_objective = []
#
# objective_positive = []
# objective_minimum = []
#
# m_0 = 0
# v_0 = 0
# t = 0
# m_t_mean = np.zeros((16, 9, 500))
# v_t_mean = np.zeros((16, 9, 500))
#
# m_t_std = np.zeros((16, 9, 500))
# v_t_std = np.zeros((16, 9, 500))
#
# # eta = np.zeros((12, 16, noAttributes))
#
# # seed = random.randrange(sys.maxsize)
# # rng = random.Random(seed)
# # to run GUI event loop
# # dat = [0, 1]
# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # Ln, = ax.plot(dat)
# # ax.set_xlim([0, 50])
# # ax.set_ylim([0, 300])
# # plt.ion()
# # plt.show()
#
# # setting title
# # plt.title("Geeks For Geeks", fontsize=20)
# #
# # # setting x-axis label and y-axis label
# # plt.xlabel("X-axis")
# # plt.ylabel("Y-axis")
# final_objective = []
# # print(mean_weight)
# for num_sim in range(500):
#     objective = np.zeros((12, 2))
#     # min_objective = []
#     eta = np.zeros((12, sum(machinesPerWC), noAttributes))
#     for ii in range(12):
#         test_weights_pos = np.zeros((sum(machinesPerWC), noAttributes))
#         test_weights_min = np.zeros((sum(machinesPerWC), noAttributes))
#         for m in range(sum(machinesPerWC)):
#             for j in range(noAttributes):
#                 eta[ii][m][j] = random.normalvariate(0, np.exp(std_weight[m][j]))
#                 test_weights_pos[m][j] = mean_weight[m][j] + eta[ii][m][j]
#                 test_weights_min[m][j] = mean_weight[m][j] - eta[ii][m][j]
#
#         seed = random.randrange(sys.maxsize)
#         rng = random.Random(seed)
#         # print("Seed was:", seed)
#         env = Environment()
#
#         # test_weights = test_weights_pos
#         job_shop = jobShop(env, test_weights_pos)
#         env.process(source(env, number, arrivalMean, job_shop))
#         env.process(cfp_wc1(env, job_shop))
#         env.process(cfp_wc2(env, job_shop))
#         env.process(cfp_wc3(env, job_shop))
#         env.process(cfp_wc4(env, job_shop))
#         env.process(cfp_wc5(env, job_shop))
#         env.run(until=maxTime)
#
#         makespan_result_pos = np.mean(currentCycleTime)
#         currentTardiness_result_pos = np.mean(currentTardiness)
#         # print(currentCycleTime)
#         makespan = []
#         currentTardiness = []
#         currentCycleTime = []
#         # print(makespan_result_pos, currentTardiness_result_pos)
#         objective[ii][0] = makespan_result_pos
#         # print(test_weights)
#
#         # print(makespan_result_pos, currentTardiness_result_pos)
#
#         seed = random.randrange(sys.maxsize)
#         rng = random.Random(seed)
#         # print("Seed was:", seed)
#         env = Environment()
#
#         # test_weights = test_weights_min
#         job_shop = jobShop(env, test_weights_min)
#         env.process(source(env, number, arrivalMean, job_shop))
#         env.process(cfp_wc1(env, job_shop))
#         env.process(cfp_wc2(env, job_shop))
#         env.process(cfp_wc3(env, job_shop))
#         env.process(cfp_wc4(env, job_shop))
#         env.process(cfp_wc5(env, job_shop))
#         env.run(until=maxTime)
#         makespan_result_min = np.mean(currentCycleTime)
#         currentTardiness_result_min = np.mean(currentTardiness)
#         currentTardiness = []
#         currentCycleTime = []
#
#         objective[ii][1] = makespan_result_min
#
#         # print(makespan_result_min, currentTardiness_result_min)
#     makespan = []
#     objective_norm = np.zeros((12, 2))
#     # Normalise the current populations performance
#     for ii in range(12):
#         value_obj_pos = (objective[ii][0] - np.mean(objective, axis=0)[0]) / np.std(objective, axis=0)[0]
#         objective_norm[ii][0] = value_obj_pos
#
#         value_obj_min = (objective[ii][1] - np.mean(objective, axis=0)[1]) / np.std(objective, axis=0)[1]
#         objective_norm[ii][1] = value_obj_min
#
#     # Calculate the gradients
#     # gradient_sum = np.zeros(12)
#     # for i in range(12):
#     #     gradient_sum[i] = (objective_positive[i] - objective_minimum[i]) / 2
#
#     delta_mean_final = np.zeros((sum(machinesPerWC), 9))
#     delta_std_final = np.zeros((sum(machinesPerWC), 9))
#     for m in range(sum(machinesPerWC)):
#         for j in range(noAttributes):
#             delta_mean = 0
#             delta_std = 0
#             for ii in range(12):
#                 delta_mean += ((objective_norm[ii][0] - objective_norm[ii][1]) / 2) * eta[ii][m][j] / np.exp(
#                     std_weight[m][j])
#
#                 # print(delta_mean)
#
#                 delta_std += ((objective_norm[ii][0] - objective_norm[ii][1]) / 2) * (eta[ii][m][j] ** 2 - np.exp(
#                     std_weight[m][j])) / np.exp(std_weight[m][j])
#
#             delta_mean_final[m][j] = delta_mean / 12
#             delta_std_final[m][j] = delta_std / 12
#         # print(delta_mean_final[m][j])
#
#     # for j in range(noAttributes):
#     #     for m in range(sum(machinesPerWC)):
#     #         mean_weight[m][j] -= alpha_mean * delta_mean_final[m][j]
#     #         std_weight[m][j] -= alpha_std * delta_std_final[m][j]
#     #
#     # alpha_mean = 0.1 / np.sqrt(num_sim + 1)
#     # alpha_std = 0.025 / np.sqrt(num_sim + 1)
#     # print(num_sim, np.mean(min_objective))
#
#     t += 1
#     # g_t = mean_weight[m][j]
#
#     for m in range(sum(machinesPerWC)):
#         for j in range(noAttributes):
#             g_t = delta_mean_final[m][j]
#             m_t_mean[m][j][t] = (beta_1 * m_t_mean[m][j][t - 1] + (1 - beta_1) * g_t)
#             v_t_mean[m][j][t] = (beta_2 * v_t_mean[m][j][t - 1] + (1 - beta_2) * g_t ** 2)
#             m_hat_t = (m_t_mean[m][j][t] / (1 - beta_1 ** t))
#             v_hat_t = (v_t_mean[m][j][t] / (1 - beta_2 ** t))
#             mean_weight[m][j] -= alpha_mean * m_hat_t / (np.sqrt(v_hat_t) + 10 ** -8)
#
#             g_t = delta_std_final[m][j]
#             m_t_std[m][j][t] = (beta_1 * m_t_std[m][j][t - 1] + (1 - beta_1) * g_t)
#             v_t_std[m][j][t] = (beta_2 * v_t_std[m][j][t - 1] + (1 - beta_2) * g_t ** 2)
#             m_hat_t = (m_t_std[m][j][t] / (1 - beta_1 ** t))
#             v_hat_t = (v_t_std[m][j][t] / (1 - beta_2 ** t))
#             std_weight[m][j] -= alpha_std * m_hat_t / (np.sqrt(v_hat_t) + 10 ** -8)
#
#         alpha_mean = 0.1 / np.sqrt(num_sim + 1)
#         alpha_std = 0.025 / np.sqrt(num_sim + 1)
#         beta_1 = 0.9 / np.sqrt(num_sim + 1)
#         beta_2 = 0.999 / np.sqrt(num_sim + 1)
#
#     # final_objective.append(np.mean(np.mean(objective, axis=0)))
#     # Ln.set_ydata(final_objective)
#     # Ln.set_xdata(range(len(final_objective)))
#     # plt.pause(1)
#
#     print(num_sim, np.mean(np.mean(objective, axis=0)))
