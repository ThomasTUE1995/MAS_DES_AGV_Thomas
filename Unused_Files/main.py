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

number = 5000  # Max number of jobs if infinite is false
noJobCap = False  # For infinite
maxTime = 300.0  # Runtime limit
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
noAttributes = 9

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 8
# weights = np.zeros((16, noAttributes))
mean_weight = np.zeros(noAttributes)
std_weight = np.zeros(noAttributes)
std_weight = std_weight + np.log(0.3)
alpha_mean = 0.1
alpha_std = 0.025
beta_1 = 0.9
beta_2 = 0.999
# eta = 10 ** -8
no_generation = 500
population_size = 24
for ii in range(sum(machinesPerWC)):
    mean_weight[0] = -0.5

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


def bid_winner(env, job, noOfMachines, currentWC):
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
    for i in range(len(job)):
        new_job = job[i]
        due_date = new_job.dueDate[new_job.currentOperation - 1]
        currentOperation = new_job.currentOperation
        totalOperations = new_job.numberOfOperations
        job_weight = 1
        proc_time = new_job.processingTime[currentOperation - 1]
        for j in range(noOfMachines):
            if last_job[j] != 0:
                # time = new_job.processingTime[new_job.currentOperation - 1] + setupTime[new_job.type - 1][
                #     int(last_job[j]) - 1]
                setup_time = setupTime[new_job.type - 1][int(last_job[j]) - 1]
            else:
                # time = new_job.processingTime[new_job.currentOperation - 1]
                setup_time = 0

            # new_bid = 1 / (no_in_system(queue[j]) + new_job.processingTime[new_job.currentOperation - 1])
            queue_length = no_in_system(queue[j])

            new_bid = bid_calculation(test_weights_new, j + 1, no_jobs,
                                      proc_time, setup_time,
                                      queue_length, currentWC, env.now, maxduedate, currentOperation, totalOperations,
                                      due_date, job_weight, total_machines)

            if i == 0:
                current_bid[j] = new_bid
                current_job[j] = i
                current_time[j] = new_job.processingTime[new_job.currentOperation - 1] + setup_time
            elif new_bid >= current_bid[j]:
                current_bid[j] = new_bid
                current_job[j] = i
                current_time[j] = new_job.processingTime[new_job.currentOperation - 1] + setup_time

    # Determine the winning bids
    for dup in sorted(list_duplicates(current_job)):
        bestbid = -1
        bestmachine = 0
        bestime = 0
        for i in dup[1]:
            if i == 0:
                bestbid = current_bid[i]
                bestmachine = i
                bestime = current_time[i]
            if bestbid < current_bid[i]:
                bestbid = current_bid[i]
                bestmachine = i
                bestime = current_time[i]
        best_bid.append(bestmachine)  # Machine winner
        best_time.append(bestime)
        best_job.append(int(dup[0]))  # Job to be processed

    for i in range(len(best_job)):
        c = globals()['WC' + str(currentWC)](env, job[best_job[i]], job_shop, best_bid[i], best_time[i])
        env.process(c)
        removed_job.append(best_job[i])

    for i in reversed(removed_job):
        func = eval('job_shop.storeWC' + str(currentWC))
        yield func.get(lambda m: m == list_jobs[i])


def next_workstation(job, job_shop, env):
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        c = job_pool_agent(job, job_shop, nextWC)
        env.process(c)
    else:
        finish_time = env.now
        currentTardiness.append(finish_time - job.dueDate[job.numberOfOperations])
        makespan.append(finish_time)
        # print(finish_time - job.dueDate[0], job.currentOperation)
        currentCycleTime.append(finish_time - job.dueDate[0])


def bid_calculation(weights_new, machinenumber, pool, processing_time, setup_time, queue, wc, proc_time, max_due_date,
                    current, total, due_date, job_weight, machines):
    attribute = [0] * noAttributes
    attribute[0] = normalize(processing_time, 8.75, 0) * weights_new[0]
    attribute[1] = normalize(queue, 100, 0) * weights_new[1]
    attribute[2] = normalize((total - current), total, 1) * weights_new[2]
    attribute[3] = normalize(setup_time, 0.25, 0) * weights_new[3]
    attribute[4] = normalize((due_date - (processing_time + setup_time) - proc_time), 100, -100) * weights_new[4]
    attribute[5] = normalize(pool, 100, 0) * weights_new[5]
    attribute[6] = normalize(machines, 5, 1) * weights_new[6]
    attribute[7] = normalize(max_due_date, 100, 0) * weights_new[7]

    total_bid = 0
    for jj in range(noAttributes):
        total_bid += attribute[jj]

    # print(attribute[4])
    return total_bid


def normalize(value, max_value, min_value):
    return (value - min_value) / (max_value - min_value)


def WC1(env, job, job_shop, choice, tib):
    QueuesWC1.append({i: len(job_shop.machinesWC1[i].put_queue) for i in range(len(job_shop.machinesWC1))})
    last_job_WC1[choice] = job.type
    with job_shop.machinesWC1[choice].request() as req:
        # Wait in queue
        yield req
        yield env.timeout(tib)
        QueuesWC1.append({i: len(job_shop.machinesWC1[i].put_queue) for i in range(len(job_shop.machinesWC1))})
        next_workstation(job, job_shop, env)


def WC2(env, job, job_shop, choice, tib):
    QueuesWC2.append({i: len(job_shop.machinesWC2[i].put_queue) for i in range(len(job_shop.machinesWC2))})
    last_job_WC2[choice] = job.type
    with job_shop.machinesWC2[choice].request() as req2:
        # Wait in queue
        yield req2
        yield env.timeout(tib)
        QueuesWC2.append({i: len(job_shop.machinesWC2[i].put_queue) for i in range(len(job_shop.machinesWC2))})
        next_workstation(job, job_shop, env)


def WC3(env, job, job_shop, choice, tib):
    QueuesWC3.append({i: len(job_shop.machinesWC3[i].put_queue) for i in range(len(job_shop.machinesWC3))})
    last_job_WC3[choice] = job.type
    with job_shop.machinesWC3[choice].request() as req2:
        # Wait in queue
        yield req2
        yield env.timeout(tib)
        QueuesWC3.append({i: len(job_shop.machinesWC3[i].put_queue) for i in range(len(job_shop.machinesWC3))})
        next_workstation(job, job_shop, env)


def WC4(env, job, job_shop, choice, tib):
    QueuesWC4.append({i: len(job_shop.machinesWC4[i].put_queue) for i in range(len(job_shop.machinesWC4))})
    last_job_WC4[choice] = job.type
    with job_shop.machinesWC4[choice].request() as req2:
        # Wait in queue
        yield req2
        yield env.timeout(tib)
        QueuesWC4.append({i: len(job_shop.machinesWC4[i].put_queue) for i in range(len(job_shop.machinesWC4))})
        next_workstation(job, job_shop, env)


def WC5(env, job, job_shop, choice, tib):
    QueuesWC5.append({i: len(job_shop.machinesWC5[i].put_queue) for i in range(len(job_shop.machinesWC5))})
    last_job_WC5[choice] = job.type
    with job_shop.machinesWC5[choice].request() as req2:
        # Wait in queue
        yield req2
        yield env.timeout(tib)
        QueuesWC5.append({i: len(job_shop.machinesWC5[i].put_queue) for i in range(len(job_shop.machinesWC5))})
        next_workstation(job, job_shop, env)


def job_pool_agent(job, job_shop, currentWC):
    func = eval('job_shop.storeWC' + str(currentWC))
    yield func.put(job)


def cfp_wc1(env, job_shop):
    while True:
        if len(job_shop.storeWC1.items) > 0:
            c = bid_winner(env, job_shop.storeWC1.items, machinesPerWC[0], 1)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc2(env, job_shop):
    while True:
        if len(job_shop.storeWC2.items) > 0:
            c = bid_winner(env, job_shop.storeWC2.items, machinesPerWC[1], 2)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc3(env, job_shop):
    while True:
        if len(job_shop.storeWC3.items) > 0:
            c = bid_winner(env, job_shop.storeWC3.items, machinesPerWC[2], 3)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc4(env, job_shop):
    while True:
        if len(job_shop.storeWC4.items) > 0:
            c = bid_winner(env, job_shop.storeWC4.items, machinesPerWC[3], 4)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def cfp_wc5(env, job_shop):
    while True:
        if len(job_shop.storeWC5.items) > 0:
            c = bid_winner(env, job_shop.storeWC5.items, machinesPerWC[4], 5)
            env.process(c)
        tib = 1.0
        yield env.timeout(tib)


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)


def no_in_system(R):
    """Total number of jobs in the resource R"""
    return len(R.put_queue) + len(R.users)


def source(env, number, interval, job_shop):
    if not noJobCap:  # If there is a limit on the number of jobs
        for i in range(number):
            job = New_Job('job%02d' % i)
            firstWC = operationOrder[job.type - 1][0]
            d = job_pool_agent(job, job_shop, firstWC)
            env.process(d)
            t = random.expovariate(1.0 / interval)
            yield env.timeout(t)
    else:
        while True:  # Needed for infinite case as True refers to "until".
            i = number
            number += 1
            job = New_Job('job%02d' % i)
            firstWC = operationOrder[job.type - 1][0]
            d = job_pool_agent(job, job_shop, firstWC)
            env.process(d)
            t = random.expovariate(1.0 / interval)
            yield env.timeout(t)


class jobShop:
    def __init__(self, env, weights):
        machine_wc1 = {i: Resource(env) for i in range(machinesPerWC[0])}
        machine_wc2 = {i: Resource(env) for i in range(machinesPerWC[1])}
        machine_wc3 = {i: Resource(env) for i in range(machinesPerWC[2])}
        machine_wc4 = {i: Resource(env) for i in range(machinesPerWC[3])}
        machine_wc5 = {i: Resource(env) for i in range(machinesPerWC[4])}

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


class New_Job:
    def __init__(self, name):
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
        for i in range(self.numberOfOperations):
            meanPT = processingTimes[self.type - 1][i]
            self.processingTime[i] = random.gammavariate(3, meanPT / 3)
            self.dueDate[i + 1] = self.dueDate[i] + self.processingTime[i] * dueDateTightness


# Setup and start the simulation
weights = np.random.rand(noAttributes)
print(weights)
seed = random.randrange(sys.maxsize)
rng = random.Random(seed)
print("Seed was:", seed)
env = Environment()

job_shop = jobShop(env, weights)
env.process(source(env, number, arrivalMean, job_shop))
env.process(cfp_wc1(env, job_shop))
env.process(cfp_wc2(env, job_shop))
env.process(cfp_wc3(env, job_shop))
env.process(cfp_wc4(env, job_shop))
env.process(cfp_wc5(env, job_shop))
print(f'\n Running simulation with seed {seed}... \n')
env.run(until=maxTime)
print('\n Done \n')

print("Total tardiness is: ", sum(currentTardiness))
print("Total makespan is: ", max(makespan))
pos_count = len(list(filter(lambda x: (x >= 0), currentTardiness)))
print("Number of late jobs is: ", pos_count)

currentTardiness = []
makespan = []

df1 = pd.DataFrame(QueuesWC1)
df2 = pd.DataFrame(QueuesWC2)
df3 = pd.DataFrame(QueuesWC3)
df4 = pd.DataFrame(QueuesWC4)
df5 = pd.DataFrame(QueuesWC5)

nrow = 2
ncol = 3

fig, axes = plt.subplots(nrow, ncol)
# plt.title("Individual Queue Loads WC1")

axes1 = df1.plot(ax=axes[0, 0])
axes2 = df2.plot(ax=axes[0, 1])
axes3 = df3.plot(ax=axes[0, 2])
axes4 = df4.plot(ax=axes[1, 0])
axes5 = df5.plot(ax=axes[1, 1])

axes1.set_ylabel('Queue Length')
axes1.set_xlabel('Job')

axes2.set_ylabel('Queue Length')
axes2.set_xlabel('Job')

axes3.set_ylabel('Queue Length')
axes3.set_xlabel('Job')

axes4.set_ylabel('Queue Length')
axes4.set_xlabel('Job')

axes5.set_ylabel('Queue Length')
axes5.set_xlabel('Job')

plt.show()
