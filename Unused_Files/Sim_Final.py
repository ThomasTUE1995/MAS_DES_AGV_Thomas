"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS in Jade to see the effects.
"""
import random
from collections import defaultdict
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
# import matplotlib.cbook
import pandas as pd
import simpy
from simpy import *

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit
# Old job settings
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]] # Workstation where each operation can be processed
numberOfOperations = [4, 3, 5]
machinesPerWC = [4, 2, 5, 3, 2]
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]
setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]
mean_setup = [0.515, 0.306, 0.515, 0.429, 0.306] # Mean setup time dependent on demand and job types
demand = [0.2, 0.5, 0.3]

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 7 # Number of Attributes considered for bidding rule
noAttributesJob = 3 # Number of attributes considered for sequencing rule
totalAttributes = noAttributes + noAttributesJob

no_generation = 500


def list_duplicates(seq):
    """ Checks for duplicates in a list and returns there index """
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


def bid_winner(env, job, noOfMachines, currentWC, job_shop, last_job, makespan_currentWC, machine, store):
    """ This partially models both the Job Pool Agents and the Machine Agents. Here, the bids for the MA are calculated,
    and the winner for each job is determined. """
    current_bid = [0] * noOfMachines
    current_job = [0] * noOfMachines
    best_bid = []
    best_job = []
    no_of_jobs = len(job)

    total_rp = [0] * no_of_jobs
    for j in range(no_of_jobs):
        total_rp[j] = (remain_processing_time(job[j]))
    # Determine the bids for each job
    for jj in range(noOfMachines):
        pool = len(machine[jj].items)
        expected_start = expected_start_time(jj, currentWC, machine)
        start_time = max(env.now, makespan_currentWC[jj] + expected_start)
        new_bid = [0] * no_of_jobs
        i = 0
        for j in job:
            attributes = bid_calculation(job_shop.test_weights, machine_number_WC[currentWC - 1][jj],
                                         j.processingTime[j.currentOperation - 1], start_time, j.currentOperation,
                                         j.numberOfOperations,
                                         j.dueDate[j.currentOperation], total_rp[i], pool,
                                         j.dueDate[j.numberOfOperations], env.now,
                                         j.priority)
            new_bid[i] = sum(attributes)
            if j.number > 499:
                bid_structure = [currentWC, jj, new_bid[i], j.name + str(j.currentOperation), env.now]
                bid_structure.extend(attributes)
                job_shop.bids.append(bid_structure)
            i += 1

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
        if job[int(dup[0])].number > 499:
            job_shop.winningbids.append(
                [currentWC, bestmachine, job[int(dup[0])].name + str(job[int(dup[0])].currentOperation),
                 job[int(dup[0])].type, job[int(dup[0])].priority])

    # Put the jobs in the queue of the winning machines
    for ii in range(len(best_job)):
        put_job_in_queue(currentWC, best_bid[ii], job[best_job[ii]], job_shop, env, machine)

    # Remove job from JPA
    for ii in reversed(best_job):
        yield store.get(lambda mm: mm == job[ii])


def expected_start_time(jj, currentWC, machine):
    """ Used to estimate the start time if needed in a bidding attribute"""
    # machine = eval('job_shop.machinesWC' + str(currentWC))
    extra_start_time = 0
    for kk in range(len(machine[jj].items)):
        current_job = machine[jj].items[kk]
        extra_start_time += (current_job.processingTime[current_job.currentOperation - 1] + mean_setup[currentWC - 1])

    return extra_start_time


def minimum_setup_time(pool, job):
    """ Used to caluclate the minimum setup for a job for a machine given its current queue"""
    min_setup = []
    for jj in pool:
        min_setup.append(setupTime[job.type - 1][jj.type - 1])

    return min(min_setup)



def remain_processing_time(job):
    """ Calculate the remaining processing time of a job"""
    total_rp = 0
    for ii in range(job.currentOperation - 1, job.numberOfOperations):
        total_rp += job.processingTime[ii]

    return total_rp


def next_workstation(job, job_shop, env, all_store):
    """ This function either sends the next operation of a job to its next WC or,
     removes the job from the system and appends the objective function"""
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        store = all_store[nextWC - 1]
        store.put(job)
    else:
        finish_time = env.now
        job_shop.tardiness[job.number] = max(job.priority * (finish_time - job.dueDate[job.numberOfOperations]), 0)
        job_shop.WIP -= 1
        job_shop.makespan[job.number] = finish_time - job.dueDate[0]

        # If the 2500th job has finished and jobs 500 to 2500 have also finished, stop the simulation
        if job.number > 2499:
            if np.count_nonzero(job_shop.makespan[499:2499]) == 2000:
                job_shop.finishtime = env.now
                job_shop.end_event.succeed()

        # if (job_shop.WIP > 2500) | (env.now > 30_000):
        #     # print(job_shop.WIP)
        #     job_shop.end_event.succeed()
        #     job_shop.early_termination = 1


def bid_calculation(weights_new, machinenumber, processing_time, start_time,
                    current, total, due_date_operation, total_rp, queue_length, due_date, now, job_priority,
                    ):
    attribute = [0] * noAttributes
    attribute[0] = processing_time / 8.75 * weights_new[machinenumber - 1][0]
    attribute[1] = (current - 1) / (5 - 1) * weights_new[machinenumber - 1][1]
    attribute[2] = (due_date - now + 400) / (97.5 + 400) * \
                   weights_new[machinenumber - 1][2]
    attribute[3] = total_rp / 21.25 * weights_new[machinenumber - 1][3]
    attribute[4] = (((due_date - now) / total_rp) + 7) / (12 + 7) * weights_new[machinenumber - 1][4]  # Critical Ratio
    attribute[5] = (job_priority - 1) / (10 - 1) * weights_new[machinenumber - 1][5]  # Job Weight
    attribute[6] = queue_length / 25 * weights_new[machinenumber - 1][6]  # Current Workload of Machine

    return attribute


def normalize(value, max_value, min_value):
    return (value - min_value) / (max_value - min_value)


def set_makespan(current_makespan, job, last_job, env, setup_time):
    add = current_makespan + job.processingTime[job.currentOperation - 1] + setup_time

    new = env.now + job.processingTime[job.currentOperation - 1] + setup_time

    return max(add, new)


def put_job_in_queue(currentWC, choice, job, job_shop, env, machines):
    machines[choice].put(job)
    """ If there was no job in the queue, a condition flag is triggered, 
    sending a signal to the machine that a new job is available for processing"""
    if not job_shop.condition_flag[currentWC - 1][choice].triggered:
        job_shop.condition_flag[currentWC - 1][choice].succeed()


def choose_job_queue(weights_new_job, machinenumber, processing_time, due_date, env, setup_time, job_priority):
    """ Calculates priorities of jobs in machines queue """
    attribute_job = [0] * noAttributesJob
    attribute_job[2] = setup_time / 1.25 * weights_new_job[machinenumber - 1][noAttributes + 2]
    attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    attribute_job[0] = (due_date - processing_time - setup_time - env.now - (-200)) / (100 + 200) * \
                       weights_new_job[machinenumber - 1][noAttributes]
    return sum(attribute_job)


def machine_processing(job_shop, current_WC, machine_number, env, weights_new, relative_machine, last_job, machine,
                       makespan, all_store, schedule):
    """ This models the processing of machines. Note that this is done as an object instead of a Simpy Resoruce,
    as the queue of a resource cannot be manipulated. """
    while True:
        relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
        # print( machine_number)
        if machine[relative_machine].items:
            setup_time = []
            priority_list = []
            if (len(machine[relative_machine].items) == 1) | (last_job[relative_machine] == 0):
                ind_processing_job = 0
                setup_time.append(0)
            else:
                for job in machine[relative_machine].items:
                    setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
                    # priority_list.append(job.dueDate[job.numberOfOperations])
                    job_queue_priority = choose_job_queue(weights_new, machine_number,
                                                          job.processingTime[job.currentOperation - 1],
                                                          job.dueDate[job.currentOperation], env, setuptime,
                                                          job.priority)
                    priority_list.append(job_queue_priority)
                    setup_time.append(setuptime)
                ind_processing_job = priority_list.index(max(priority_list))

            # ind_processing_job = 0
            next_job = machine[relative_machine].items[ind_processing_job]
            setuptime = setup_time[ind_processing_job]
            total_proc_time = next_job.processingTime[next_job.currentOperation - 1] + setuptime
            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, last_job[relative_machine],
                                                      env, setuptime)
            last_job[relative_machine] = next_job.type
            schedule.append(
                [relative_machine, next_job.name, next_job.type, makespan[relative_machine], total_proc_time, setuptime, env.now,
                 next_job.priority])
            machine[relative_machine].items.remove(next_job)
            yield env.timeout(total_proc_time)
            next_workstation(next_job, job_shop, env, all_store)
        else:
            # If there are no jobs available, wait until a job becomes available
            yield job_shop.condition_flag[current_WC - 1][relative_machine]
            job_shop.condition_flag[current_WC - 1][relative_machine] = simpy.Event(env)


def cfp_wc(env, last_job, machine, makespan, store, job_shop, currentWC):
    """ This models another part of the JPA. It is used to send of Call For Proposals."""
    while True:
        if store.items:
            c = bid_winner(env, store.items, machinesPerWC[currentWC], currentWC + 1, job_shop, last_job, makespan,
                           machine, store)
            env.process(c)
        #Wait for a certain amount of time before a new CFP is sent out
        wait_time = 0.5
        yield env.timeout(wait_time)


def no_in_system(R):
    """Total number of jobs in the resource R"""
    return len(R.put_queue) + len(R.users)


def source(env, number1, interval, job_shop, due_date_setting):
    """" Define the soruce, the rate at which jobs enter the system."""
    if not noJobCap:  # If there is a limit on the number of jobs
        for ii in range(number1):
            job = New_Job('job%02d' % ii, env, ii, due_date_setting)
            firstWC = operationOrder[job.type - 1][0]
            store = eval('job_shop.storeWC' + str(firstWC))
            store.put(job)
            # d = job_pool_agent(job, firstWC, job_shop, store)
            # env.process(d)
            arrival_time = random.expovariate(1.0 / interval)
            yield env.timeout(arrival_time)
    else:
        while True:  # If there is no limit on the amount of jobs"
            ii = number1
            number1 += 1
            job = New_Job('job%02d' % ii, env, ii, due_date_setting)
            job_shop.tardiness.append(-1)
            job_shop.makespan.append(0)
            job_shop.WIP += 1
            firstWC = operationOrder[job.type - 1][0]
            store = eval('job_shop.storeWC' + str(firstWC))
            store.put(job)
            arrival_time = random.expovariate(1.0 / interval)
            yield env.timeout(arrival_time)


class jobShop:
    """ Define the job shop and alls its processes, stores, resources and other attributes"""
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

        self.scheduleWC1 = []
        self.scheduleWC2 = []
        self.scheduleWC3 = []
        self.scheduleWC4 = []
        self.scheduleWC5 = []

        self.condition_flag = []
        for wc in range(len(machinesPerWC)):
            Q = eval('self.QueuesWC' + str(wc + 1))
            self.condition_flag.append([])
            for ii in range(machinesPerWC[wc]):
                Q.append([])
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
        self.WIP = 0
        self.early_termination = 0

        self.bids = []
        self.winningbids = []


class New_Job:
    """ Assign attributes to new jobs that are inserted into the system"""
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


def do_simulation_with_weights(mean_weight_new, arrivalMean, due_date_tightness, iter):
    """ Given weights for the various attributes, run the simulation"""
    env = Environment()
    job_shop = jobShop(env, mean_weight_new)
    env.process(source(env, number, arrivalMean, job_shop, due_date_tightness))
    all_stores = []

    for wc in range(len(machinesPerWC)):
        last_job = eval('job_shop.last_job_WC' + str(wc + 1))
        machine = eval('job_shop.machinesWC' + str(wc + 1))
        makespan = eval('job_shop.makespanWC' + str(wc + 1))
        store = eval('job_shop.storeWC' + str(wc + 1))
        schedule = eval('job_shop.scheduleWC' + str(wc + 1))
        all_stores.append(store)

        env.process(cfp_wc(env, last_job, machine, makespan, store, job_shop, wc))

        for ii in range(machinesPerWC[wc]):
            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, ii, last_job,
                                   machine, makespan, all_stores, schedule))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)
    objective = np.mean(job_shop.tardiness[499:2499])

    return np.mean(objective)


if __name__ == '__main__':
    df = pd.read_csv('Runs/Attribute_Runs/Run-weights-NoSetup1-90-4-5000-7-3.csv', header=None)
    weights = df.values.tolist()

    no_runs = 500
    # obj = do_simulation_with_weights(weights, 1.5429, 4, 1)
    obj = np.zeros(no_runs)
    jobshop_pool = Pool(processes=no_runs)
    seeds = range(no_runs)
    func1 = partial(do_simulation_with_weights, weights, 1.4572, 4)
    makespan_per_seed = jobshop_pool.map(func1, seeds)
    print(makespan_per_seed)
    for h in range(no_runs):
        obj[h] = makespan_per_seed[h]

    print(np.mean(obj), np.std(obj))
