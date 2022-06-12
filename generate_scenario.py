import random
import numpy as np
import numpy.random
import pandas as pd


def new_scenario(max_workcenters, min_workcenters, no_of_jobs, total_machines, min_proc, max_proc, setup_factor, uti,
                 seed):
    numpy.random.seed(seed)

    operationOrder = []
    processingTime = []
    priority_per_job = np.random.uniform(low=1.0, high=10.0, size=no_of_jobs)
    operations_per_job = np.random.choice(range(min_workcenters, max_workcenters + 1), no_of_jobs)
    demand = np.random.choice(range(10), no_of_jobs)

    demand = [x / sum(demand) for x in demand]

    for i in operations_per_job:
        operationOrder.append(list(np.random.choice(range(1, max_workcenters + 1), i, replace=False)))

    for i in range(no_of_jobs):
        processingTime.append(list(np.random.uniform(min_proc, max_proc, operations_per_job[i])))

    workload_per_wc = np.zeros(max_workcenters)
    for i in range(no_of_jobs):
        for count, value in enumerate(operationOrder[i]):
            workload_per_wc[value - 1] += processingTime[i][count] * demand[i]

    division_per_workcenter = [x / sum(workload_per_wc) for x in workload_per_wc]
    machines_per_wc = [round(x * total_machines) for x in division_per_workcenter]
    machine_number_WC = []

    k = 1
    for i in range(max_workcenters):
        machine_number_WC.append(list(range(k, machines_per_wc[i] + k)))
        k += machines_per_wc[i]

    # Set setup times
    max_setup = setup_factor * max(max(processingTime))
    b = np.random.uniform(0, max_setup, size=(no_of_jobs, no_of_jobs))
    setupTime = (b + b.T) / 2

    for i in range(no_of_jobs):
        setupTime[i][i] = 0

    # Calculate arrival rate
    utilization = [uti]
    job_in_wc = []
    mean_setup = []
    for j in range(no_of_jobs):
        mean_setup.append(list(np.zeros(len(operationOrder[j]))))

    for w in range(max_workcenters):
        job_in_wc.append([])
        for j in range(no_of_jobs):
            if (w + 1) in operationOrder[j]:
                job_in_wc[w].append([j, operationOrder[j].index(w + 1)])

    for i in job_in_wc:
        for j in i:
            for k in i:
                mean_setup[j[0]][j[1]] += setupTime[j[0]][k[0]] * demand[k[0]]

    mean_processing_time = sum([np.mean(processingTime[i]) * demand[i] for i in range(no_of_jobs)])

    # mean_setup_time = sum([np.mean(mean_setup[i]) * demand[i] for i in range(no_of_jobs)])

    mean_operations_per_job = sum(operations_per_job[i] * demand[i] for i in range(no_of_jobs))
    arrival_rate = [mean_processing_time * mean_operations_per_job / (total_machines * i) for i in utilization]
    sumProcessingTime = [sum(processingTime[i]) for i in range(no_of_jobs)]

    number_of_AGVs = sum(machines_per_wc) * (
                (0.75 * (mean_operations_per_job * 2) + 1) / (mean_processing_time * mean_operations_per_job))
    number_of_AGVs_V2 = ((0.75 * (mean_operations_per_job * 2) + 1) / (arrival_rate[0] * utilization[0]))

    print("Arrival rate:", arrival_rate[0])
    print("Number of AGVs V1:", number_of_AGVs)
    print("Number of AGVs V2:", number_of_AGVs_V2)

    # Get maximum critical ratio
    max_ddt = 8
    CR = []
    DDT = []
    for j in range(no_of_jobs):
        CR.append([(sumProcessingTime[j] * max_ddt - sum(processingTime[j][0:i])) / sum(processingTime[j][i:]) for i in
                   range(operations_per_job[j] - 1)])

        DDT.append(sumProcessingTime[j] * max_ddt)

    maxval = max(map(max, CR))

    return processingTime, operationOrder, machines_per_wc, list(setupTime), demand, list(
        priority_per_job), arrival_rate, machine_number_WC, maxval, max(DDT)


# ================ Testing =============

test = False

if test:
    scenario = 'scenario_1'

    #              SCENARIO ------ WC - JT - MACH - PROC ---- AGVS -- MAXWIP - TIME - SEED - UTI
    #              =======================================================================================================
    situations = {'scenario_1': [[5, 2], 5, 16, [2, 9], [1, 1, 1, 1, 1], 250, 10_000, 150, 0.95],  # ARR 1.5804
                  'scenario_2': [[5, 2], 5, 16, [10, 50], [0, 0, 1, 0, 0], 300, 30_000, 150, 0.90],  # ARR 8.6249
                  'scenario_3': [[5, 2], 5, 32, [2, 9], [2, 2, 2, 2, 2], 350, 10_000, 150, 0.85],  # ARR 0.889
                  'scenario_4': [[5, 2], 5, 32, [10, 50], [0, 0, 2, 0, 0], 400, 30_000, 150, 0.90],
                  # ARR 5.1749   !--------
                  'scenario_5': [[5, 2], 20, 16, [2, 9], [2, 2, 2, 2, 2], 250, 10_000, 150, 0.80],  # ARR 1.8285
                  'scenario_6': [[5, 2], 20, 16, [10, 50], [2, 2, 2, 2, 2], 300, 40_000, 150, 0.75],
                  # ARR 10.617   TODO: DO AGAIN!
                  'scenario_7': [[5, 2], 20, 32, [2, 9], [2, 2, 2, 2, 2], 400, 10_000, 150, 0.75],  # ARR 0.9752
                  'scenario_8': [[5, 2], 20, 32, [10, 50], [2, 2, 2, 2, 2], 300, 30_000, 150, 0.55],  # ARR 7.2389
                  'scenario_9': [[10, 2], 5, 16, [2, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 400, 20_000, 150, 0.45],
                  # ARR 5.006
                  'scenario_10': [[10, 2], 5, 16, [10, 50], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 500, 40_000, 150, 0.60],
                  # ARR 18.781
                  'scenario_11': [[10, 2], 5, 32, [2, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 500, 10_000, 150, 0.45],
                  # ARR 2.503
                  'scenario_12': [[10, 2], 5, 32, [10, 50], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 250, 10_000, 150, 0.90],
                  'scenario_13': [[10, 2], 20, 16, [2, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 500, 10_000, 150, 0.45],
                  # ARR 4.253
                  'scenario_14': [[10, 2], 20, 16, [10, 50], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 250, 10_000, 150, 0.90],
                  'scenario_15': [[10, 2], 20, 32, [2, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 500, 10_000, 150, 0.55],
                  'scenario_16': [[10, 2], 20, 32, [10, 50], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 250, 10_000, 150, 0.90]}

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

    new_scenario(max_workcenters, min_workcenters, no_of_jobs, total_machines, min_proc, max_proc, setup_factor, uti,
                 seed)
