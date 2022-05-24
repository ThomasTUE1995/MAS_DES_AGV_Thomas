import random
import numpy as np
import numpy.random
import pandas as pd


def new_scenario(max_workcenters, min_workcenters, no_of_jobs, total_machines, min_proc, max_proc, setup_factor, uti):
    np_seed = numpy.random.seed(150)

    operationOrder = []
    processingTime = []
    priority_per_job = np.random.uniform(low=1.0, high=10.0, size=no_of_jobs)
    operations_per_job = np.random.choice(range(min_workcenters, max_workcenters + 1), no_of_jobs)
    demand = np.random.choice(range(10), no_of_jobs)
    demand = [x / sum(demand) for x in demand]

    for i in operations_per_job:
        operationOrder.append(list(np.random.choice(range(1, max_workcenters + 1), i, replace=False)))

    for i in range(no_of_jobs):
        # print(operations_per_job[i])
        processingTime.append(list(np.random.uniform(min_proc, max_proc, operations_per_job[i])))

    # print(operations_per_job)
    # print(processingTime)

    workload_per_wc = np.zeros(max_workcenters)
    for i in range(no_of_jobs):
        for count, value in enumerate(operationOrder[i]):
            workload_per_wc[value - 1] += processingTime[i][count] * demand[i]

    division_per_workcenter = [x / sum(workload_per_wc) for x in workload_per_wc]
    machines_per_wc = [round(x * total_machines) for x in division_per_workcenter]
    # print(sum(machines_per_wc))
    machine_number_WC = []
    # print(workload_per_wc)

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

    # setupTime = pd.DataFrame(setupTime)
    # file_name = f"Setup_Time_New1.csv"
    # setupTime.to_csv(file_name)

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
        # mean_setup.append([])
        for j in i:
            for k in i:
                mean_setup[j[0]][j[1]] += setupTime[j[0]][k[0]] * demand[k[0]]

    mean_processing_time = sum([np.mean(processingTime[i]) * demand[i] for i in range(no_of_jobs)])

    # mean_setup_time = sum([np.mean(mean_setup[i]) * demand[i] for i in range(no_of_jobs)])

    mean_operations_per_job = sum(operations_per_job[i] * demand[i] for i in range(no_of_jobs))
    arrival_rate = [mean_processing_time * mean_operations_per_job / (total_machines * i) for i in utilization]
    sumProcessingTime = [sum(processingTime[i]) for i in range(no_of_jobs)]

    # Get maximum critical ratio
    max_ddt = 8
    CR = []
    DDT = []
    for j in range(no_of_jobs):
        CR.append([(sumProcessingTime[j] * max_ddt - sum(processingTime[j][0:i])) / sum(processingTime[j][i:]) for i in
                   range(operations_per_job[j] - 1)])
        DDT.append(sumProcessingTime[j] * max_ddt)



    return processingTime, operationOrder, machines_per_wc, list(setupTime), demand, list(priority_per_job), arrival_rate, machine_number_WC, max(CR), max(DDT)






