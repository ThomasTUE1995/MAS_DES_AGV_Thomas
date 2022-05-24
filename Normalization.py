

# Machine shop settings
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]  # Processing Times
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]  # Workcenter per operations
numberOfOperations = [4, 3, 5]  # Number of operations per job type
setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]  # Setuptypes from one job type to another
demand = [0.2, 0.5, 0.3]

# Machine information
machinesPerWC = [4, 2, 5, 3, 2]  # Number of machines per workcenter
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]  # Index of machines


normalization_MA_array = [[],
                          [],
                          []]

normalization_AGV_array = [[],
                           [],
                           []]



utilization = [85, 90, 95]

k = 9

for util in range(len(utilization)):

    RDue_list = []
    for type in range(len(processingTimes)):
        processing_sum = 0
        for o in processingTimes[type]:
            processing_sum += o
        RDue_list.append(k*processing_sum)

    RDue_max = max(RDue_list)
    RDue_min = -1 * (0.50 * RDue_max)

    CR_max_list = []
    for type in range(len(processingTimes)):
        for oper in range(0, len(processingTimes[type])-1):

            numerator_first = 0
            for o in processingTimes[type]:
                numerator_first += o
            numerator_first = numerator_first * k

            numerator_second = 0
            for o in processingTimes[type][0:oper]:
                numerator_second += o

            denumerator = 0
            for o in processingTimes[type][oper:]:
                denumerator += o

        CR_max_list.append((numerator_first - numerator_second)/denumerator)

    CR_max = max(CR_max_list)
    CR_min = -1 * CR_max


    normalization_MA_array[util].append(RDue_min)
    normalization_MA_array[util].append(RDue_max)
    normalization_MA_array[util].append(CR_min)
    normalization_MA_array[util].append(CR_max)
    normalization_MA_array[util].append(RDue_min)
    normalization_MA_array[util].append(RDue_max)
    normalization_MA_array[util].append(-10)
    normalization_MA_array[util].append(20)

    normalization_AGV_array[util].append(-10)
    normalization_AGV_array[util].append(25)
    normalization_AGV_array[util].append(RDue_min)
    normalization_AGV_array[util].append(RDue_max)
    normalization_AGV_array[util].append(RDue_min)
    normalization_AGV_array[util].append(RDue_max)




print(normalization_MA_array)
print("")
print(normalization_AGV_array)













