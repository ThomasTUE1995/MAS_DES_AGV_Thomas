
import itertools


Number_of_WC = [[5,2],[10,2]]
Job_Types = [5,20]
Total_Machines = [16,32]
Processing_time = [[2,9],[10,50]]


situations = {}
count = 1

for (a, b, c, d) in itertools.product(Number_of_WC, Job_Types, Total_Machines, Processing_time):

    situations["scenario_"+str(count)] = [a, b, c, d, [0], 250]
    count += 1


print(situations)






