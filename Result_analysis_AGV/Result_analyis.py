
import itertools
import pandas as pd
import warnings
import numpy as np
import math
import xlsxwriter

warnings.filterwarnings('ignore')

# Simulation parameters
simulation_parameter_1 = [3, 4, 5, 6, 7, 8, 9]
simulation_parameter_2 = [2, 6]
simulation_parameter_3 = [0.0]
simulation_parameter_4 = [False]

result_list = {}

for (a, b, c, d) in itertools.product(simulation_parameter_1, simulation_parameter_2, simulation_parameter_3,
                                          simulation_parameter_4):

    simulation_name = "(" + str(a) + "-" + str(b) + "-" + str(c) + "-" + str(d) + ")"
    read_str = "Runs/Run-" + simulation_name + ".csv"
    df_read = pd.read_csv(read_str, header=None)
    read_file = df_read.values.tolist()

    skip_header = False
    mean_tardiness_list = []

    for idx, values in enumerate(read_file):
        if not skip_header:
            skip_header = True
        else:

            mean_tardiness = float(values[3])
            mean_tardiness_list.append(mean_tardiness)

    result_list[simulation_name] = mean_tardiness_list


workbook = xlsxwriter.Workbook('Benchmark_dispatch/Results_Try.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0

#currency_format = workbook.add_format({'num_format': '0.0000%'})
order = sorted(result_list.keys())
for key in order:
    row += 1
    worksheet.write(row, col, key)
    i = 1
    for item in result_list[key]:

        worksheet.write(row, col + i, item)
        i += 1

workbook.close()








