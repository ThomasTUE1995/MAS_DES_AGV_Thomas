
import itertools
import pandas as pd
import warnings
import numpy as np
import math
import xlsxwriter

warnings.filterwarnings('ignore')

# Simulation parameters
simulation_parameter_1 = [3, 4, 5, 6, 7, 8, 9]
simulation_parameter_2 = [3]
simulation_parameter_3 = [0.0]
simulation_parameter_4 = [False]

Mean_Weighted_Tardiness_base = []

base_str = "Run-" + "(base)" + ".csv"
df_base = pd.read_csv(base_str, header=None)
base_file = df_base.values.tolist()

skip_header = False
for base_values in base_file:
    if not skip_header:
        skip_header = True
    else:
        Mean_Weighted_Tardiness_base.append(base_values[3])

result_list = {}

for (a, b, c, d) in itertools.product(simulation_parameter_1, simulation_parameter_2, simulation_parameter_3,
                                          simulation_parameter_4):

    simulation_name = "(" + str(a) + "-" + str(b) + "-" + str(c) + "-" + str(d) + ")"
    read_str = "Runs/Run-" + simulation_name + ".csv"
    df_read = pd.read_csv(read_str, header=None)
    read_file = df_read.values.tolist()

    skip_header = False
    percentage_list = []

    for idx, values in enumerate(read_file):
        if not skip_header:
            skip_header = True
        else:
            base_value = float(Mean_Weighted_Tardiness_base[idx - 1])
            compare_value = float(values[3])
            percentage = abs(round(((base_value - compare_value) / abs(base_value)), 6))
            percentage_list.append(percentage)

    result_list[simulation_name] = percentage_list


# workbook = xlsxwriter.Workbook('Results_Dispatch.xlsx')
# workbook = xlsxwriter.Workbook('Results_No_AGVs.xlsx')
# workbook = xlsxwriter.Workbook('Results_No_AGVs_6.xlsx')
workbook = xlsxwriter.Workbook('Results_No_AGVs_3.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0

currency_format = workbook.add_format({'num_format': '0.0000%'})

order = sorted(result_list.keys())
for key in order:
    row += 1
    worksheet.write(row, col, key)
    i = 1
    for item in result_list[key]:

        worksheet.write(row, col + i, item, currency_format)
        i += 1

workbook.close()








