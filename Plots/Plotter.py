import string

import pandas
import pandas as pd
from matplotlib import pyplot as plt, rcParams
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.interpolate import make_interp_spline, BSpline

init_data = np.zeros((500, 3, 1))
plot_data_mean = np.zeros((500, 3))
plot_data_std = np.zeros((6, 500))

skip_bid = [7, [0, 1], [0, 3], [0, 4], [0, 5], [0, 7], [0, 7], [1, 3], [1, 4], [1, 5], [1, 7], [1, 7], [3, 4], [3, 5],
            [3, 7], [3, 7], [4, 5], [4, 7], [4, 7],
            [5, 7], [5, 7], [7, 7]]
skip_seq = [3, 3, 3, 3, 3, 0, 1, 3, 3, 3, 0, 1, 3, 3, 0, 1, 3, 0, 1, 0, 1, [8, 9]]

mean = np.zeros((22, 1000))
for skip in range(len(skip_bid)):
    str1 = "Runs/Attribute_Runs/Run-NoSetup2-90-4-5000" + "-" + str(skip_bid[skip]) + "-" + str(
                skip_seq[skip]) + ".txt"
    my_file1 = open(str1)
    content_list = my_file1.readlines()
    j = 0
    for i in range(len(content_list)):
        output = str.split(content_list[i], " ")
        if output[0]:
            if (float(output[1]) > 250) & (float(output[0]) > 20):
                mean[skip][j] = mean[skip][j - 1]
            else:
                mean[skip][j] = float(output[1])
            j += 1
print(mean)

fig, ax = plt.subplots(1)
x = range(1, 1001)
xnew = np.linspace(1, 1000, 1000)
for j in range(1):
    spl = make_interp_spline(x, mean[:][j], k=3)
    power_smooth = spl(xnew)
    ax.plot(xnew, power_smooth, linewidth=1)

ax.xaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.grid(which='major', color='#CCCCCC', linestyle='--')

plt.show()


# new_file = []
# filename = 'Run_Order.txt'
# my_file = open(filename)
# content_list = my_file.readlines()
# for j in range(0, len(content_list), 1):
#     filename1 = str(content_list[j])
#     new_file.append(str(filename1.rstrip()))
#
# for j in range(6):
#     my_file1 = open(new_file[j])
#     content_list = my_file1.readlines()
#     for i in range(len(content_list)):
#         output = str.split(content_list[i], " ")
#         plot_data_std[j][i] = float(output[2])
#
#
# fig, ax = plt.subplots(1)
# print(plot_data_std[:][1])
# x = range(1, 501)
# for j in range(6):
#     ax.plot(x, plot_data_std[j][:], linewidth=3)
#
# names = ['85-4', '90-4', '90-6', '95-4', '95-6', '95-8']
# ax.legend(names)
# plt.show()

# Load Data
# for i in range(82, 83):
#     # filename = 'Run' + str(i) + '.txt'
#     filename = 'Runs/Learning_Rate_Runs/Run-Custom5-85-4-1000.txt'
#     i = 0
#     my_file = open(filename)
#     content_list = my_file.readlines()
#     for j in range(len(content_list)):
#         output = str.split(content_list[j], " ")
#
#         init_data[j][0][i] = int(output[0])
#         init_data[j][1][i] = float(output[1])
#         init_data[j][2][i] = float(output[2])
#
# for j in range(500):
#     sample_mean = init_data[j][1][:]
#     plot_data_mean[j][0] = np.mean(sample_mean)
#     ci = 1.96 * np.std(sample_mean) / np.sqrt(50)
#     plot_data_mean[j][1] = plot_data_mean[j][0] + ci
#     plot_data_mean[j][2] = plot_data_mean[j][0] - ci
#
#     sample_std = init_data[j][2][:]
#     plot_data_std[j][0] = np.mean(sample_std)
#     ci = 1.96 * np.std(sample_std) / np.sqrt(50)
#     plot_data_std[j][1] = plot_data_std[j][0] + ci
#     plot_data_std[j][2] = plot_data_std[j][0] - ci
#
# a_m = [row[0] for row in plot_data_mean]
# # b_m = [row[1] for row in plot_data_mean]
# # c_m = [row[2] for row in plot_data_mean]
#
# a_s = [row[0] for row in plot_data_std]
# # b_s = [row[1] for row in plot_data_std]
# # c_s = [row[2] for row in plot_data_std]
#
# # print([row[0] for row in plot_data])
#
# # Create subplots
# x = range(1, 501)
# fig, ax = plt.subplots(2)
# axis_font = {'fontname': 'Arial', 'size': '14'}
#
# # Set first subplot
# ax[0].plot(x, a_m, linewidth=3)
#
# ax[0].xaxis.set_major_locator(MultipleLocator(50))
# ax[0].yaxis.set_major_locator(MultipleLocator(10))
#
# ax[0].grid(which='major', color='#CCCCCC', linestyle='--')
# ax[0].grid(which='minor', color='#CCCCCC', linestyle=':')
#
# ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
# ax[0].yaxis.set_minor_locator(AutoMinorLocator(4))
#
# # ax[0].fill_between(x, c_m, b_m, color='r', alpha=.5)
#
# # ax[0].set(xlabel='No. of generations', ylabel='Mean Tardiness')
# ax[0].set_ylabel('Mean Tardiness', fontsize=18)
# ax[0].set_xlabel('No. of generations', fontsize=18)
# ax[0].set_title('Learning Curve', fontsize=24)
# ax[0].grid(which='minor', alpha=0.2)
#
# # Set the tick labels font
# for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
#     label.set_fontname('Arial')
#     label.set_fontsize(13)
#
# # Set second subplot
# ax[1].plot(x, a_s, linewidth=3)
#
# ax[1].xaxis.set_major_locator(MultipleLocator(50))
# ax[1].yaxis.set_major_locator(MultipleLocator(0.01))
#
# ax[1].grid(which='major', color='#CCCCCC', linestyle='--')
# ax[1].grid(which='minor', color='#CCCCCC', linestyle=':')
#
# ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
# ax[1].yaxis.set_minor_locator(AutoMinorLocator(4))
#
# # ax[1].fill_between(x, c_s, b_s, color='r', alpha=.5)
#
# for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
#     label.set_fontname('Arial')
#     label.set_fontsize(13)
#
# ax[1].set_ylabel('Std. Tardiness', fontsize=18)
# ax[1].set_xlabel('No. of generations', fontsize=18)
# ax[1].grid(which='minor', alpha=0.2)
#
# # Show plot
# plt.show()
