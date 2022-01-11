from matplotlib import pyplot as plt, rcParams
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

init_data_1 = np.zeros((18, 500))
new_file = []

filename = 'Results_List.txt'
my_file = open(filename)
content_list = my_file.readlines()
for j in range(0, len(content_list), 1):
    filename1 = str(content_list[j])
    new_file.append(str(filename1.rstrip()))

# print(new_file)

for j in range(3):
    my_file1 = open(new_file[j])
    content_list = my_file1.readlines()
    for i in range(len(content_list)):
        init_data_1[j][i] = float(content_list[i])

fig, ax = plt.subplots(1)
boxprops = dict(linestyle='-', linewidth=2.0, color='black')
whiskerprops = dict(linestyle='-', linewidth=2.0, color='black')
capprops = dict(linestyle='-', linewidth=2.0, color='black')
meanlineprops = dict(linestyle='--', linewidth=2.0, color='blue')
medianprops = dict(linestyle='-', linewidth=2.0, color='firebrick')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')

ind = [[0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2]]
# fig.delaxes(ax[0][1])
# fig.delaxes(ax[0][2])
# fig.delaxes(ax[1][2])
# for j in range(1):
#     init_data = [init_data_1[0:499][j * 3], init_data_1[0:499][j * 3 + 1], init_data_1[0:499][j * 3 + 2]]
#     ax[ind[0][j], ind[1][j]].boxplot(init_data, showmeans=True, notch=True, showfliers=False, meanline=False,
#                                      meanprops=meanpointprops,
#                                      medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
#                                      capprops=capprops)
#     for label in (ax[ind[0][j], ind[1][j]].get_xticklabels() + ax[ind[0][j], ind[1][j]].get_yticklabels()):
#         label.set_fontname('Arial')
#         label.set_fontsize(13)
#
#     ax[ind[0][j], ind[1][j]].grid(which='major', color='#CCCCCC', linestyle='--')
#     ax[ind[0][j], ind[1][j]].grid(which='minor', color='#CCCCCC', linestyle='--')
#     label_size = np.ceil(np.mean(init_data[0]) / 10)
#     print(label_size)
#     ax[ind[0][j], ind[1][j]].set_xticklabels(['FCFS', 'EDD', 'Custom'])
#     ax[ind[0][j], ind[1][j]].yaxis.set_minor_locator(AutoMinorLocator(int(label_size)))


init_data = [init_data_1[0:499][0], init_data_1[0:499][1], init_data_1[0:499][2]]
ax.boxplot(init_data, showmeans=True, notch=True, showfliers=False, meanline=False,
                                 meanprops=meanpointprops,
                                 medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                                 capprops=capprops)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(13)

ax.grid(which='major', color='#CCCCCC', linestyle='--')
ax.grid(which='minor', color='#CCCCCC', linestyle='--')
label_size = np.ceil(np.mean(init_data[0]) / 10)
print(label_size)
ax.set_xticklabels(['FCFS', 'EDD', 'Custom'])
ax.yaxis.set_minor_locator(AutoMinorLocator(int(label_size)))
ax.set_ylabel("Mean Weighted Tardiness", fontsize=16)

# utilization = [0.85, 0.9, 0.95]
# due_date_tightness = [4, 6, 8]
#
# for i, row in enumerate(ax):
#     for j, cell in enumerate(row):
#         if i == len(ax) - 1:
#             cell.set_xlabel("Due Date Tightness: {0:d}".format(due_date_tightness[j]), fontsize=16)
#         if j == 0:
#             cell.set_ylabel("Utilization: {0:.2f}".format(utilization[i]), fontsize=16)

plt.show()

# filename1 = 'Results/FCFS_90_6.txt'
# filename2 = 'Results/EDD_90_6.txt'
# filename3 = 'Results/Custom_90_6.txt'
#
# init_data_1 = np.zeros(500)
# my_file = open(filename1)
# content_list = my_file.readlines()
# for j in range(len(content_list)):
#     init_data_1[j] = float(content_list[j])
#
# init_data_2 = np.zeros(500)
# my_file = open(filename2)
# content_list = my_file.readlines()
# for j in range(len(content_list)):
#     init_data_2[j] = float(content_list[j])
#
# init_data_3 = np.zeros(500)
# my_file = open(filename3)
# content_list = my_file.readlines()
# for j in range(len(content_list)):
#     init_data_3[j] = float(content_list[j])
#
# fig, ax = plt.subplots(1)
#
# init_data = [init_data_1, init_data_2, init_data_3]
#
# print(np.mean(init_data_1), np.mean(init_data_2), np.mean(init_data_3))
#

#
# ax.boxplot(init_data, showmeans=True, notch=True, showfliers=False, meanline=False, meanprops=meanpointprops,
#            medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
#
# ax.grid(which='major', color='#CCCCCC', linestyle='--')
# ax.grid(which='minor', color='#CCCCCC', linestyle=':')
#
# ax.set_xticklabels(['FCFS', 'EDD', 'Custom'])
#
# for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#     label.set_fontname('Arial')
#     label.set_fontsize(13)
#
# # ax.xaxis.set_major_locator(MultipleLocator(1))
# ax.yaxis.set_major_locator(MultipleLocator(1))
#
# # ax.xaxis.set_minor_locator(AutoMinorLocator(1))
# ax.yaxis.set_minor_locator(AutoMinorLocator(0.5))
# # plt.boxplot(init_data, showmeans=True, grid=True)
# plt.show()
