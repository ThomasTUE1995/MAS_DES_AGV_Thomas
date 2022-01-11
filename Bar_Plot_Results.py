import pandas as pd
from matplotlib import pyplot as plt, rcParams
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

utilization = [85, 85, 90, 90, 95, 95]
due_date_settings = [4, 6, 4, 6, 4, 6]
fig, ax = plt.subplots(2, 3)
index = np.zeros((4, 6))
for i in range(6):
    filename = 'Results/Dispatching-' + str(utilization[i]) + '-' + str(due_date_settings[i]) + '.csv'
    filename1 = 'Results/Custom_2.csv'
    filename2 = 'Results/Custom_1.csv'

    data = pd.read_csv(filename, index_col=0)
    df = pd.DataFrame(data)

    data = pd.read_csv(filename1, index_col=0)
    df1 = pd.DataFrame(data)

    data = pd.read_csv(filename2, index_col=0)
    df2 = pd.DataFrame(data)

    D = {'Makespan': [], 'Mean Tardiness': [], 'Max Tardiness': [], 'FlowTime': []}

    makespan = [df["Makespan"].min(), df1.get('Makespan')[i], df2.get('Makespan')[i]]
    D['Makespan'] = [makespan[i] / max(makespan) for i in range(len(makespan))]
    index[0, i] = np.argmin(df["Makespan"])

    mean_tardiness = [df["Mean Weighted Tardiness"].min(), df1.get('Mean Weighted Tardiness')[i],
                      df2.get('Mean Weighted Tardiness')[i]]
    D['Mean Tardiness'] = [mean_tardiness[i] / max(mean_tardiness) for i in range(len(mean_tardiness))]
    index[1, i] = np.argmin(df["Mean Weighted Tardiness"])

    max_tardiness = [df["Max Weighted Tardiness"].min(), df1.get('Mean Weighted Tardiness')[i],
                     df2.get('Mean Weighted Tardiness')[i]]
    D['Max Tardiness'] = [max_tardiness[i] / max(max_tardiness) for i in range(len(max_tardiness))]
    index[2, i] = np.argmin(df["Max Weighted Tardiness"])

    flowtime = [df["Mean Flow Time"].min(), df1.get('Mean Flow Time')[i], df2.get('Mean Flow Time')[i]]
    D['FlowTime'] = [flowtime[i] / max(flowtime) for i in range(len(flowtime))]
    index[3, i] = np.argmin(df["Mean Flow Time"])

    # Normalize the Data
    Names = ["SPT", "W-ACTS", "Proposed"]
    width = 0.2
    x = np.arange(4)
    if i < 3:
        j = 1
    else:
        j = 0
    k = i % 3

    bar1 = ax[j, k].bar(x - 0.2, [D['Makespan'][0], D['Mean Tardiness'][0], D['Max Tardiness'][0], D['FlowTime'][0]],
                        width,
                        color=["Blue"])
    bar2 = ax[j, k].bar(x, [D['Makespan'][1], D['Mean Tardiness'][1], D['Max Tardiness'][1], D['FlowTime'][1]], width,
                        color=["Red"])
    bar3 = ax[j, k].bar(x + 0.2, [D['Makespan'][2], D['Mean Tardiness'][2], D['Max Tardiness'][2], D['FlowTime'][2]],
                        width,
                        color=["Orange"])

    ax[j, k].set_xlabel("Objective")
    ax[j, k].set_ylabel("Normalized Value")
    for f, rect in enumerate(bar1):
        height = rect.get_height()
        ax[j, k].text(rect.get_x() + rect.get_width() / 2.0, height, f'C{index[f, i]:.0f}', ha='center', va='bottom')
    # ax.legend(["Combination", "W-ATCS", "Proposed"])
plt.setp(ax, xticks=range(4), xticklabels=['Makespan', 'Mean Tardiness', 'Max Tardiness', 'Flowtime'])
plt.legend(["Combination", "W-ATCS", "Proposed"])

utilization = [0.85, 0.90, 0.95]
due_date_tightness = [6, 4]
for i, row in enumerate(ax):
    for j, cell in enumerate(row):
        print(j)
        if i == len(ax) - 1:
            cell.set_xlabel("Utilization: {0:.2f}".format(utilization[j]), fontsize=14)
        if j == 0:
            cell.set_ylabel("Due Date Tightness: {0:d}".format(due_date_tightness[i]), fontsize=14)

plt.show()
