import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sns as sns

new_file = []
filename = "../Results/Tardiness_85_4.csv"
data = pd.read_csv(filename, header=None)
data = (data.values.tolist())
data_new = []
for i in range(len(data)):
    data_new.extend(data[i])

print(np.std(data_new))

fig, ax = plt.subplots(1)
boxprops = dict(linestyle='-', linewidth=2.0, color='black')
whiskerprops = dict(linestyle='-', linewidth=2.0, color='black')
capprops = dict(linestyle='-', linewidth=2.0, color='black')
meanlineprops = dict(linestyle='--', linewidth=2.0, color='blue')
medianprops = dict(linestyle='-', linewidth=2.0, color='firebrick')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')

ax.boxplot(data_new, showmeans=True, notch=True, showfliers=False, meanline=False,
                                 meanprops=meanpointprops,
                                 medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops,
                                 capprops=capprops)


plt.show()

