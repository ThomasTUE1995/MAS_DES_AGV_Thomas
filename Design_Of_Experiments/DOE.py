import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns


# IF APA IS TRUE -> DIRECT JOB RELEASE

all_results = []
situations = []
nr_agvs = []

# Situation 1:
# ------------------
# JAFAMT: 0.0
# APA: TRUE
# AGVS: [2,2,3,2,2]
# AGV PER WC

result = [0.89, 129.1, 79.42, 50.43]
all_results.append(result)
situations.append("S1")
nr_agvs.append(sum([2, 2, 3, 2, 2]))

# Situation 2:
# ------------------
# JAFAMT: 0.0
# APA: FALSE
# AGVS: [2,2,3,2,2]
# AGV PER WC

result = [0.64, 80.99, 80.58, 50.94]
all_results.append(result)
situations.append("S2")
nr_agvs.append(sum([2, 2, 3, 2, 2]))

# Situation 3:
# ------------------
# JAFAMT: 0.0
# APA: TRUE
# AGVS: [2,1,2,1,1]
# AGV PER WC

result = [0.98, 122.26, 79.52, 76.38]
all_results.append(result)
situations.append("S3")
nr_agvs.append(sum([2, 1, 2, 1, 1]))

# Situation 4:
# ------------------
# JAFAMT: 0.0
# APA: FALSE
# AGVS: [2,1,2,1,1]
# AGV PER WC

result = [0.77, 87.67, 79.94, 76.87]
all_results.append(result)
situations.append("S4")
nr_agvs.append(sum([2, 1, 2, 1, 1]))

# Situation 5:
# ------------------
# JAFAMT: 0.0
# APA: FALSE
# AGVS: [1,1,1,1,1]
# AGV ALL WC

result = [0.76, 72.33, 79.68, 95.02]
all_results.append(result)
situations.append("S5")
nr_agvs.append(sum([1, 1, 1, 1, 1]))

# Situation 6:
# ------------------
# JAFAMT: 0.0
# APA: TRUE
# AGVS: [1,1,1,1,1]
# AGV ALL WC

result = [1.78, 125.21, 79.96, 93.21]
all_results.append(result)
situations.append("S6")
nr_agvs.append(sum([1, 1, 1, 1, 1]))

# Situation 7:
# ------------------
# JAFAMT: 2.0
# APA: TRUE
# AGVS: [1,1,1,1,1]
# AGV ALL WC

result = [3.78, 288.36, 79.27, 86.15]
all_results.append(result)
situations.append("S7")
nr_agvs.append(sum([1, 1, 1, 1, 1]))

# Situation 8:
# ------------------
# JAFAMT: 2.0
# APA: FALSE
# AGVS: [1,1,1,1,1]
# AGV ALL WC

result = [2.3, 197.6, 80.65, 87.54]
all_results.append(result)
situations.append("S8")
nr_agvs.append(sum([1, 1, 1, 1, 1]))


# Situation 9:
# ------------------
# JAFAMT: 1.0
# APA: FALSE
# AGVS: [1,1,1,1,1]
# AGV ALL WC

result = [1.46, 109.1, 80.72, 92.82]
all_results.append(result)
situations.append("S9")
nr_agvs.append(sum([1, 1, 1, 1, 1]))



mean_tardiness = []
max_tardiness = []
agv_utilizations = []
for result in all_results:
    mean_tardiness.append(result[0])
    agv_utilizations.append(result[3])



sns.set()  # Setting seaborn as default style even if use only matplotlib

fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
fig.suptitle('Design of Experiments')
axes[0].set_title('Mean Tardiness')
axes[1].set_title('Mean AGV Utilization')

# Bulbasaur
plots = sns.barplot(ax=axes[0], x=situations, y=mean_tardiness, label="hoi")
l1 = axes[0].set_title("Mean Tardiness")

AGV_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]

count = 0
# Iterrating over the bars one-by-one
for bar in plots.patches:

    plots.annotate(format(nr_agvs[count], '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=7, xytext=(0, 8),
                   textcoords='offset points')

    count += 1

# Utilization bars
plots2 = sns.barplot(ax=axes[1], x=situations, y=agv_utilizations)
l2 = axes[1].set_title("AGV Utilization")

# Iterrating over the bars one-by-one
for bar in plots2.patches:
    plots2.annotate(format(bar.get_height(), '.2f'),
                    (bar.get_x() + bar.get_width() / 2,
                     bar.get_height()), ha='center', va='center',
                    size=8, xytext=(0, 8),
                    textcoords='offset points')

import matplotlib.patches as mpatches

S1 = mpatches.Patch(label='Situation 1', color="#6376a3")
S2 = mpatches.Patch(label='Situation 2', color='#ba8763')
S3 = mpatches.Patch(label='Situation 3', color='#749c6d')
S4 = mpatches.Patch(label='Situation 4', color='#a15e60')
S5 = mpatches.Patch(label='Situation 5', color='#817baa')
S6 = mpatches.Patch(label='Situation 6', color='#867766')
S7 = mpatches.Patch(label='Situation 7', color='#c095be')
S8 = mpatches.Patch(label='Situation 8', color='#8b8b8b')
S9 = mpatches.Patch(label='Situation 9', color='#c2ba87')

legend_outside = plt.legend(handles=[S1, S2, S3, S4, S5, S6, S7, S8, S9], bbox_to_anchor=(-0.15, -0.30),
                            loc='lower center', ncol=3)

plt.savefig('DOE_Result.png',
            dpi=400,
            format='png',
            bbox_extra_artists=(legend_outside,),
            bbox_inches='tight')

# fig.legend(handles=[S1, S2, S3, S4, S5, S6, S7], bbox_to_anchor=(0, 0))

plt.tight_layout()
plt.show()

