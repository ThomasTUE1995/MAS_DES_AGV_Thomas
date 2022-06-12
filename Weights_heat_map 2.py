import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray
import seaborn as sns

scenario = "scenario_1"
JAFAMT_value = 0.0
immediate_release_bool = False

agvsPerWC = [1, 1, 1, 1, 1]  # Scenario 1
# agvsPerWC = [0, 0, 2, 0, 0]  # Scenario 2

# =======================================================================
sim_par_1_string = "AGV_ALL_WC"
sim_par_2_string = "JAFAMT_" + str(JAFAMT_value) + "_" + str(immediate_release_bool)
str1 = "Runs/" + "scenario_" + str(scenario) + "/" + sim_par_1_string + "/" + sim_par_2_string + "/" + str(
    agvsPerWC) + "/Final_Runs/Run-weights-" + str(
    90) + "-" + str(
    4) + ".csv"

df = pd.read_csv(str1, header=None)
weights = df.values.tolist()
# =======================================================================


# Y-axis - Machines attributes
Machine_Agents = []
AGV_Agents = []
for agent in range(len(weights)):
    if agent <= (len(weights) - sum(agvsPerWC) - 1):
        Machine_Agents.append("Machine Agent " + str(agent + 1))
    else:
        AGV_Agents.append("AGV Agent " + str(1 + agent - (len(weights) - (sum(agvsPerWC)))))

# X-axis - Machines attributes
Machine_Attributes = []
AGV_Attributes = []
for attribute in range(len(weights[0]) - 1):
    Machine_Attributes.append("Attribute " + str(attribute + 1))

    if attribute == 11:
        break

    AGV_Attributes.append("Attribute " + str(attribute + 1))

weights = np.array(weights)
weights_ma = np.delete(weights, range(len(weights) - sum(agvsPerWC), len(weights)), axis=0)
weights_ma = np.delete(weights_ma, 8, axis=1)
weights_agv = np.delete(weights, range(0, len(weights) - sum(agvsPerWC)), axis=0)
weights_agv = np.delete(weights_agv, [6, 12], axis=1)

fig, ax = plt.subplots(figsize=(15, 10))

fontsize = 15

# ===== MACHINES =====
plt.rcParams['figure.figsize'] = (15.0, 10.0)
plt.rcParams['font.family'] = "serif"
square = False  # Can be used to force equal squares
s = sns.heatmap(weights_ma.T, cmap='coolwarm', xticklabels=Machine_Agents, yticklabels=Machine_Attributes, robust=True,
                annot=True, fmt=".1f", annot_kws={'size': 10}, square=square)
plt.xticks(rotation=45)
# fig.tight_layout()
plt.xlabel('Machine Agents', fontsize=fontsize)
plt.ylabel('Machine Attributes', fontsize=fontsize)
plt.show()

# ===== AGVS =====
plt.rcParams['figure.figsize'] = (15.0, 10.0)
plt.rcParams['font.family'] = "serif"
square = False  # Can be used to force equal squares
s = sns.heatmap(weights_agv.T, cmap='coolwarm', xticklabels=AGV_Agents, yticklabels=AGV_Attributes, robust=True,
                annot=True, fmt=".1f", annot_kws={'size': 10}, square=square)
plt.xticks(rotation=45)
# fig.tight_layout()
plt.xlabel('AGV Agents', fontsize=fontsize)
plt.ylabel('AGV Attributes', fontsize=fontsize)
plt.show()

exit()

# ===== MACHINES =====
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(weights_ma.T)

# sns.heatmap(df, cmap='coolwarm')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(Machine_Agents)), labels=Machine_Agents)
ax.set_yticks(np.arange(len(Machine_Attributes)), labels=Machine_Attributes)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for j in range(len(Machine_Agents)):
    for i in range(len(Machine_Attributes)):
        text = ax.text(j, i, round(weights_ma[j, i], 2),
                       ha="center", va="center", color="w")

ax.set_title("Weights Heat Map Machines Attributes")
fig.tight_layout()
plt.show()

# ===== AGVS =====
fig, ax = plt.subplots(figsize=(15, 10))
im = ax.imshow(weights_agv.T)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(AGV_Agents)), labels=AGV_Agents)
ax.set_yticks(np.arange(len(AGV_Attributes)), labels=AGV_Attributes)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for j in range(len(AGV_Agents)):
    for i in range(len(AGV_Attributes)):
        text = ax.text(j, i, round(weights_agv[j, i], 2),
                       ha="center", va="center", color="w")

ax.set_title("Weights Heat Map AGV Attributes")
fig.tight_layout()
plt.show()
