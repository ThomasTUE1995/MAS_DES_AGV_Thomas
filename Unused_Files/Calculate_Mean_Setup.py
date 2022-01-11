import numpy as np

demand = [0.09753129, 0.07806598, 0.09295521, 0.10675914, 0.07615153, 0.15247753,
          0.13691097, 0.06290661, 0.13686145, 0.05938029]
setupTime = [[0.0, 0.9, 0.1, 1.1, 0.9, 1.1, 0.3, 0.9, 1.1, 1.1],
             [0.9, 0.0, 1.1, 0.3, 0.0, 0.8, 0.9, 0.1, 1.1, 0.8],
             [0.1, 1.1, 0.0, 0.3, 0.1, 0.4, 0.5, 0.9, 0.4, 0.1],
             [1.1, 0.3, 0.3, 0.0, 0.4, 0.0, 0.3, 0.6, 1.0, 0.1],
             [0.9, 0.0, 0.1, 0.4, 0.0, 0.2, 1.1, 0.8, 0.3, 0.8],
             [1.1, 0.8, 0.4, 0.0, 0.2, 0.0, 0.5, 0.2, 0.8, 0.3],
             [0.3, 0.9, 0.5, 0.3, 1.1, 0.5, 0.0, 0.4, 0.8, 0.2],
             [0.9, 0.1, 0.9, 0.6, 0.8, 0.2, 0.4, 0.0, 0.1, 0.1],
             [1.1, 1.1, 0.4, 1.0, 0.3, 0.8, 0.8, 0.1, 0.0, 0.6],
             [1.1, 0.8, 0.1, 0.1, 0.8, 0.3, 0.2, 0.1, 0.6, 0.0]]
operationOrder = [[5, 4], [1, 3, 4, 5, 2], [5, 1, 3], [1, 5, 4, 3, 2], [4, 3], [3, 5], [1, 2, 5], [5, 2, 4, 3],
                  [5, 4, 2], [2]]
types_per_wc = []
for wc in range(5):
    i = 0
    types = []
    for sublist in operationOrder:
        if (wc + 1) in sublist:
            types.append(i)
        i += 1
    types_per_wc.append(types)

mean_setup_wc = []
for i in range(5):
    mean_demand = sum(demand[i] for i in types_per_wc[i])
    mean_setup = 0
    for j in range(len(types_per_wc[i])):
        for n in range(j + 1, len(types_per_wc[i])):
            print(types_per_wc[i][j], types_per_wc[i][n], i)
            mean_setup += demand[types_per_wc[i][j]] * demand[types_per_wc[i][n]] / mean_demand * setupTime[types_per_wc[i][j]][types_per_wc[i][n]] * 2
    mean_setup_wc.append(mean_setup)

print(types_per_wc)
print(mean_setup_wc)
