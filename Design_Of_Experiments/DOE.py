import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.DataFrame({'Situation': ['Situation 1', 'Situation 2', 'Situation 3', 'Situation 4',
                                 'Situation 1', 'Situation 2', 'Situation 3', 'Situation 4',
                                 'Situation 1', 'Situation 2', 'Situation 3', 'Situation 4',
                                 'Situation 1', 'Situation 2', 'Situation 3', 'Situation 4'],
                   'Mean Tardiness': [1.04, 5.05, 2.89, 0,
                                      1.62, 3.03, 4.29, 0,
                                      1.87, 6.85, 3.46, 0,
                                      0, 5.34, 0, 0],
                   'Design Options:': ['Design Option 1', 'Design Option 1', 'Design Option 1', 'Design Option 1',
                                       'Design Option 2', 'Design Option 2', 'Design Option 2', 'Design Option 2',
                                       'Design Option 3', 'Design Option 3', 'Design Option 3', 'Design Option 3',
                                       'Design Option 4', 'Design Option 4', 'Design Option 4', 'Design Option 4'],
                   "AGV utilization:": [93.83, 89.98, 82.09, 0,
                                        91.97, 89.46, 83.66, 0,
                                        89.75, 86.14, 82.88, 0,
                                        0, 87.95, 0, 0],
                   "Max Tardiness:": [168.0, 766.5, 291.51, 0,
                                      246.29, 543.02, 294.11, 0,
                                      240.77, 1169.79, 494.06, 0,
                                      0, 1025.19, 0, 0],
                   "Number AGVs:": ["[1,1,1,1,1]", "[0,0,1,0,0]", "[2,2,3,2,2]", "[1,1,1,1,1]",
                                    "[1,1,1,1,1]", "[0,0,1,0,0]", "[2,2,3,2,2]", "[1,1,1,1,1]",
                                    "[1,1,1,1,1]", "[0,0,1,0,0]", "[2,2,3,2,2]", "[1,1,1,1,1]",
                                    "[1,1,1,1,1]", "[0,0,1,0,0]", "[2,2,3,2,2]", "[1,1,1,1,1]"],
                   "Mean Wip:": [41.89, 39.05, 77.71, 0,
                                 41.38, 39.49, 84.48, 0,
                                 46.15, 40.14, 80.57, 0,
                                 0, 37.27, 0, 0]})


fig, ax = plt.subplots(figsize=(12, 8))

# set seaborn plotting aesthetics
sns.set(style='white')

# create grouped bar chart
sns.barplot(x='Situation', y='Mean Tardiness', hue='Design Options:', data=df)

# add overall title
plt.title('Design Of Experiments', fontsize=16)

# add axis titles
plt.xlabel('Situations')
plt.ylabel('Mean Tardiness')

# rotate x-axis labels
plt.xticks(rotation=45)
plt.show()


# Design Options
# Design Option 1: False - 0.0
# Design Option 2: True - 0.0
# Design Option 3: False - 2.0
# Design Option 4: True - 2.0




