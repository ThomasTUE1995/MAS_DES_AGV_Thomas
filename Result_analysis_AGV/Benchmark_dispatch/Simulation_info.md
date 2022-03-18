min_jobs = [499, 999, 1499]  # Minimum number of jobs in order te reach steady state
max_jobs = [2499, 2999, 3499]  # Maximum number of jobs to collect information from
wip_max = [150, 200, 300]  # Maximum WIP allowed in the system
arrival_time = [1.9104, 1.8043, 1.7093]  # (5.2455 + 1.25) x 4
utilization = [85, 90, 95]
due_date_settings = [4, 4, 4]
normaliziation = [[-75, 150, -8, 12, -75, 150],
                  [-200, 150, -15, 12, -200, 150],
                  [-300, 150, -35, 12, -300, 150]]  # Normalization ranges needed for the bidding

normalization_AGV = [[], [], []]
no_runs = 50
no_processes = 5  # Change dependent on number of threads computer has, be sure to leave 1 thread remaining

# Simulation Parameter 1 - AGV scheduling control:
# 1: Linear Bidding Auction
# 3: Nearest Idle AGV Rule (AGV & JOB)
# 4: Random AGV Rule (AGV) - Random Job Rule (Job)
# 5: Longest Time In System Rule (JOB) - Minimal Distance Rule (AGV)
# 6: Longest Waiting Time at Pickup Point (JOB) - Minimal Transfer Rule (AGV)
# 7: Longest Average Waiting Time At Pickup Point (JOB) - Minimal Transfer Rule (AGV)
# 8: Earliest Due Time (JOB) - Minimal Transfer Rule (AGV)
# 9: Earliest Release Time (JOB) - Minimal Transfer Rule (AGV)
simulation_parameter_1 = [3, 4, 5, 6, 7, 8, 9]

# Simulation Parameter 2 - Number of AGVs per work center
# 0: 6 AGV per WC - ZERO TRAVEL TIME!
# 1: 6 AGV per WC
# 2: 3 AGV per WC
# 3: No_AGVS = Machines + 1
# 4: No_AGVS = Machines + 2
# 5: No_AGVS = Machines - 1 - Gives many errors due arrival rate
# 6: No_AGVS = Machines
simulation_parameter_2 = [2, 3, 6]

# Simulation Parameter 3 - Job almost finished at machines trigger values
simulation_parameter_3 = [0.0]

# Simulation Parameter 4 - Direct or periodically job release APA (Direct = True)
simulation_parameter_4 = [False]