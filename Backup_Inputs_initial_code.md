
 min_jobs = [499, 499, 999, 999, 1499, 1499]  # Minimum number of jobs in order te reach steady state
max_jobs = [2499, 2499, 2999, 2999, 3499, 3499]  # Maximum number of jobs to collect information from
wip_max = [150, 150, 200, 200, 300, 300]  # Maximum WIP allowed in the system

arrival_time = [1.5429, 1.5429, 1.4572, 1.4572, 1.3804, 1.3804]
utilization = [85, 85, 90, 90, 95, 95]

due_date_settings = [4, 6, 4, 6, 4, 6]

normaliziation = [[-75, 150, -8, 12, -75, 150],
                  [-30, 150, -3, 12, -30, 150],
                  [-200, 150, -15, 12, -200, 150],
                  [-75, 150, -6, 12, -75, 150],
                  [-300, 150, -35, 12, -300, 150],
                  [-150, 150, -15, 12, -150, 150]]  # Normalization ranges needed for the bidding

normalization_AGV = [[], [], [], [], [], []]