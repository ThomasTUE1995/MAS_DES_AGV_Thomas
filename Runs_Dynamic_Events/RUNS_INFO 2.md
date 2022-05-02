
# RUN INFO


## AGV_ALL_WC: AGVs are not dedicated to WC's
simulation_parameter_1 = [2]

### SIT_1: Direct job release APA
simulation_parameter_3 = [0.0]
simulation_parameter_4 = [True]

#### ARR_1: 
min_jobs = [499, 999, 1499]  
max_jobs = [2499, 2999, 3499] 
wip_max = [150, 200, 300]  
arrival_time = [1.5429, 1.4572, 1.3804]
due_date_settings = [4, 4, 4]

normaliziation_MA_array =  [[-75, 225, -10, 40, 0, 150, -100, 200],
                            [-200, 150, -15, 12, -200, 150, 0, 0],  
                            [-300, 150, -35, 12, -300, 150, 0, 0]]  

normalization_AGV_array =  [[0, 40, -50, 225, -50, 255],
                            [1, 2, 3, 4, 5, 6],  
                            [1, 2, 3, 4, 5, 6]]  

##### Number of AGVS
agvsPerWC = [2, 1, 1, 2, 1]


#### ARR_2:

### SIT_2:
simulation_parameter_3 = [0.5]
simulation_parameter_4 = [True]

#### ARR_1:
min_jobs = [499, 999, 1499]  
max_jobs = [2499, 2999, 3499] 
wip_max = [150, 200, 300]  
arrival_time = [1.5429, 1.4572, 1.3804]
due_date_settings = [4, 4, 4]

normaliziation_MA_array =  [[-75, 225, -10, 40, 0, 150, -100, 200],
                            [-200, 150, -15, 12, -200, 150, 0, 0],  
                            [-300, 150, -35, 12, -300, 150, 0, 0]]  

normalization_AGV_array =  [[0, 40, -50, 225, -50, 255],
                            [1, 2, 3, 4, 5, 6],  
                            [1, 2, 3, 4, 5, 6]]  

#### ARR_2:

### SIT_3:
#### ARR_1:
#### ARR_2:

## AGV_PER_WC: AGVs are dedicated to WC's
simulation_parameter_1 = [1]

### SIT_1:
simulation_parameter_3 = [0.0]
simulation_parameter_4 = [True]

#### ARR_1:
min_jobs = [499, 999, 1499]  
max_jobs = [2499, 2999, 3499] 
wip_max = [150, 200, 300]  
arrival_time = [1.5429, 1.4572, 1.3804]
due_date_settings = [4, 4, 4]

normaliziation_MA_array =  [[-75, 225, -10, 40, 0, 150, -100, 200],
                            [-200, 150, -15, 12, -200, 150, 0, 0],  
                            [-300, 150, -35, 12, -300, 150, 0, 0]]  

normalization_AGV_array =  [[0, 40, -50, 225, -50, 255],
                            [1, 2, 3, 4, 5, 6],  
                            [1, 2, 3, 4, 5, 6]] 

##### Number of AGVS
agvsPerWC = [2, 1, 3, 1, 1]

##### Number of AGVS
agvsPerWC = [2, 2, 3, 2, 2]

##### Number of AGVS
agvsPerWC = [3, 3, 3, 3, 3]

#### ARR_2:

### SIT_2:
#### ARR_1:
#### ARR_2:

### SIT_3:
#### ARR_1:
#### ARR_2:


## NO_AGV: NOT FINISHED YET