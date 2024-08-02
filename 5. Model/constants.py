# number of scenarios, here is 5
S = 5

# number of poential charging staion locations
I = 25

# number of cars 
H = 30

# A minute is one time interval, so there are 60 time intervals within an hour and 61 time points.
T = 60

# fixed cost of each charging station, here we assmue 150000 dollars, 150000 / (5*365*24)
station_cost = 3.42

# Purchasing cost of each car, 15000 / (5*365*24)
car_cost = 0.34

# Income of each accepted trip
income_per_car = 20 # assume here is 20 to 25 dollars

# Capacity of each charging station, i.e. number of charging slots can be built at each charging station
capacity = 10

# Burdget: (I-5)*3.42 + (H-5)*0.34 = 76.9 
W = 77 

# Battery limitation
battery_limitation = 55