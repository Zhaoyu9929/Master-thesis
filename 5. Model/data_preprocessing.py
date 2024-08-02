import pandas as pd
import numpy as np

def total_demand_level(scenario):
    path1 = r"C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\predict probability for each scenario.csv"
    total_demand_level = pd.read_csv(path1)

    demand_value = total_demand_level['demand'].iloc[scenario]
    probability = total_demand_level['probability'].iloc[scenario]
    return demand_value, probability

def trips_assigned(scenario):
    path2 = r"C:\Users\yanzh\Desktop\code_and_data\4. Deep learning part\trip_counts_with_probability.csv"
    trips_assigned = pd.read_csv(path2)
    demand_value, _ =total_demand_level(scenario)
    trips_assigned['trips'] = np.round(demand_value * trips_assigned['probability'])
    return trips_assigned

def mapping_id():
    path = r"C:\Users\yanzh\Desktop\code_and_data\coordinates of charging station.csv"
    df_charging_station_location = pd.read_csv(path)
    location_id_to_networkx_point = df_charging_station_location.set_index('Location_id')['networkx_point'].to_dict()
    return location_id_to_networkx_point