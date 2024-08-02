import pandas as pd
import numpy as np
from constants import S, I, T

# Read the csv file of coordinatate of charging station
def location_converting():
    path = r"C:\Users\yanzh\Desktop\code_and_data\coordinates of charging station.csv"
    df_charging_station_location = pd.read_csv(path)
    location_id_to_index = df_charging_station_location.set_index('Location_id')['index'].to_dict()
    return location_id_to_index

def process_scenario_data():
    """Processes scenario files from hardcoded paths to adjust time and extract necessary data."""
    file_paths = [rf'C:\Users\yanzh\Desktop\code_and_data\5. Model\scenairo {i}.csv' for i in range(5)]

    dfs = {}
    for i, file in enumerate(file_paths):
        df = pd.read_csv(file)
        df['starting_time'] = pd.to_datetime(df['starting_time'])
        df['ending_time'] = pd.to_datetime(df['ending_time'])
        df = df[(df['ending_time'] - df['starting_time']) <= pd.Timedelta(hours=5)]
        df['starting_time_period'] = np.floor(df['starting_time'].dt.minute).astype(int)
        # Here check if the hour part is greater than or 1
        df['ending_time_period'] = np.where((df['ending_time'].dt.hour >= 22) | ((df['ending_time'].dt.hour == 21) & (df['ending_time'].dt.minute >= 60)),
                                            60,
                                            np.ceil(df['ending_time'].dt.minute).astype(int) + 1)

        dfs[i] = df
    return dfs

def generate_travel_arcs(dfs, location_id_to_index):
    """Generates travel arcs based on processed data and station index mapping."""
    travel_arcs = []
    for s, data in dfs.items():
        for k in range(len(data)):
            row = data.iloc[k]
            
            # Change the starting point id to index (from 0 to 65)
            starting_point_id = row['starting_point']
            starting_point = location_id_to_index.get(starting_point_id, None)  
            
            ending_point_id = row['ending_point']
            ending_point = location_id_to_index.get(ending_point_id, None) 
            
            if starting_point is not None and ending_point is not None:
                arcs = (s, k, (starting_point, row['starting_time_period']), (ending_point, row['ending_time_period']))
                travel_arcs.append(arcs)
    return travel_arcs

def time_expanded_location_graphs():
    # Define the nodes in the graph
    # root = 'root'
    sink = 'sink'
    # V = [(s, i, t) for s in range(S) for i in range(I) for t in range(T)] + [sink]

    # Waiting arcs
    waiting_arcs = [(s, (i, t), (i, t+1)) for s in range(S) for i in range(I) for t in range(T)]
    
    # Travel arcs
    dfs = process_scenario_data()
    location_id_to_index = location_converting()
    travel_arcs = generate_travel_arcs(dfs, location_id_to_index)
    
    # Final collection arcs
    final_collection_arcs = [(s, (i, T), sink) for s in range(S) for i in range(I)]

    # All arcs
    all_arcs = waiting_arcs + travel_arcs + final_collection_arcs 

    return waiting_arcs, travel_arcs, final_collection_arcs, all_arcs