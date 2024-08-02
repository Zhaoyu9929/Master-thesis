import networkx as nx
import osmnx as ox
import pandas as pd
import pickle

def load_and_prepare_graph():
    # Import Manhattan network and change node labels to integers
    G = ox.graph_from_place('Manhattan, New York, USA', network_type='drive')

    # Remove nodes that cannot access at least 10% of other nodes
    num_nodes = len(G.nodes)
    remove_list = [node for node in G.nodes if len(nx.descendants(G, node)) < num_nodes / 10]
    G.remove_nodes_from(remove_list)

    # The node labels of the graph are converted to integers for easier handling and reference, 
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_node_ID')
    G = ox.add_edge_speeds(G)

    # Load average speeds and apply them to the edges
    speed_df_path = r"C:\Users\yanzh\Desktop\code_and_data\archive\nyc_avg_speeds_2019-06.csv"
    speed_df = pd.read_csv(speed_df_path)
    speed_df = speed_df[['osm_way_id', 'hour', 'speed']]
    G = ox.add_edge_speeds(G, speeds=speed_df)
    G = ox.add_edge_travel_times(G, precision=1)
    return G

# load the graph
def load_graph_from_pkl():
    file_path = "graphs_list.pkl"
    with open(file_path, "rb") as f:
        graph = pickle.load(f)
    return graph


# Update the freeflow speed information of the edge which lacks speed information
def update_graph_edges(graph, hour):
    travel_time_key = f'travel_time_hour_{hour}'
    for u, v, key, data in graph.edges(keys=True, data=True):
        if travel_time_key not in data:
            # Safely access the travel time in a multigraph structure
            freeflow_travel_time = graph[u][v][key].get('travel_time', None)
            if freeflow_travel_time is not None:
                graph[u][v][key][travel_time_key] = freeflow_travel_time

# Verify if each graph is an instance of MultiGraph
def process_graphs(graphs):
    for hour, graph in enumerate(graphs):
        if isinstance(graph, nx.MultiGraph):
            update_graph_edges(graph, hour)
        else:
            print(f"Graph for hour {hour} is not a MultiGraph.")

# Define the travel time function using Dijkstra algorithm
def travel_time_func(G_hour, point1, point2, hour):
    # Define the weight key for the specific hour
    weight_key = f'travel_time_hour_{hour}'

    # Use Dijkstra's algorithm to find the shortest path length and path
    # This function returns both the length of the path and the actual path as a list of nodes
    travel_time, path = nx.single_source_dijkstra(G_hour, source=point1, target=point2, weight=weight_key)

    # Round the travel time to 2 decimal places
    travel_time = round(travel_time, 4)

    return travel_time, path



# print the trips that do not meet the battery limitation
# # 最大运行时间为60分钟
# T_max = 60

# # 映射位置ID到网络点
# location_id_to_networkx_point = mapping_id()
# # 从pkl文件加载图
# graphs = load_graph_from_pkl()

# # 创建一个字典来存储行驶时间
# travel_time_dict = {}

# # 迭代所有场景
# for s in range(S):
#     # 迭代每个场景中的所有旅行
#     for k in range(len(dfs[s])):
        
#         # 获取当前旅行的起点ID
#         origin_id = dfs[s]['starting_point'].iloc[k]
#         if origin_id in location_id_to_networkx_point:
#             origin = location_id_to_networkx_point[origin_id]
#         else:
#             print(f"Origin ID {origin_id} not found in mapping for Scenario {s}, Trip {k}")
#             continue
        
#         # 获取当前旅行的终点ID
#         destination_id = dfs[s]['ending_point'].iloc[k]
#         if destination_id in location_id_to_networkx_point:
#             destination = location_id_to_networkx_point[destination_id]
#         else:
#             print(f"Destination ID {destination_id} not found in mapping for Scenario {s}, Trip {k}")
#             continue

#         # 获取当前旅行的开始时间的小时
#         hour = dfs[s]['starting_time'].dt.hour.iloc[k]
#         G_hour = graphs[hour]

#         # 计算此行程的行驶时间
#         travel_time, _ = travel_time_func(G_hour, origin, destination, hour)

#         # 检查NaN或Inf值
#         if np.isnan(travel_time) or np.isinf(travel_time):
#             print(f"Invalid travel time detected: Scenario {s}, Trip {k}, Origin {origin}, Destination {destination}, Hour {hour}, Travel Time {travel_time}")
#             continue

#         # 将行驶时间存储在字典中
#         travel_time_dict[(s, k)] = travel_time

# # 打印所有行驶时间超过60分钟的旅行
# for (s, k), travel_time in travel_time_dict.items():
#     if travel_time > T_max:
#         print(f"Trip in Scenario {s}, Trip {k} exceeds 60 minutes with travel time {travel_time} minutes")
