import numpy as np
import pandas as pd
from constants import S, I, H, T, station_cost, car_cost, income_per_car, capacity
from TL_graph import time_expanded_location_graphs, process_scenario_data
from data_preprocessing import total_demand_level, mapping_id
from utils import travel_time_func, load_graph_from_pkl, load_and_prepare_graph
import gurobipy as gp
from gurobipy import Model, GRB, quicksum
import osmnx as ox
import networkx as nx

def main():
    # Time-expanded location graphs
    waiting_arcs, travel_arcs, final_collection_arcs, all_arcs = time_expanded_location_graphs()
    dfs = process_scenario_data()

    # Initialize the model
    m = Model('CSLP')

    # Set Gurobi parameters to manage memory usage
    m.setParam('NodefileStart', 0.5)  # Start writing node files to disk after 0.5GB of memory is used
    m.setParam('MIPFocus', 1)  # Focus on finding feasible solutions quickly
    m.setParam('Method', 2)  # Use barrier method for continuous optimization
    m.setParam('Presolve', 2)  # Aggressive presolve
    
    # First stage decision variables
    y_i = m.addVars(range(I), vtype=GRB.BINARY, name='build_variable')
    L_i = m.addVars(range(I), vtype=GRB.INTEGER, name='purchased_car', lb=0, ub=10)


    # Second stage decision variables
    x_k = m.addVars([(s, k) for s in range(S) for k in range(len(dfs[s]))], vtype=GRB.BINARY, name='accepted_trip')
    f_ha = m.addVars([(s, h, a) for s in range(S) for h in range(H) for a in range(len(all_arcs))], vtype=GRB.BINARY, name='flow_realized_by_car')
    x_hk = m.addVars([(s, h, k) for s in range(S) for h in range(H) for k in range(len(dfs[s]))], vtype=GRB.BINARY, name='trip_realized_by_car')

    # Objective function: maximize profit
    m.setObjective(
        quicksum(
            total_demand_level(s)[1] * quicksum(income_per_car * x_k[s, k] for k in range(len(dfs[s])))
            for s in range(S)
        ) -
        quicksum(station_cost * y_i[i] for i in range(I)) -
        car_cost * quicksum(L_i[i] for i in range(I)),
        GRB.MAXIMIZE
    )

    # Constraint: trips cannot be accepted if no cars are purchased and no stations are built
    for s in range(S):
        for k in range(len(dfs[s])):
            m.addConstr(
                quicksum(x_hk[s, h, k] for h in range(H)) <= quicksum(L_i[i] for i in range(I)),
                name=f"trip_cannot_be_accepted_without_cars_s{s}_k{k}")

    # Budget constraints
    budget = 10000000
    m.addConstr(station_cost * quicksum(y_i[i] for i in range(I)) + 
                car_cost * quicksum(L_i[i] for i in range(I)) <= budget, 
                name='budget_constraint')


    # Total number of cars should be fixed as 50 cars
    m.addConstr(quicksum(L_i[i] for i in range(I)) <= 50, name='total_cars_constraint')
    
    # Constraint: one car per accepted trip
    for s in range(S):
        for k in range(len(dfs[s])):
            m.addConstr(
                quicksum(x_hk[s, h, k] for h in range(H)) == x_k[s, k], 
                name=f"one_car_per_accepted_trip_s{s}_k{k}"
            )

    # Capacity constraints
    for s in range(S):
        for i in range(I):
            # Directly handling the final collection arcs to 'sink'
            final_arc_key = (s, (i, T), 'sink')
            if final_arc_key in f_ha:
                m.addConstr(
                    quicksum(f_ha[s, h, final_arc_key] for h in range(H)) <= capacity * y_i[i], 
                    name=f'final_capacity_s{s}_i{i}'
                )
            for t in range(T):
                # Filter and append arcs relevant to the current (s, i, t)
                outgoing_waiting_arcs = [(s_arc, src, dst) for s_arc, src, dst in waiting_arcs if s_arc == s and src == (i, t)]
                m.addConstr(
                    quicksum(f_ha[s, h, arc] for h in range(H) for arc in outgoing_waiting_arcs if arc in f_ha) <= capacity * y_i[i], 
                    name=f'waiting_capacity_s{s}_i{i}_t{t}'
                )
    for s in range(S):
        for i in range(I):
            for t in range(T):
            
                # Given s, i, t, a specific waiting arc can be identified
                incoming_waiting_arcs = [(s_arc, src, dst) for s_arc, src, dst in waiting_arcs if s_arc == s and dst == (i, t)]

                # Add the constraint
                m.addConstr(quicksum(f_ha[s, h, arc] for h in range(H) for arc in incoming_waiting_arcs if arc in f_ha) <= y_i[i])

                # Next is travel arc
                incoming_travel_arcs = [(s_arc, k, src, dst) for s_arc, k, src, dst in travel_arcs if s_arc == s and dst == (i, t)]

                # Add the constraint
                m.addConstr(quicksum(f_ha[s, h, arc] for h in range(H) for arc in incoming_travel_arcs if arc in f_ha) <= y_i[i])    
    
    # Solve the model
    m.optimize()

    # Directly check the optimization status
    if m.Status == GRB.OPTIMAL:
        print("Optimization was successful. Printing results.")

        # Calculate and print objective function components
        total_income = sum(
            total_demand_level(s)[1] * sum(income_per_car * x_k[s, k].X for k in range(len(dfs[s])))
            for s in range(S)
        )
        total_station_cost = sum(station_cost * y_i[i].X for i in range(I))
        total_car_cost = car_cost * sum(L_i[i].X for i in range(I))
        
        print(f"Total income: {total_income}")
        print(f"Total station cost: {total_station_cost}")
        print(f"Total car cost: {total_car_cost}")
        
        objective_value = total_income - total_station_cost - total_car_cost
        print(f"Objective value (calculated): {objective_value}")
        print(f"Objective value (from solver): {m.ObjVal}")
        
        for i in range(I):
            print(y_i[i])
            print(L_i[i])
            if y_i[i].X > 0.5:
                print(f"Build a charging station at location {i} with {L_i[i].X} cars.")
    else:
        print(f"Optimization issue with status code: {m.Status}")

if __name__ == "__main__":
    main()
