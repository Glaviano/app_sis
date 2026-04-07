import numpy as np
import networkx as nx
from numba import njit
from numba.typed import List, Dict
from numba import types

def generate_network(n, net_type, **kwargs):
    if net_type == 'Erdős-Rényi':
        p = kwargs.get('p', 0.005)
        G = nx.erdos_renyi_graph(n, p)
    elif net_type == 'Barabási-Albert':
        m = kwargs.get('m', 25)
        G = nx.barabasi_albert_graph(n, m)
    elif net_type == 'Small World':
        k = kwargs.get('k', 50)
        p = kwargs.get('p', 0.1)
        G = nx.watts_strogatz_graph(n, k, p)
    else:
        G = nx.erdos_renyi_graph(n, 0.005)
        
    lista_adj_numba = List.empty_list(types.int32[:])
    for i in range(n):
        vicini = np.array(list(G.neighbors(i)), dtype=np.int32)
        lista_adj_numba.append(vicini)
    return lista_adj_numba, G

@njit
def infection_input(p, n):
    infection_dict = Dict.empty(key_type=types.int32, value_type=types.int32)
    for i in range(n):
        if np.random.rand() < p:
            infection_dict[i] = 1 
    return infection_dict

@njit
def recovery_function_sis(infection_dict, gamma):
    # Rimuoviamo il tracking complesso di R0 in favore di un calcolo più moderno in pandas in app.py
    new_infection_dict = Dict.empty(key_type=types.int32, value_type=types.int32)
    for node in infection_dict.keys():
        # L'RNG va prima, in modo compatibile con l'originale
        if np.random.rand() > gamma:
            new_infection_dict[node] = 1 
    return new_infection_dict

@njit
def seasonal_cosine(t, min_val, max_val):
    T = 365 
    A = (max_val - min_val) / 2
    C = (max_val + min_val) / 2
    phi = 180 
    return A * np.cos((2 * np.pi * t / T)+phi) + C

@njit
def infection_function_sis(adj_list, infection_dict, beta_min, beta_max, t, lockdown, lock_eff):
    new_infection_list = List.empty_list(types.int32)
    beta = seasonal_cosine(t, beta_min, beta_max)
    # Calcolo di beta_corretto che simula il taglio link
    if lockdown:
        beta_corrected = beta * (1 - lock_eff)
    else:
        beta_corrected = beta
        
    for node in infection_dict.keys():
        neighbors = adj_list[node]
        number_of_neighbors = len(neighbors)
        if number_of_neighbors == 0: 
            continue
            
        x = np.random.binomial(number_of_neighbors, beta_corrected) 
        for j in range(x):
            index = np.random.randint(0, number_of_neighbors-j)
            neighbor = neighbors[index]
            if neighbor not in infection_dict:
                new_infection_list.append(neighbor)
                
            swap_index = number_of_neighbors - 1 - j
            temp = neighbors[index]
            neighbors[index] = neighbors[swap_index]
            neighbors[swap_index] = temp

    for neighbor in new_infection_list:
        infection_dict[neighbor] = 1 
    return infection_dict, len(new_infection_list)

@njit
def death_function_sis(infection_dict, mu):
    new_infection_dict = Dict.empty(key_type=types.int32, value_type=types.int32)
    cont = 0
    for node in infection_dict.keys():
        if np.random.rand() > mu:
            new_infection_dict[node] = 1 
        else:
            cont += 1 
    return new_infection_dict, cont

@njit
def SIS(adj_list, infection_dict, beta_min, beta_max, gamma, death_rate, lock_eff, lockdown_threshold, lockdown_duration, max_steps):
    lockdown_step = 0                           
    lockdown = False                             
    n = len(adj_list)
    
    infected_over_time = np.zeros(max_steps, dtype=np.int32)
    daily_infected_arr = np.zeros(max_steps, dtype=np.int32)
    daily_deaths_arr = np.zeros(max_steps, dtype=np.int32)
    cumulative_deaths_arr = np.zeros(max_steps, dtype=np.int32)
    lockdown_status_arr = np.zeros(max_steps, dtype=np.int8)
    node_states = np.zeros((max_steps, n), dtype=np.int8)
    
    cumulative_deaths = 0
    
    for step in range(max_steps):
        if lockdown:
            lockdown_step += 1
            
        infection_dict = recovery_function_sis(infection_dict, gamma)
        infection_dict, daily = infection_function_sis(adj_list, infection_dict, beta_min, beta_max, step, lockdown, lock_eff)
        infection_dict, count_daily_deaths = death_function_sis(infection_dict, death_rate) 
        
        cumulative_deaths += count_daily_deaths
        n_infected = len(infection_dict)
        
        if n_infected/n > lockdown_threshold and not lockdown:
            lockdown_step = 0
            lockdown = True
            
        if lockdown and lockdown_step >= lockdown_duration:
            if n_infected/n < lockdown_threshold:
                lockdown = False
            else:
                lockdown_step = 0 # resetto lockdown
                
        infected_over_time[step] = n_infected
        daily_infected_arr[step] = daily
        daily_deaths_arr[step] = count_daily_deaths
        cumulative_deaths_arr[step] = cumulative_deaths
        lockdown_status_arr[step] = 1 if lockdown else 0
        
        for k in infection_dict.keys():
            node_states[step, k] = 1
        
    return infected_over_time, daily_infected_arr, daily_deaths_arr, cumulative_deaths_arr, lockdown_status_arr, node_states