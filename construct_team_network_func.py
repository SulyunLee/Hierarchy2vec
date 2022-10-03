
import pandas as pd
import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

def construct_seasonal_mentorship_network(df, year, team):
    '''
    This function constructs mentorship team network based on coaches in specific team
    in specific year. It constructs the hierarchical graph (tree), where mentor-relationships
    are connected with edges between node pairs.
    '''
    # select list of coaches with specific year and team
    coaches = df[(df.Year == year) & (df.Team == team)]
    coaches.reset_index(drop=True, inplace=True)

    # Construct hierarchical coach graph
    team_G = nx.Graph(year=year, team=team)

    # iterate through each coach and add nodes.
    for idx, row in coaches.iterrows():
        position = row.final_position
        num = row.final_hier_num
        team_G.add_node(row.Name, id=position, hier_num=num)

    # extract coaches with specific position ID
    hc = [x for x,y in team_G.nodes().data() if y['id'] == 'HC'][0]
    oc_list = [x for x,y in team_G.nodes().data() if y['id'] == 'OC']
    dc_list = [x for x,y in team_G.nodes().data() if y['id'] == 'DC']
    sc_list = [x for x,y in team_G.nodes().data() if y['id'] == 'SC']
    o_list = [x for x,y in team_G.nodes().data() if y['id'] == 'O']
    d_list = [x for x,y in team_G.nodes().data() if y['id'] == 'D']
    s_list = [x for x,y in team_G.nodes().data() if y['id'] == 'S']

    edgelist = [] # list to store all edges
    ## Connect position coaches and coordinators
    for s in s_list:
        if len(sc_list) == 0:
            edgelist.append(tuple([s, hc]))
            continue
        for sc in sc_list:
            edgelist.append(tuple([s, sc]))

    for o in o_list:
        if len(oc_list) == 0:
            edgelist.append(tuple([o, hc]))
            continue
        for oc in oc_list:
            edgelist.append(tuple([o, oc]))

    for d in d_list:
        if len(dc_list) == 0:
            edgelist.append(tuple([d, hc]))
            continue
        for dc in dc_list:
            edgelist.append(tuple([d, dc]))

    ## Connect coordinators and head coach
    for sc in sc_list:
        edgelist.append(tuple([sc, hc]))
    
    for oc in oc_list:
        edgelist.append(tuple([oc, hc]))
    
    for dc in dc_list:
        edgelist.append(tuple([dc, hc]))
    
    # add edges from the edgelist
    team_G.add_edges_from(edgelist)

    return team_G

def construct_cumulative_colleague_network(initial_g, df, min_year, max_year, bias, hierarchy_prob=None):
    '''
    This function constructs cumulative collaboration network by connecting pairwise 
    team members of all teams between <min_year> and <max_year> by aggregating with 
    <initial_g>
    Inputs:
      - initial_g: NetworkX Graph. The graph to be used as an initial graph for adding
                   new collaboration ties.
      - df: DataFrame. Data that shows the coaches' collaboration records containing
            Coaches' names, serving teams, and start/end years.
      - min_year: Integer. The earliest year in the range to be added to the cumulative 
                  collaboration network
      - max_year: Integer. The latest year in the range to be added to the cumulative collaboration
                  network.
      - bias: String. The bias type that tweaks the random walk scheme. The possible options are
              "unbiased", "hierarchy", "strength", and "recency"
      - hierarchy_prob: Dictionary. If <bias> is "hierarchy", then this argument must be a dictionary
                        that contains the relative probabilities for each hierarchical relationship;
                        Key: "downward", "peer", or "upward" / Value: relative probability (float or int)

    Outputs:
      - cumulative_g: NetworkX Graph. The cumulative collaboration network that connects
                      all pairwise team members between the input range of years on top of
                      the input <initial_g> graph.
    '''
    # extract teams
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    cumulative_g = initial_g.copy()

    # iterate through every year and every team to search for all collaboration pairs
    for year in tqdm(range(min_year, max_year+1)):
        for team in teams:
            try:
                # coaches who worked together in the same team
                records = df[(df.StartYear <= year) & (df.EndYear >= year) & (df.ServingTeam == team)]
            except:
                records = df[(df.Year == year) & (df.Team == team)]

            if (records.shape[0] > 1) and (records.Name.unique().shape[0] > 1):
                # Add coach names (nodes) to the graph
                coach_list = list(records.Name)
                new_coaches = [coach for coach in coach_list if coach not in cumulative_g.nodes()] # extract coaches not already in the graph
                cumulative_g.add_nodes_from(new_coaches)

                # Add collaboration ties (edges) to the graph
                # If the <bias> is unbiased, the edges are unweighted. Otherwise, the edges are weighted
                # based on the random walk probability
                if bias == "unbiased":
                    edges = itertools.combinations(coach_list,2) # extract all combinations of coach pairs as edges
                    new_edges = [edge for edge in edges if edge not in cumulative_g.edges()] # extract edges not already in the graph
                    cumulative_g.add_edges_from(new_edges)

                elif bias == "hierarchy":
                    # generate a dictionary that maps coach names to the cooresponding position 
                    # hierarchy numbers
                    mapping_dict = dict(zip(records.Name, records.final_hier_num))

                    # Add colleague edges to the graph
                    edges = itertools.permutations(coach_list,2) # extract all combinations of coach pairs as edges
                    # assign edge attributes: downward, peer, upward
                    new_edges = []
                    prob_list = []
                    for edge in edges:
                        coach1_num = mapping_dict[edge[0]]
                        coach2_num = mapping_dict[edge[1]]

                        if coach1_num < coach2_num: # from mentor to apprentice
                            prob = hierarchy_prob["downward"]
                        elif coach1_num == coach2_num: # between peers
                            prob = hierarchy_prob["peer"]
                        elif coach1_num > coach2_num: # from apprentice to mentor
                            prob = hierarchy_prob["upward"]
                        
                        new_edges.append(edge)
                        prob_list.append(prob)

                    cumulative_g.add_edges_from(new_edges)
                    # assign edge weights
                    nx.set_edge_attributes(cumulative_g, name="prob", values=dict(zip(new_edges, prob_list)))

                elif bias == "strength":
                    edges = itertools.combinations(coach_list,2)
                    # the edge has an attribute "prob" which has the number of previous collaborations 
                    # up to the current point"
                    for edge in edges:
                        if cumulative_g.has_edge(*edge):
                            # increment the strength if the edge already exists
                            # prob: number of previous collaborations
                            cumulative_g[edge[0]][edge[1]]['prob'] += 1
                        else:
                            # add the edge and set the strength as 1
                            cumulative_g.add_edge(*edge, prob=1)

                elif bias == "recency":
                    # the edge has an attribute "las_year" which has the year of the most recent
                    # collaboration
                    edges = itertools.combinations(coach_list,2)
                    for edge in edges:
                        cumulative_g.add_edge(*edge, last_year=year)

    return cumulative_g

def construct_cumulative_hierbiased_colleague_network(initial_g, df, min_year, max_year, selection_prob_dict):
    '''
    This function constructs weighted cumulative collaboration network where the
    edge weights are assigned based on the hierarchical relationships among team members.
    Inputs:
      - initial_g: NetworkX Graph. The graph to be used as an initial graph for adding
                   new collaboration ties.
      - df: DataFrame. Data that shows the coaches' collaboration records containing
            Coaches' names, serving teams, and start/end years.
      - min_year: The earliest year in the range to be added to the cumulative 
                  collaboration network
      - max_year: The latest year in the range to be added to the cumulative collaboration
                  network.

    Outputs:
      - cumulative_g: NetworkX Graph. The cumulative collaboration network that connects
                      all pairwise team members between the input range of years on top of
                      the input <initial_g> graph.
    '''
    # extract all teams
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    cumulative_g = initial_g.copy()

    # iterate through every year and every team to search for collaborations
    for year in tqdm(range(min_year, max_year+1)):
        for team in teams:
            try:
                # coaches who worked together
                records = df[(df.StartYear <= year) & (df.EndYear >= year) & (df.ServingTeam == team)]
            except:
                records = df[(df.Year == year) & (df.Team == team)]

            if (records.shape[0] > 1) and (records.Name.unique().shape[0] > 1):
                # Add coach names to the graph
                coach_list = list(records.Name)
                new_coaches = [coach for coach in coach_list if coach not in cumulative_g.nodes()] # extract coaches not already in the graph
                cumulative_g.add_nodes_from(new_coaches)

                # generate a dictionary that maps coach name to position hierarchy number
                mapping_dict = dict(zip(records.Name, records.final_hier_num))

                # Add colleague edges to the graph
                edges = itertools.permutations(coach_list,2) # extract all combinations of coach pairs as edges
                # assign edge attributes: downward, peer, upward
                new_edges = []
                prob_list = []
                for edge in edges:
                    coach1_num = mapping_dict[edge[0]]
                    coach2_num = mapping_dict[edge[1]]

                    if coach1_num < coach2_num:
                        direction = 'downward'
                    elif coach1_num == coach2_num:
                        direction = 'peer'
                    elif coach1_num > coach2_num:
                        direction = 'upward'
                    
                    prob = selection_prob_dict[direction]
                    prob_list.append(prob)
                    new_edges.append(edge)

                cumulative_g.add_edges_from(new_edges)
                nx.set_edge_attributes(cumulative_g, name="prob", values=dict(zip(new_edges, prob_list)))

    return cumulative_g

def construct_cumulative_strengthbiased_colleague_network(initial_g, df, min_year, max_year):
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    cumulative_g = initial_g.copy()

    # iterate through every year and every team to search for collaborations
    for year in tqdm(range(min_year, max_year+1)):
        for team in teams:
            try:
                # coaches who worked together
                records = df[(df.StartYear <= year) & (df.EndYear >= year) & (df.ServingTeam == team)]
            except:
                records = df[(df.Year == year) & (df.Team == team)]

            if (records.shape[0] > 1) and (records.Name.unique().shape[0] > 1):
                # Add coach names to the graph
                coach_list = list(records.Name)
                new_coaches = [coach for coach in coach_list if coach not in cumulative_g.nodes()] # extract coaches not already in the graph
                cumulative_g.add_nodes_from(new_coaches)

                # add colleague edges to the graph
                edges = itertools.combinations(coach_list,2)
                for edge in edges:
                    if cumulative_g.has_edge(*edge):
                        # increment the strength if the edge already exists
                        # prob: number of previous collaborations
                        cumulative_g[edge[0]][edge[1]]['prob'] += 1
                    else:
                        # add the edge and set the strength as 1
                        cumulative_g.add_edge(*edge, prob=1)
    return cumulative_g

def construct_cumulative_recencybiased_colleague_network(initial_g, df, min_year, max_year):
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    cumulative_g = initial_g.copy()

    # iterate through every year and every team to search for collaborations
    for year in tqdm(range(min_year, max_year+1)):
        for team in teams:
            try:
                # coaches who worked together
                records = df[(df.StartYear <= year) & (df.EndYear >= year) & (df.ServingTeam == team)]
            except:
                records = df[(df.Year == year) & (df.Team == team)]

            if (records.shape[0] > 1) and (records.Name.unique().shape[0] > 1):
                # Add coach names to the graph
                coach_list = list(records.Name)
                new_coaches = [coach for coach in coach_list if coach not in cumulative_g.nodes()] # extract coaches not already in the graph
                cumulative_g.add_nodes_from(new_coaches)

                # add colleague edges to the graph
                # last_year: year of the most recent collaboration
                edges = itertools.combinations(coach_list,2)
                for edge in edges:
                    cumulative_g.add_edge(*edge, last_year=year)

    return cumulative_g



def construct_seasonal_colleague_network(df, min_year, max_year):
    try:
        teams = df.ServingTeam.unique()
    except:
        teams = df.Team.unique()

    seasonal_G = nx.DiGraph()
    ids = df["ID"]
    records = df[["Name", "Year", "Team", "final_position"]].to_dict('records')
    id_record_dict = dict(zip(ids, records))
    for year in tqdm(range(min_year, max_year+1)):
        for team in teams:
            try: 
                records = df[(df.StartYear <= year) & (df.EndYear >= year) & (df.ServingTeam == team)]
            except:
                records = df[(df.Year == year) & (df.Team == team)]

            if (records.shape[0] > 1) and (records.Name.unique().shape[0] > 1):
                coach_list = list(records.ID)
                # all pairs of node directions
                edges = list(itertools.product(coach_list, coach_list))
                seasonal_G.add_edges_from(edges)

        nx.set_node_attributes(seasonal_G, id_record_dict)

    return seasonal_G
    
