
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

def map_rel(row, paygrade_dict, paygrade_map, hierarchy_prob):
    source = row.emp_a
    target = row.emp_b
    rel = row.rel
    if rel == "N+0":
        return hierarchy_prob["peer"]
    elif rel == "Other":
        if (source in paygrade_dict) and (target in paygrade_dict):
            source_paygrade = paygrade_dict[source]
            target_paygrade = paygrade_dict[target]
    
            source_ranking = paygrade_map[source_paygrade]
            target_ranking = paygrade_map[target_paygrade]

            if source_ranking < target_ranking:
                return hierarchy_prob["downward"]
            elif source_ranking == target_ranking:
                return hierarchy_prob["peer"]
            else: 
                return hierarchy_prob["upward"]
        else: 
            return -1
    elif "+" in rel:
        return hierarchy_prob["downward"]
    elif "-" in rel:
        return hierarchy_prob["upward"]

def assign_hierarchical_rel(df, paygrade_dict, paygrade_map, hierarchy_prob):
    df.loc[:, 'prob'] = df.apply(map_rel, args=(paygrade_dict, paygrade_map, hierarchy_prob), axis=1)
    df = df.loc[df['prob'] != -1,["emp_a", "emp_b", "prob"]].reset_index(drop=True)
    df.columns = ["source","target","prob"]

    oppos_hierarchy_prob = {1:5, 3:3, 5:1}
    df2 = df[["target", "source"]]
    df2.loc[:, 'prob'] = df.prob.map(oppos_hierarchy_prob)
    df2.columns = ["source","target","prob"]

    final_df = pd.concat([df, df2], axis=0).reset_index(drop=True)

    return final_df


def construct_cumulative_collab_network(initial_g, df, week_num, bias, paygrade_dict=None, paygrade_map=None, hierarchy_prob=None):
    '''
    This function constructs cumulative collaboration network by connecting messaging ties
    between pairs of employees by sequentially aggregating with <initial_g> 
    Inputs:
        - initial_g: Networkx graph. The graph to be used as an initial graph for addition new 
                    collaboration ties.
        - df: DataFrame. Data that shows the employees' collaboration records containing IDs of employee
                pairs, date of collaborations, and their hierarchical relationship(only for bias=hierarchy).
        - week_num: Int. The number of week sequence starting from zero. This is only used for recency-based
                    network, which is assigned as edge attributes to memorize the most recent collaborations.
        - bias: Str. The bias type that tweaks the random walk scheme. The possible options are 
                "unbiased", "hierarchy", "strength", and "recency".
        - paygrade_df: DataFrame. Data that stores the paygrade information of individual employees.
                        The DataFrame must include employee IDs in "employee_id" column, date in "dt" column,
                        string type paygrade in "paygrade" column.
        - paygrade_map: Dictionary. Dictionary that maps string type paygrade with the integer rankings.
                        Smaller number means higher rank of the hierarchical position.
        - hierarchy_prob: Dictionary. If <bias> is "hierarchy", then this argument must be a dictionary
                            that contains the relative probabilities for each hierarchical relationship;
                            Key: "downward", "peer", or "upward" / Value: relative probability (float or int)

    Outputs:
        - cumulative_g: NetworkX Graph. The cumulative collaboration network that connects all collaborations
                        happened in each week on top of the input <initial_g> graph.
                        
    '''

    cumulative_g = initial_g.copy()

    # unbiased random walk
    if bias == "unbiased":
        cumulative_g.add_edges_from(zip(df.emp_a, df.emp_b))

    elif bias == "hierarchy":
        df = df[["emp_a", "emp_b", "rel"]]
        final_df = assign_hierarchical_rel(df, paygrade_dict, paygrade_map, hierarchy_prob)
        # assign edge attributes (probability of random walks)
        cumulative_g.add_weighted_edges_from(zip(final_df.source, final_df.target, final_df.prob), weight='prob')

    elif bias == "strength":
        edges = zip(df.emp_a, df.emp_b)
        for edge in edges:
            if cumulative_g.has_edge(*edge):
                cumulative_g[edge[0]][edge[1]]['prob'] += 1
            else:
                cumulative_g.add_edge(*edge, prob=1)
    
    elif bias == "recency":
        edges = zip(df.emp_a, df.emp_b)
        for edge in edges:
            cumulative_g.add_edge(*edge, prob=week_num)
    
    return cumulative_g

        
