'''
This script generates node embedding that considers the collaboration information of coaches
during the previous collaboration experience.
The node embeddings are generated based on the following random walk schemes:
1) Unbiased walk (Vanilla DeepWalk)
2) Hierarchy-based walk
3) Strength (Duration)-based walk
4) Recency-based walk
'''
import random
import gensim
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from construct_team_network_func import *
from datatype_change import *
from data_cleaning_func import *


def get_random_walk(G, node, walk_length, num_walks, bias, current_year):
    '''
    Given a graph and a node, return a sequence of nodes generated by random walks.
    Input:
      - G: NetworkX Graph. Graph of the collaboration network
      - node: NetworkX Node. The target node to start the random walks
      - walk_length: Int. The length of the node sequence generated by random walks
      - num_walks: Int. The number of random walk repeat
      - bias: Str. The type of random walk scheme. The possible options are: "unbiased",
              "hierarchy", "strength", and "recency" 
      - current_year: Int. The current year (prediction year). This is used to calculate
                      the recency of collaborations.

    Output:
      - walk paths: List. The list of lists that contains sequences of nodes generated by random walks.
    '''
    walk_paths = []
    # repeat the random walks for the <num_walks> times
    for walk in range(num_walks):
        path = [node]
        current_node = node
        # sample the next visiting node for the walk length
        random.seed(100)
        for step in range(walk_length):
            # sample neighborhoods based on the hierarchical probability distribution.
            if bias == "hierarchy":
                out_edges = G.out_edges(current_node)
                probability_list = [G.get_edge_data(edge[0], edge[1])['prob'] for edge in out_edges]
                next_walk = random.choices(list(out_edges), weights=probability_list, k=1)[0]
                next_visit = next_walk[1]
            # sample neighborhoods based on the collaboration frequency (sqrt transformation)
            elif bias == "strength":
                edges = G.edges(current_node)
                probability_list = [np.sqrt(G.get_edge_data(edge[0], edge[1])['prob']) for edge in edges]
                next_walk = random.choices(list(edges), weights=probability_list, k=1)[0]
                next_visit = next_walk[1]
            # sample neighborhoods based on the recency of collaboration (sqrt transformation)
            elif bias == "recency":
                edges = G.edges(current_node)
                probability_list = [np.sqrt(1/(current_year-G.get_edge_data(edge[0], edge[1])['last_year'])) for edge in edges]
                next_walk = random.choices(list(edges), weights=probability_list, k=1)[0]
                next_visit = next_walk[1]
            # uniform sampling distribution for the unbiased random walks
            else:
                neighbors = list(nx.all_neighbors(G, current_node)) # extract neighbors
                next_visit = random.choice(neighbors) # randomly select the next visiting node

            path.append(next_visit)
            current_node = next_visit
        walk_paths.append(path)

    # return the list of walks
    return walk_paths
        
def deepwalk(model, G, walk_length, num_walks, window_size, emb_size, epochs, update, bias, current_year):
    '''
    Use DeepWalk (Skip-gram) approach to learn the node embeddings for nodes in the 
    given graph <G>.
    Inputs:
      - model: Gensim Word2Vec Model. The initial model to be used for generating node embeddings. The optimizations of node embeddings
               starts from the existing model without randomly initializing parameters.
      - G: NetworkX Graph. The cumulative collaboration network to generate the node embeddings
      - walk_length: Int. The length of the node sequence generated by random walks
      - num_walks: Int. The number of random walk repeat
      - window_size: Int. The size of sliding window.
      - emb_size: Int. The size of the node embeddings (the number of vector dimensions)
      - epochs: Int. The number of iterations for optimizing embeddings
      - update: bool. Indicate whether the model should be updated based on the previous model.
                If the <update> is True, the embeddings are updated starting from the embeddings learned
                from the provided <model>.
      - bias: Str. The type of random walk scheme. The possible options are: "unbiased",
              "hierarchy", "strength", and "recency" 
      - current_year: Int. The current year (prediction year). This is used to calculate
                      the recency of collaborations.

    Output:
      - model: Gensim Word2Vec Model. Model that is trained based on the given graph <G>.
      - nodes: List. List of nodes in the graph <G>
      - embeddings: Numpy 2-D Array. Array that contains the learned embeddings of nodes. Each row is a node
                    and each column is a dimension of the embeddings.
    '''
    total_walk_paths = [] # list that stores all walks for all nodes

    ### RANDOM WALK 
    for node in G.nodes():
        if G.degree(node) != 0:
            walk_paths = get_random_walk(G, node, walk_length, num_walks, bias, current_year)
            total_walk_paths.extend(walk_paths)

    # if this is the first model to train, initialize the model with the Word2Vec model.
    if update != True:
        # initiate word2vec model
        model = gensim.models.Word2Vec(size=emb_size, window=window_size, sg=1, hs=1, workers=3, seed=100)

    # Build vocabulary
    model.build_vocab(total_walk_paths, update=update)

    # Train
    model.train(total_walk_paths, total_examples=model.corpus_count, epochs=epochs)
    nodes = list(model.wv.vocab) # list of node names
    embeddings = model.wv.__getitem__(model.wv.vocab) # embeddings for every node

    return model, nodes, embeddings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_size', '--emb_size', default=32, type=int, help="The number of embedding dimensions for each node")
    parser.add_argument('-window_size', '--window_size', default=3, type=int, help="The window size of Skip-gram model")
    parser.add_argument('-bias', '--bias', default='unbiased', type=str, help="The bias given to the random walk (option: unbiased, hierarchy, strength, recency)")

    args = parser.parse_args()
    emb_size = args.emb_size
    window_size = args.window_size
    bias = args.bias

    #################################################################
    # Load datasets
    NFL_coach_record_filename = "data/NFL_experiment/NFL_Coach_Data_final_position.csv"
    history_record_filename = "data/NFL_experiment/all_coach_records_cleaned.csv"

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    history_df = pd.read_csv(history_record_filename)
    #################################################################
    # Parameter setting for node embedding 
    walk_length = 40
    num_walks = 10
    epochs = 30

    selection_prob = {'downward': 1, 'peer': 3, 'upward': 5} # relative probabilities of hierarchical ties

    #################################################################
    ## Clean NFL and coach records datasets
    #################################################################
    # clean NFL coach record data
    NFL_record_df = clean_NFL_data(NFL_record_df, 2002, 2019)

    # clean NFL coaches' history record data
    nfl_coaches = NFL_record_df.Name.unique()
    NFL_coach_history_qualified = clean_history_data(history_df, nfl_coaches)


    #################################################################
    ## Generate cumulative collaboration network embedding
    #################################################################
    print("Generating NFL & college coaching network before 2002")

    ### Construct the cumulative network for all coaching records before 2002
    cumulative_NFL_collab_G_dict = dict() # stores graphs built by cumulative collaboration ties in each year.

    ### Construct network
    # walk is not biased (unweighted)
    if bias == "unbiased":
        # construct colleague network up to 2001.
        before2002_collab_G = nx.Graph()
        before2002_collab_G = construct_cumulative_colleague_network(before2002_collab_G, NFL_coach_history_qualified, int(NFL_coach_history_qualified.StartYear.min()), 2001, "unbiased")

    # walk is biased (weighted by hierarchy level)
    elif bias == "hierarchy":
        before2002_collab_G = nx.DiGraph()
        before2002_collab_G = construct_cumulative_colleague_network(before2002_collab_G, NFL_coach_history_qualified, int(NFL_coach_history_qualified.StartYear.min()), 2001, \
                                                                    "hierarchy", selection_prob)

    # walk is biased (weighted by the frequency of collaborations)
    elif bias == "strength":
        before2002_collab_G = nx.Graph()
        before2002_collab_G = construct_cumulative_colleague_network(before2002_collab_G, NFL_coach_history_qualified, int(NFL_coach_history_qualified.StartYear.min()), 2001, "strength")
    
    # walk is biased (weighted by the recency of collaborations)
    elif bias == "recency":
        before2002_collab_G = nx.Graph()
        before2002_collab_G = construct_cumulative_colleague_network(before2002_collab_G, NFL_coach_history_qualified, int(NFL_coach_history_qualified.StartYear.min()), 2001, "recency")

    # network statistics of collaboration network before 2002
    print(nx.info(before2002_collab_G))
    try:
        num_cc = nx.number_connected_components(before2002_collab_G)
    except:
        num_cc = nx.number_strongly_connected_components(before2002_collab_G)
    print("Number of connected components: {}".format(num_cc))

    # save graph
    cumulative_NFL_collab_G_dict[2001] = before2002_collab_G

    ### Deepwalk based on the cumulative network before 2002
    print("Generating embeddings of cumulative collaboration network before 2002")
    model, before2002_nodes, before2002_emb = deepwalk(None, before2002_collab_G, walk_length, num_walks, window_size, emb_size, epochs, False, bias, 2002)

    ### Create a dictionary that contains the coaches' embedding in each year.
    ### - Key: the coach name
    ### - Value: dictionary of embeddings for each year.
    ###     -Key: the next year (prediction year). 
    ###         e.g., if 2002 collaborations are added to the network,
    ###                 the prediction year is 2003.
    ###     - Value: the embedding to be used for the prediction.
    cumulative_node_emb_dict = dict()
    for idx, node in enumerate(before2002_nodes):
        cumulative_node_emb_dict[node] = dict()
        cumulative_node_emb_dict[node][2002] = before2002_emb[idx]
        
    ### Construct the cumulative network by adding one year of NFL & college football record
    ### to the existing network before 2002
    years = range(2002, 2020)
    cumulative_NFL_collab_G = before2002_collab_G.copy()
    for year in years:
        print("Constructing cumulative network for year {}".format(year))
        # add one year of collaborations
        # unbiased random walk
        if bias == "unbiased":
            cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_record_df, year, year)
            cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_coach_history_qualified, year, year)
            num_cc = nx.number_connected_components(cumulative_NFL_collab_G)

        # hierarchy-based random walk
        elif bias == "hierarchy":
            cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_record_df, year, year, selection_prob)
            cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_coach_history_qualified, year, year, selection_prob)
            num_cc = nx.number_strongly_connected_components(cumulative_NFL_collab_G)

        # strength-based random walk
        elif bias == "strength":
            cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_record_df, year, year)
            cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_coach_history_qualified, year, year)
            num_cc = nx.number_connected_components(cumulative_NFL_collab_G)

        # recency-based random walk
        elif bias == "recency":
            cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_record_df, year, year) 
            cumulative_NFL_collab_G = construct_cumulative_colleague_network(cumulative_NFL_collab_G, NFL_coach_history_qualified, year, year)  

        print(nx.info(cumulative_NFL_collab_G))
        print("Number of connected components: {}".format(num_cc))

        # save graph
        cumulative_NFL_collab_G_dict[year] = cumulative_NFL_collab_G
        # Learn node embeddings: update the embeddings starting from the previous learned model.
        model, cumulative_NFL_nodes, cumulative_NFL_emb = deepwalk(model, cumulative_NFL_collab_G, walk_length, num_walks, window_size, emb_size, epochs, True, bias, year+1)

        # Add new embedding to the dictionary
        for idx, node in enumerate(cumulative_NFL_nodes):
            if node in cumulative_node_emb_dict:
                cumulative_node_emb_dict[node][year+1] = cumulative_NFL_emb[idx]
            else:
                cumulative_node_emb_dict[node] = dict()
                cumulative_node_emb_dict[node][year+1] = cumulative_NFL_emb[idx]

    # convert the node embedding dictionary to a dataframe
    cumulative_emb_df = dict_of_dict_to_dataframe(cumulative_node_emb_dict, emb_size)
    cumulative_emb_df.to_csv("data/NFL_experiment/embeddings/cumulative_collab_G_node_embedding_size{}_{}_w{}_df.csv".format(emb_size, bias, window_size), index=False, encoding="utf-8-sig")