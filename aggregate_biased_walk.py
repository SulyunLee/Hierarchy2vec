
import numpy as np
import pandas as pd
import argparse

if __name__ == "__main__":
    """
    This script aggregates biased walk using "average" aggregation 
    The output csv file can be used as input to Hierarchy2vec AvgNetEmb
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_size', '--emb_size', type=int, default=32, help="node embedding size")
    parser.add_argument('-hier', '--hier', default=False, type=bool, help="Mentorship network or not")
    parser.add_argument('-w', '--w', default=3, type=int, help="window size")
    parser.add_argument('-hierarchy', '--hierarchy', default=False, type=bool, help="hierarchically biased walk")
    parser.add_argument('-strength', '--strength', default=False, type=bool, help="strength biased walk")
    parser.add_argument('-recency', '--recency', default=False, type=bool, help="recency biased walk")

    args = parser.parse_args()
    emb_size = args.emb_size
    hier = args.hier
    w = args.w
    hierarchy = args.hierarchy
    strength = args.strength
    recency = args.recency

    #################################################################
    # Load datasets
    df_hierarchy_filename = "data/NFL_Coach_Data_with_features_emb{}_hierarchy_w{}.csv".format(emb_size, w)
    df_strength_filename = "data/NFL_Coach_Data_with_features_emb{}_strength_w{}.csv".format(emb_size, w)
    df_recency_filename = "data/NFL_Coach_Data_with_features_emb{}_recency_w{}.csv".format(emb_size, w)

    #################################################################
    collab_feature_names = ["cumul_emb{}".format(i) for i in range(emb_size)]

    bias_list = [hierarchy, strength, recency]
    bias_name_list = ["hierarchy", "strength", "recency"]
    collab_array_list = []
    for idx in range(len(bias_list)):
        bias = bias_list[idx]
        bias_name = bias_name_list[idx]

        if bias:
            fname = "data/NFL_Coach_Data_with_features_emb{}_{}_w{}.csv".format(emb_size, bias_name, w)
            df = pd.read_csv(fname)
            collab_array = np.array(df[collab_feature_names])
            collab_array_list.append(collab_array)

    # average collaboration features across hierarchy, strength, and recency
    avg_collab_feature = np.mean(collab_array_list, axis=0)

    # assign averaged collaboration features to the dataframe columns
    df[collab_feature_names] = avg_collab_feature

    df.to_csv("data/NFL_Coach_Data_with_features_emb{}_averaged_w{}.csv".format(emb_size, w), index=False, encoding="utf-8-sig")

            
