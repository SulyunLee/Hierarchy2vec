
import shelve
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import statistics
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from prepare_input_func import *
from models import *
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-individual', '--individual', type=bool, default=False, help="Add individual features for the node features")
    parser.add_argument('-collab', '--collab', type=bool, default=False, help="Add collaboration features for the node features")
    parser.add_argument('-start_year', '--start_year', type=int, default = 2002, help="Starting year of the dataset")
    parser.add_argument('-end_year', '--end_year', type=int, default = 2019, help="Ending year of the dataset")
    parser.add_argument('-train_split_year', '--train_split_year', type=int, help="Maximum year for training set")
    parser.add_argument('-valid_split_year', '--valid_split_year', type=int, help="Maximum year for validation set")
    parser.add_argument('-w', '--w', default=3, type=int, help="window size")
    parser.add_argument('-emb_size', '--emb_size', type=int, default=32, help="node embedding size")
    parser.add_argument('-hierarchy', '--hierarchy', type=bool, default=False, help="Include hierarchy biased walk for aggregation")
    parser.add_argument('-strength', '--strength', type=bool, default=False, help="Include strength biased walk for aggregation")
    parser.add_argument('-recency', '--recency', type=bool, default=False, help="Include recency biased walk for aggregation")
    parser.add_argument('-drop_rate', '--drop_rate', type=float, default=0, help="probability of dropout")
    parser.add_argument('-label', '--label', type=str, default="failure", help="Label name to be used in prediction")


    args = parser.parse_args()
    individual = args.individual
    collab = args.collab
    start_year = args.start_year
    end_year = args.end_year
    train_split_year = args.train_split_year
    valid_split_year = args.valid_split_year
    w = args.w
    emb_size = args.emb_size
    hierarchy = args.hierarchy
    strength = args.strength
    recency = args.recency
    drop_rate = args.drop_rate
    label = args.label

    #################################################################
    # Load datasets
    df_hierarchy_filename = "data/NFL_experiment/NFL_Coach_Data_with_features_emb{}_hierarchy_w{}.csv".format(emb_size, w)
    df_strength_filename = "data/NFL_experiment/NFL_Coach_Data_with_features_emb{}_strength_w{}.csv".format(emb_size, w)
    df_recency_filename = "data/NFL_experiment/NFL_Coach_Data_with_features_emb{}_recency_w{}.csv".format(emb_size, w)
    team_labels_filename = "data/NFL_experiment/team_labels.csv"

    df_hierarchy = pd.read_csv(df_hierarchy_filename)
    df_strength = pd.read_csv(df_strength_filename)
    df_recency = pd.read_csv(df_recency_filename)
    team_labels_df = pd.read_csv(team_labels_filename)
    #################################################################

    ### Define features
    individual_features = []
    collab_features = []
    if individual: 
        individual_features = ["TotalYearsInNFL", "Past5yrsWinningPerc_best", "Past5yrsWinningPerc_avg"]
    if collab:
        collab_features = df_hierarchy.columns[df_hierarchy.columns.str.contains("cumul_emb")].tolist()

    coach_feature_names = individual_features + collab_features
    
    # Sort dataframes by "Year" and "Team" names
    df_hierarchy.sort_values(by=["Year","Team"], inplace=True, ignore_index=True)
    df_strength.sort_values(by=["Year","Team"], inplace=True, ignore_index=True)
    df_recency.sort_values(by=["Year","Team"], inplace=True, ignore_index=True)
    team_labels_df.sort_values(by=["Year","Team"], inplace=True, ignore_index=True)

    ### Split data into train, validation, and test sets
    train_hierarchy, valid_hierarchy, test_hierarchy = \
            split_train_valid_test(df_hierarchy, start_year, train_split_year, valid_split_year, end_year)

    train_strength, valid_strength, test_strength = \
            split_train_valid_test(df_strength, start_year, train_split_year, valid_split_year, end_year)

    train_recency, valid_recency, test_recency = \
            split_train_valid_test(df_recency, start_year, train_split_year, valid_split_year, end_year)

    train_labels, valid_labels, test_labels = \
            split_train_valid_test(team_labels_df, start_year, train_split_year, valid_split_year, end_year)

    print("Number of training records: {}, validation records: {}, testing records: {}".format(train_hierarchy.shape[0], valid_hierarchy.shape[0], test_hierarchy.shape[0]))

    # Generate team features
    normalized_train_team_feature = normalized_valid_team_feature = normalized_test_team_feature = torch.Tensor()

    # Split data into different parts
    train_individual_arr, valid_individual_arr, test_individual_arr = \
            np.array(train_hierarchy[individual_features]), \
            np.array(valid_hierarchy[individual_features]), \
            np.array(test_hierarchy[individual_features])
    train_hierarchy_collab_arr, valid_hierarchy_collab_arr, test_hierarchy_collab_arr =\
            np.array(train_hierarchy[collab_features]),\
            np.array(valid_hierarchy[collab_features]), \
            np.array(test_hierarchy[collab_features])
    train_strength_collab_arr, valid_strength_collab_arr, test_strength_collab_arr =\
            np.array(train_strength[collab_features]),\
            np.array(valid_strength[collab_features]), \
            np.array(test_strength[collab_features])
    train_recency_collab_arr, valid_recency_collab_arr, test_recency_collab_arr =\
            np.array(train_recency[collab_features]),\
            np.array(valid_recency[collab_features]), \
            np.array(test_recency[collab_features])
    train_team_info_arr, valid_team_info_arr, test_team_info_arr = \
            np.array(train_hierarchy[["Year","Team"]]),\
            np.array(valid_hierarchy[["Year","Team"]]),\
            np.array(test_hierarchy[["Year","Team"]])

    ### Normalization
    normalized_train_individual, normalized_valid_individual, normalized_test_individual\
            = normalize(train=train_individual_arr, \
                        valid=valid_individual_arr,\
                        test=test_individual_arr)
    normalized_train_hierarchy, normalized_valid_hierarchy, normalized_test_hierarchy\
            = normalize(train=train_hierarchy_collab_arr, \
                        valid=valid_hierarchy_collab_arr,\
                        test=test_hierarchy_collab_arr)
    normalized_train_strength, normalized_valid_strength, normalized_test_strength\
            = normalize(train=train_strength_collab_arr, \
                        valid=valid_strength_collab_arr,\
                        test=test_strength_collab_arr)
    normalized_train_recency, normalized_valid_recency, normalized_test_recency\
            = normalize(train=train_recency_collab_arr, \
                        valid=valid_recency_collab_arr,\
                        test=test_recency_collab_arr)

    # Modeling
    print("Training model...")
    loss = nn.BCEWithLogitsLoss()
    epochs = 100000

    repeated_train_loss_arr = np.zeros((10))
    repeated_valid_loss_arr = np.zeros((10))
    repeated_test_loss_arr = np.zeros((10))
    repeated_train_auc_arr = np.zeros((10))
    repeated_valid_auc_arr = np.zeros((10))
    repeated_test_auc_arr = np.zeros((10))
    repeated_train_acc_arr = np.zeros((10))
    repeated_valid_acc_arr = np.zeros((10))
    repeated_test_acc_arr = np.zeros((10))

    team_emb_dim = len(collab_features)

    fpr_dict = dict()
    tpr_dict = dict()

    for seed in range(0,10):
        torch.manual_seed(seed)

        model = Nonhier_NN_BiasAttn(individual_features = individual_features,
                                    collab_features = collab_features,
                                    team_emb_dim = team_emb_dim,
                                    team_info_names = ["Year","Team"],
                                    label_name = label,
                                    is_hierarchy = hierarchy,
                                    is_strength = strength,
                                    is_recency = recency,
                                    drop_rate = drop_rate)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Early stopping
        stopper = EarlyStopping(patience=50)

        train_loss_arr = np.zeros((epochs)) 
        valid_loss_arr = np.zeros((epochs)) 
        test_loss_arr = np.zeros((epochs)) 
        train_auc_arr = np.zeros((epochs)) 
        valid_auc_arr = np.zeros((epochs)) 
        test_auc_arr = np.zeros((epochs)) 
        train_acc_arr = np.zeros((epochs)) 
        valid_acc_arr = np.zeros((epochs)) 
        test_acc_arr = np.zeros((epochs)) 

        test_fpr_arr = dict()
        test_tpr_arr = dict()

        for epoch in range(epochs):
            model.train()

            optimizer.zero_grad()

            train_y, train_y_hat, train_alpha = model(individual_f=normalized_train_individual,
                                    hierarchy_collab_f=normalized_train_hierarchy,
                                    strength_collab_f=normalized_train_strength,
                                    recency_collab_f=normalized_train_recency,
                                    team_info=train_team_info_arr,
                                    labels=train_labels)
            train_loss = loss(train_y_hat, train_y)
            train_loss_arr[epoch] = train_loss

            # get the train predictions
            train_prob = torch.sigmoid(train_y_hat)
            train_pred = torch.round(train_prob)
            train_auc = round(roc_auc_score(train_y, train_prob.detach().numpy()), 3)
            train_acc = (train_pred == train_y).float().sum() / len(train_y)
            train_acc = round(train_acc.item(), 3)
            train_auc_arr[epoch] = train_auc
            train_acc_arr[epoch] = train_acc

            train_loss.backward()
            optimizer.step()
        
            with torch.no_grad():
                model.eval()

                # predict on the validation set
                valid_y, valid_y_hat, valid_alpha = model(individual_f=normalized_valid_individual,
                                        hierarchy_collab_f=normalized_valid_hierarchy,
                                        strength_collab_f=normalized_valid_strength,
                                        recency_collab_f=normalized_valid_recency,
                                        team_info=valid_team_info_arr,
                                        labels=valid_labels)
                valid_loss = loss(valid_y_hat, valid_y)
                valid_loss_arr[epoch] = valid_loss

                # get the valid predictions
                valid_prob = torch.sigmoid(valid_y_hat)
                valid_pred = torch.round(valid_prob)
                valid_auc = round(roc_auc_score(valid_y, valid_prob.detach().numpy()), 3)
                valid_acc = (valid_pred == valid_y).float().sum() / len(valid_y)
                valid_acc = round(valid_acc.item(), 3)
                valid_auc_arr[epoch] = valid_auc
                valid_acc_arr[epoch] = valid_acc

                # predict on the test set
                test_y, test_y_hat, test_alpha = model(individual_f=normalized_test_individual,
                                        hierarchy_collab_f=normalized_test_hierarchy,
                                        strength_collab_f=normalized_test_strength,
                                        recency_collab_f=normalized_test_recency,
                                        team_info=test_team_info_arr,
                                        labels=test_labels)
                test_loss = loss(test_y_hat, test_y)
                test_loss_arr[epoch] = test_loss

                # get the test predictions
                test_prob = torch.sigmoid(test_y_hat)
                test_pred = torch.round(test_prob)
                test_auc = round(roc_auc_score(test_y, test_prob.detach().numpy()), 3)
                fpr, tpr, _ = roc_curve(test_y, test_prob)
                test_acc = (test_pred == test_y).float().sum() / len(test_y)
                test_acc = round(test_acc.item(), 3)
                test_fpr_arr[epoch] = fpr
                test_tpr_arr[epoch] = tpr
                test_auc_arr[epoch] = test_auc
                test_acc_arr[epoch] = test_acc

                counter, stop = stopper.step(valid_loss, model)
                if counter == 1:
                    remember_epoch = epoch - 1
                if stop:
                    break

        repeated_train_loss_arr[seed] = train_loss_arr[remember_epoch]
        repeated_valid_loss_arr[seed] = valid_loss_arr[remember_epoch]
        repeated_test_loss_arr[seed] = test_loss_arr[remember_epoch]
        repeated_train_auc_arr[seed] = train_auc_arr[remember_epoch]
        repeated_valid_auc_arr[seed] = valid_auc_arr[remember_epoch]
        repeated_test_auc_arr[seed] = test_auc_arr[remember_epoch]
        repeated_train_acc_arr[seed] = train_acc_arr[remember_epoch]
        repeated_valid_acc_arr[seed] = valid_acc_arr[remember_epoch]
        repeated_test_acc_arr[seed] = test_acc_arr[remember_epoch]
        fpr_dict[seed] = test_fpr_arr[remember_epoch]
        tpr_dict[seed] = test_tpr_arr[remember_epoch]


    print("Train loss \t Valid loss \t Test loss") 
    print(round(repeated_train_loss_arr.mean(),3), round(repeated_valid_loss_arr.mean(),3), round(repeated_test_loss_arr.mean(),3))
    print("Train AUC \t Valid AUC \t Test AUC")
    print(round(repeated_train_auc_arr.mean(),3), round(repeated_valid_auc_arr.mean(),3), round(repeated_test_auc_arr.mean(),3))
    print("Train Accuracy \t Valid Accuracy \t Test Accuracy")
    print(round(repeated_train_acc_arr.mean(),3), round(repeated_valid_acc_arr.mean(),3), round(repeated_test_acc_arr.mean(),3))

    # plot the ROC curve
#     i = 9
#     roc_auc = auc(fpr_dict[i], tpr_dict[i])

#     d = shelve.open("results/test_fpr_tpr")
#     d['hierarchy2vec_FlatAgg'] = {'fpr': fpr_dict[i], \
#                         'tpr': tpr_dict[i], \
#                         'auc': roc_auc}
#     d.close()






