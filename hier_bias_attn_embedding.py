
import shelve
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import statistics
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_curve, auc
from prepare_input_func import *
from models import *
from utils import *

def split_features(x, indiv_f_dim, collab_f_dim):
    indiv_f = x[:,:indiv_f_dim]
    collab_f_H = x[:,indiv_f_dim:indiv_f_dim + collab_f_dim]
    collab_f_S = x[:,indiv_f_dim+collab_f_dim:indiv_f_dim+collab_f_dim*2]
    collab_f_R = x[:,indiv_f_dim+collab_f_dim*2:]

    return indiv_f, collab_f_H, collab_f_S, collab_f_R

def generate_season_ids(df):
    season_ids = dict()
    id_num = 0
    id_arr = np.zeros((df.shape[0], 1))
    for idx, row in df.iterrows():
        if (row.Year, row.Team) in season_ids:
            id_arr[idx] = season_ids[(row.Year, row.Team)]
        else:
            season_ids[(row.Year, row.Team)] = id_num
            id_arr[idx] = id_num
            id_num += 1

    return id_arr

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
    strength = args.strength
    recency = args.recency
    hierarchy = args.hierarchy
    drop_rate = args.drop_rate
    label = args.label

    robustness_check = False
    hier_var = '1:3:5'
    strength_var = 'sqrt'
    #################################################################
    # Load datasets
    if robustness_check:
        df_hierarchy_filename = "data/robustness_check/NFL_Coach_Data_with_features_emb{}_hierarchy{}_w{}.csv".format(emb_size, hier_var, w)
        df_strength_filename = "data/robustness_check/NFL_Coach_Data_with_features_emb{}_strength{}_w{}.csv".format(emb_size, strength_var, w)
        df_recency_filename = "data/robustness_check/NFL_Coach_Data_with_features_emb{}_recency{}_w{}.csv".format(emb_size, strength_var, w)
    else:
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
    df_hierarchy.sort_values(by=["Year","Team"], inplace=True)
    df_hierarchy.reset_index(inplace=True, drop=True)
    df_hierarchy['ID'] = df_hierarchy.index

    df_strength.sort_values(by=["Year","Team"], inplace=True)
    df_strength.reset_index(inplace=True, drop=True)
    df_strength['ID'] = df_strength.index

    df_recency.sort_values(by=["Year","Team"], inplace=True)
    df_recency.reset_index(inplace=True, drop=True)
    df_recency['ID'] = df_recency.index

    team_labels_df.sort_values(by=["Year","Team"], inplace=True)


    # Combine individual, hierarchy, strength, and recency features
    df_combined = df_hierarchy[["ID", "Year", "Team", "final_position"] + coach_feature_names]
    df_combined = pd.merge(df_combined, df_strength[["ID"] + collab_features], on = "ID", how="left")
    df_combined = pd.merge(df_combined, df_recency[["ID"] + collab_features], on="ID", how="left")
    df_combined.columns = ["ID", "Year", "Team", "final_position"] + individual_features + ["emb_Hier{}".format(i) for i in range(emb_size)] +\
                                                        ["emb_Strength{}".format(i) for i in range(emb_size)]+\
                                                        ["emb_Recency{}".format(i) for i in range(emb_size)]

    # position ids
    pos_ids = {'O':1, 'D':2, 'OC':3, 'DC':4, 'HC':5}

    # season ids
    train_seasons = df_combined[(df_combined.Year >= start_year) & (df_combined.Year <= train_split_year)][["Year","Team"]]
    train_seasons.reset_index(drop=True, inplace=True)
    train_season_ids = torch.Tensor(generate_season_ids(train_seasons))

    valid_seasons = df_combined[(df_combined.Year > train_split_year ) & (df_combined.Year <= valid_split_year)][["Year","Team"]]
    valid_seasons.reset_index(drop=True, inplace=True)
    valid_season_ids = torch.Tensor(generate_season_ids(valid_seasons))

    test_seasons = df_combined[(df_combined.Year > valid_split_year) & (df_combined.Year <= end_year)][["Year","Team"]]
    test_seasons.reset_index(drop=True, inplace=True)
    test_season_ids = torch.Tensor(generate_season_ids(test_seasons))

    # position_ids
    position_mapped = df_combined["final_position"].map(pos_ids)
    df_combined = df_combined.assign(pos_id = position_mapped)


    ### Split data into train, validation, and test sets
    train_x, valid_x, test_x = \
            split_train_valid_test(df_combined, start_year, train_split_year, valid_split_year, end_year)

    train_labels, valid_labels, test_labels = \
            split_train_valid_test(team_labels_df, start_year, train_split_year, valid_split_year, end_year)

    print("Number of training records: {}, validation records: {}, testing records: {}".format(train_x.shape[0], valid_x.shape[0], test_x.shape[0]))

    train_pos_ids = torch.Tensor(train_x.pos_id.values).view(-1,1)
    valid_pos_ids = torch.Tensor(valid_x.pos_id.values).view(-1,1)
    test_pos_ids = torch.Tensor(test_x.pos_id.values).view(-1,1)

    ### drop unused columns
    train_x = train_x.drop(columns=["ID", "Year","Team","final_position", "pos_id"])
    valid_x = valid_x.drop(columns=["ID", "Year","Team","final_position", "pos_id"])
    test_x = test_x.drop(columns=["ID", "Year","Team","final_position", "pos_id"])


    ### Normalization
    normalized_train_x, normalized_valid_x, normalized_test_x \
            = normalize(train=train_x, \
                        valid=valid_x, \
                        test=test_x)


    ### Split features
    normalized_train_indiv, normalized_train_collab_H, normalized_train_collab_S, normalized_train_collab_R \
            = split_features(normalized_train_x, len(individual_features), emb_size)

    normalized_valid_indiv, normalized_valid_collab_H, normalized_valid_collab_S, normalized_valid_collab_R \
            = split_features(normalized_valid_x, len(individual_features), emb_size)

    normalized_test_indiv, normalized_test_collab_H, normalized_test_collab_S, normalized_test_collab_R \
            = split_features(normalized_test_x, len(individual_features), emb_size)

    ### create tensors of ids
    train_ids, valid_ids, test_ids = \
            torch.Tensor(train_x.iloc[:,0]), torch.Tensor(valid_x.iloc[:,0]),\
                            torch.Tensor(test_x.iloc[:,0])


    ### Labels
    train_y = torch.Tensor(train_labels[label]).view(-1,1)
    valid_y = torch.Tensor(valid_labels[label]).view(-1,1)
    test_y = torch.Tensor(test_labels[label]).view(-1,1)

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

        # initialize model
        model = Hier_NN_BiasAttn(indiv_f_dim = len(individual_features),
                                  collab_f_dim = len(collab_features),
                                  emb_dim = team_emb_dim,
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

            train_y_hat = model(indiv=normalized_train_indiv,\
                            hierarchy_f=normalized_train_collab_H,\
                            strength_f=normalized_train_collab_S,\
                            recency_f=normalized_train_collab_R,\
                            season_ids=train_season_ids,\
                            pos_ids=train_pos_ids)
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
                valid_y_hat = model(indiv=normalized_valid_indiv,\
                                hierarchy_f=normalized_valid_collab_H,\
                                strength_f=normalized_valid_collab_S,\
                                recency_f=normalized_valid_collab_R,\
                                season_ids=valid_season_ids,\
                                pos_ids=valid_pos_ids)
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
                test_y_hat = model(indiv=normalized_test_indiv,\
                                hierarchy_f=normalized_test_collab_H,\
                                strength_f=normalized_test_collab_S,\
                                recency_f=normalized_test_collab_R,\
                                season_ids=test_season_ids,\
                                pos_ids=test_pos_ids)
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

    # plot the loss over epochs
    plt.plot(range(0, remember_epoch), train_loss_arr[:remember_epoch], label="Train loss")
    plt.plot(range(0, remember_epoch), valid_loss_arr[:remember_epoch], label="Validation loss")
    plt.plot(range(0, remember_epoch), test_loss_arr[:remember_epoch], label="Test loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/losses.png")
    plt.close()

    # plot the ROC curve
    # i = 3
    # roc_auc = auc(fpr_dict[i], tpr_dict[i])

    # d = shelve.open("results/test_fpr_tpr")
    # d['hierarchy2vec'] = {'fpr': fpr_dict[i], \
    #                     'tpr': tpr_dict[i], \
    #                     'auc': roc_auc}
    # d.close()









