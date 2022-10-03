
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import statistics
from tqdm import tqdm
from prepare_input_func import *
from models import *
from utils import *

def prepare_inputs(record_df, labels_df, node_feature_names, label_name):
    '''
    This function generates vectors for averaged coach features, team features, and labels for each instance
    '''
    seasons = record_df[["Year", "Team"]].drop_duplicates().to_dict('records')
    num_seasons = len(seasons)
    print("Number of seasons: {}".format(num_seasons))

    # initialize vectors
    avg_feature_arr = np.zeros((num_seasons, len(node_feature_names)))
    team_label_arr = np.zeros((num_seasons)).astype(int)

    for idx, season in enumerate(seasons):
        year = season['Year']
        team = season['Team']

        # average all coach features
        avg_feature_arr[idx,:] = average_node_features(team, year, record_df, node_feature_names)

        # team label
        team_label_arr[idx] = int(get_team_label(team, year, labels_df, label_name))

    return avg_feature_arr, team_label_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-individual', '--individual', type=bool, default=False, help="Add individual features for the node features")
    parser.add_argument('-collab', '--collab', type=bool, default=False, help="Add collaboration features for the node features")
    parser.add_argument('-start_year', '--start_year', type=int, default = 2002, help="Starting year of the dataset")
    parser.add_argument('-end_year', '--end_year', type=int, default = 2019, help="Ending year of the dataset")
    parser.add_argument('-train_split_year', '--train_split_year', type=int, help="Maximum year for training set")
    parser.add_argument('-valid_split_year', '--valid_split_year', type=int, help="Maximum year for validation set")
    parser.add_argument('-bias', '--bias', default='unbiased', type=str, help="Unbiased or biased random walk while generating node embeddings (options: unbiased, hierarchy, strength, recency, averaged)")
    parser.add_argument('-w', '--w', default=3, type=int, help="window size")
    parser.add_argument('-emb_size', '--emb_size', type=int, default=32, help="node embedding size")
    parser.add_argument('-drop_rate', '--drop_rate', type=float, default=0, help="Dropout rate")
    parser.add_argument('-label', '--label', type=str, default="failure", help="Label name to be used in prediction")


    args = parser.parse_args()
    individual = args.individual
    collab = args.collab
    start_year = args.start_year
    end_year = args.end_year
    train_split_year = args.train_split_year
    valid_split_year = args.valid_split_year
    bias = args.bias
    w = args.w
    emb_size = args.emb_size
    drop_rate = args.drop_rate
    label = args.label

    #################################################################
    # Load datasets
    NFL_coach_data_filename = "data/NFL_Coach_Data_with_features_emb{}_{}_w{}.csv".format(emb_size, bias,w)
    team_labels_filename = "data/team_labels.csv"

    NFL_record_df = pd.read_csv(NFL_coach_data_filename)
    team_labels_df = pd.read_csv(team_labels_filename)
    #################################################################

    ### Define features
    individual_features = ["TotalYearsInNFL", "Past5yrsWinningPerc_best", "Past5yrsWinningPerc_avg"]
    collab_features = NFL_record_df.columns[NFL_record_df.columns.str.contains("cumul_emb")].tolist()

    
    # Define node feature names
    coach_feature_names = []
    if individual:
        coach_feature_names += individual_features
    if collab:
        coach_feature_names += collab_features

    team_feature_names = []


    ### Split data into train, validation, and test sets
    train_record, valid_record, test_record = \
            split_train_valid_test(NFL_record_df, start_year, train_split_year, valid_split_year, end_year)

    train_labels, valid_labels, test_labels = \
            split_train_valid_test(team_labels_df, start_year, train_split_year, valid_split_year, end_year)

    print("Number of training records: {}, validation records: {}, testing records: {}".format(train_record.shape[0], valid_record.shape[0], test_record.shape[0]))


    ### Generate team embeddings...
    print("Generating input features and labels...")
    train_agg_node_features, train_labels = \
            prepare_inputs(train_record, train_labels, \
            coach_feature_names, label)

    valid_agg_node_features, valid_labels = \
            prepare_inputs(valid_record, valid_labels, \
            coach_feature_names, label)

    test_agg_node_features, test_labels = \
            prepare_inputs(test_record, test_labels, \
            coach_feature_names, label)

    ### normalize input features
    normalized_train_x, normalized_valid_x, normalized_test_x = \
            normalize(train=train_agg_node_features, \
                    valid=valid_agg_node_features, \
                    test=test_agg_node_features)
    
    normalized_train_team_feature = normalized_valid_team_feature = normalized_test_team_feature = torch.Tensor()

    
    # Convert labels to tensors
    train_labels = torch.Tensor(train_labels).view(train_labels.shape[0], 1)
    valid_labels = torch.Tensor(valid_labels).view(valid_labels.shape[0], 1)
    test_labels = torch.Tensor(test_labels).view(test_labels.shape[0],1)

        
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

    team_emb_dim = normalized_train_x.shape[1]

    for seed in range(0,10):
        torch.manual_seed(seed)

        model = Nonhier_NN(input_feature_dim = normalized_train_x.shape[1],
                            team_emb_dim=team_emb_dim,
                            team_feature_dim=len(team_feature_names),
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

        for epoch in range(epochs):
            model.train()

            optimizer.zero_grad()

            train_y_hat = model(normalized_train_x, normalized_train_team_feature)
            train_loss = loss(train_y_hat, train_labels)
            train_loss_arr[epoch] = train_loss

            # get the train predictions
            train_prob = torch.sigmoid(train_y_hat)
            train_pred = torch.round(train_prob)
            train_auc = round(roc_auc_score(train_labels.detach().numpy(), train_prob.detach().numpy()), 3)
            train_auc_arr[epoch] = train_auc

            train_loss.backward()
            optimizer.step()
        
            with torch.no_grad():
                model.eval()

                # predict on the validation set
                valid_y_hat = model(normalized_valid_x, normalized_valid_team_feature)
                valid_loss = loss(valid_y_hat, valid_labels)
                valid_loss_arr[epoch] = valid_loss

                # get the valid predictions
                valid_prob = torch.sigmoid(valid_y_hat)
                valid_pred = torch.round(valid_prob)
                valid_auc = round(roc_auc_score(valid_labels.detach().numpy(), valid_prob.detach().numpy()), 3)
                valid_auc_arr[epoch] = valid_auc

                # predict on the test set
                test_y_hat = model(normalized_test_x, normalized_test_team_feature)
                test_loss = loss(test_y_hat, test_labels)
                test_loss_arr[epoch] = test_loss

                # get the test predictions
                test_prob = torch.sigmoid(test_y_hat)
                test_pred = torch.round(test_prob)
                test_auc = round(roc_auc_score(test_labels.detach().numpy(), test_prob.detach().numpy()), 3)
                test_auc_arr[epoch] = test_auc

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


    print("Train loss \t Valid loss \t Test loss \t Train AUC \t Valid AUC \t Test AUC")
    print(round(repeated_train_loss_arr.mean(),3), round(repeated_valid_loss_arr.mean(),3), round(repeated_test_loss_arr.mean(),3), round(repeated_train_auc_arr.mean(),3), round(repeated_valid_auc_arr.mean(),3), round(repeated_test_auc_arr.mean(),3))






