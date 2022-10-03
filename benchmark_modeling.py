
import shelve
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import *
from models import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

if __name__ == "__main__":
    """ 
    This script predicts team performance using a benchmark model, Fast and Jensen 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_year', '--start_year', type=int, default = 2002, help="Starting year of the dataset")
    parser.add_argument('-end_year', '--end_year', type=int, default = 2019, help="Ending year of the dataset")
    parser.add_argument('-train_split_year', '--train_split_year', type=int, help="Maximum year for training set")
    parser.add_argument('-valid_split_year', '--valid_split_year', type=int, help="Maximum year for validation set")

    args = parser.parse_args()
    start_year = args.start_year
    end_year = args.end_year
    train_split_year = args.train_split_year
    valid_split_year = args.valid_split_year

    #################################################################
    # Load datasets
    data_filename = "data/NFL_experiment/benchmark_data.csv"

    data = pd.read_csv(data_filename)
    #################################################################

    ### Split data into train, validation, and test sets
    train_record, valid_record, test_record = \
            split_train_valid_test(data, start_year, train_split_year, valid_split_year, end_year)

    ### Define features
    features = ['FirstYearHC', 'HCFailure', 'CoordFailure', \
                   'HCChampMentors', 'NumMore1yrHCTeams']
    numeric_features = ['FirstYearHC', 'HCFailure', 'CoordFailure', 'NumMore1yrHCTeams']
    label = "failure"

    train_x, valid_x, test_x = train_record[features], valid_record[features], test_record[features]
    train_y, valid_y, test_y = train_record[label], valid_record[label], test_record[label]

    repeated_train_auc_arr = np.zeros((10))
    repeated_valid_auc_arr = np.zeros((10))
    repeated_test_auc_arr = np.zeros((10))
    repeated_train_acc_arr = np.zeros((10))
    repeated_valid_acc_arr = np.zeros((10))
    repeated_test_acc_arr = np.zeros((10))

    fpr_dict = dict()
    tpr_dict = dict()

    max_depth = range(1,8)
    for seed in range(0, 10):

        train_auc_arr = np.zeros((len(max_depth))) 
        valid_auc_arr = np.zeros((len(max_depth))) 
        test_auc_arr = np.zeros((len(max_depth)))
        train_acc_arr = np.zeros((len(max_depth))) 
        valid_acc_arr = np.zeros((len(max_depth))) 
        test_acc_arr = np.zeros((len(max_depth)))
        test_fpr_arr = dict()
        test_tpr_arr = dict()
        for depth in max_depth: 
            clf = DecisionTreeClassifier(random_state=seed, max_depth=depth, splitter="random", max_features="auto")
            
            clf.fit(train_x, train_y)
            train_prob = clf.predict_proba(train_x)
            train_pred = clf.predict(train_x)
            train_auc_arr[depth-1] = roc_auc_score(train_y, train_prob[:,1])
            train_acc_arr[depth-1] = accuracy_score(train_y, train_pred)

            # test on validation set
            valid_prob = clf.predict_proba(valid_x)
            valid_pred = clf.predict(valid_x)
            valid_auc_arr[depth-1] = roc_auc_score(valid_y, valid_prob[:,1]) 
            valid_acc_arr[depth-1] = accuracy_score(valid_y, valid_pred)

            # test set
            test_prob = clf.predict_proba(test_x)
            test_pred = clf.predict(test_x)
            fpr, tpr, _ = roc_curve(test_y, test_prob[:,1], drop_intermediate=False)
            test_auc_arr[depth-1] = roc_auc_score(test_y, test_prob[:,1])
            test_acc_arr[depth-1] = accuracy_score(test_y, test_pred)
            test_fpr_arr[depth-1] = fpr
            test_tpr_arr[depth-1] = tpr

        # select the best parameter
        best_idx = np.where(valid_auc_arr == max(valid_auc_arr))[0][0]


        repeated_train_auc_arr[seed] = train_auc_arr[best_idx]
        repeated_valid_auc_arr[seed] = valid_auc_arr[best_idx]
        repeated_test_auc_arr[seed] = test_auc_arr[best_idx]
        repeated_train_acc_arr[seed] = train_acc_arr[best_idx]
        repeated_valid_acc_arr[seed] = valid_acc_arr[best_idx]
        repeated_test_acc_arr[seed] = test_acc_arr[best_idx]
        fpr_dict[seed] = test_fpr_arr[best_idx]
        tpr_dict[seed] = test_tpr_arr[best_idx]
    
    print("Train AUC \t Valid AUC \t Test AUC")
    print(round(repeated_train_auc_arr.mean(), 3), round(repeated_valid_auc_arr.mean(),3), round(repeated_test_auc_arr.mean(),3))

    print("Train Accuracy \t Valid Accuracy \t Test Accuracy")
    print(round(repeated_train_acc_arr.mean(), 3), round(repeated_valid_acc_arr.mean(),3), round(repeated_test_acc_arr.mean(),3))


    # plot the ROC curve
    i = 8
    roc_auc = auc(fpr_dict[i], tpr_dict[i])

    d = shelve.open("results/test_fpr_tpr")
    d['fastandjensen'] = {'fpr': fpr_dict[i], \
                        'tpr': tpr_dict[i], \
                        'auc': roc_auc}
    d.close()
