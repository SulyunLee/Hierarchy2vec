import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, loss, model):
        score = loss
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            # print("Early stopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.counter, self.early_stop

def split_train_valid_test(df, start_year, train_split_year, valid_split_year, end_year):
    train = df[(df.Year >= start_year) & (df.Year <= train_split_year)]
    valid = df[(df.Year > train_split_year) & (df.Year <= valid_split_year)]
    test = df[(df.Year > valid_split_year) & (df.Year <= end_year)]

    train.reset_index(drop=True, inplace=True)
    valid.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train,valid,test

def normalize(train, valid, test):
    '''
    This function normalizes input train, validation, and test datasets before feeding
    into a deep learning model.
    Uses mean zero normalization function.
    '''

    train = np.array(train) 
    valid = np.array(valid) 
    test = np.array(test) 

    means = train.mean(axis=0)
    stds = train.std(axis=0)

    normalized_train = (train - means) / stds
    normalized_valid = (valid - means) / stds
    normalized_test = (test - means) / stds

    return torch.Tensor(normalized_train), torch.Tensor(normalized_valid), torch.Tensor(normalized_test)
    
def get_complete_teams(df, position_title_list):
    position_coaches = df[df.final_hier_num == 3]
    position_coaches.reset_index(drop=True, inplace=True)

    years = position_coaches.Year.unique()
    teams = position_coaches.Team.unique()

    year_team_pairs = []
    for y in years:
        for t in teams:
            season = position_coaches[(position_coaches.Year==y) & (position_coaches.Team==t)]
            missing = set(position_title_list) - set(season.final_position_spec.values)
            if len(missing) == 0:
                year_team_pairs.append((y, t))

    return year_team_pairs
