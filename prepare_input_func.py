
import pandas as pd
import numpy as np
from collections import namedtuple

def average_node_features(team, year, df, feature_names):
    '''
    This function averages all team coaches' features
    Returns the averaged team coaches' features
    '''
    
    # extract coaches of the given year and team
    coaches = df[(df.Team == team) & (df.Year == year)]
    coach_features = coaches[feature_names] # only extract features

    # average coaches' features
    avg_coach_features = np.array(coach_features.mean())

    return avg_coach_features

def average_node_features_by_group(team, year, df, feature_names, group):
    '''
    This function averages team coaches' features in each group.
    Returns the averaged team coaches' features in each group in the order of group names
    '''

    coaches = df[(df.Team == team) & (df.Year == year)]
    
    group_coaches = coaches[coaches.final_position == group]

    if group_coaches.shape[0] != 0:
        avg_group_coaches = np.array(group_coaches[feature_names].mean())
    else:
        avg_group_coaches = np.zeros((1,len(feature_names)))

    return avg_group_coaches


def get_team_features(team, year, df, feature_names):
    '''
    This function returns the team features of the given team in given year
    '''

    team_features = df[(df.Team == team) & (df.Year == year)][feature_names]
    team_feature_arr = np.array(team_features)

    return team_feature_arr

def get_team_label(team, year, df, label_name):
    '''
    This function returns the label of the given team in given year
    '''
    label = df[(df.Team == team.replace('(NFL)', '').strip()) & (df.Year == year)][label_name]

    return label
