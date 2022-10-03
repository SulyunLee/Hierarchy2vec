
import pandas as pd
import numpy as np
from process_NFL_position_names import assign_unique_position_apply
from POSITION_ASSIGNMENT import *

def clean_NFL_data(df, start_year, end_year):
    '''
    This function cleans NFL coach data to use in the modelings.
    The function performs the following:
    1. Only extract coaching records between <start_year> and <end_year> years
    2. Remove interim head coaches
    3. Remove coaches with unqualified position titles (-1 values)

    '''
    # extract specific seasons
    df = df[(df.Year >= start_year) & (df.Year <= end_year)]
    df.reset_index(drop=True, inplace=True)

    # exclude interim head coaches
    df = df[df.final_position != "iHC"]
    df.reset_index(drop=True, inplace=True)

    # exclude coaches with no proper positions
    df = df[(df.final_position != -1) & (df.final_hier_num != -1)]
    df.reset_index(drop=True, inplace=True)

    print("The number of NFL records: {}".format(df.shape[0]))
    print("The number of NFL coaches: {}".format(df.Name.unique().shape[0]))

    return df


def clean_history_data(history_df, NFL_coaches_arr):
    '''
    This function cleans NFL coaches' historical coaching records data.
    This function performs the following:
    1. Extract coaching records of NFL coaches in the data (Name given in the array).
    2. Only include qualified position coaching titles
    '''

    # extract coaching records of NFL coaches in the data
    history_df = history_df[history_df.Name.isin(NFL_coaches_arr)]
    history_df.reset_index(drop=True, inplace=True)

    print("The number of history records: {}".format(history_df.shape[0]))
    print("The number of coaches in history records: {}".format(history_df.Name.unique().shape[0]))

    # Remove records with missing values
    history_df.dropna(inplace=True)

    # clean position name titles
    position_lists = history_df.Position.str.replace('coach', '')
    position_lists = position_lists.str.replace('Coach', '')
    position_lists = position_lists.str.replace('coordinatorinator', 'coordinator')
    position_lists = position_lists.str.split("[/;-]| &")
    position_lists = position_lists.apply(lambda x: [e.lower().strip() for e in x])
    history_df = history_df.assign(Position_list=position_lists)

    # Match the position titles to the position IDs and hierarchy number
    assigned_unique_positions = history_df.apply(assign_unique_position_apply,\
            args=[simple_position_id_mapping], axis=1)
    history_df['final_position'], history_df['final_position_spec'], history_df['final_hier_num'] = zip(*assigned_unique_positions)

    # If more than one qualified position exists, select the position in the highest hierarchy
    NFL_coach_history_qualified = history_df[history_df.final_position != -1]
    NFL_coach_history_qualified.reset_index(drop=True, inplace=True)
    print("The number of qualified NFL coaches' history records: {}".format(NFL_coach_history_qualified.shape[0]))

    return NFL_coach_history_qualified





