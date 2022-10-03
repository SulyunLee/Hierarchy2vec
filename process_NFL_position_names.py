'''
Author: Sulyun Lee
This script processes the cleaned data of "2002-2019_NFL_Coach_Data.csv" in the
format for constructing network of NFL teamts.

'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from POSITION_ASSIGNMENT import *
import argparse

def assign_unique_position_apply(row, position_id_mapping):
    position_list = row.Position_list

    # there is only one position for the coach
    if len(position_list) == 1:
        try:
            # search the position ID and hierarchy number from the dictionary
            position_id, position_id_spec, hier_num = position_id_mapping[position_list[0]]
        except:
            # if not found, this coach will be excluded in the graph
            position_id = position_id_spec = hier_num = -1
    # multiple positions for one coach
    else:
        # if "head coach" in position_list:
            # position_list.remove("head coach")
        ids = []
        spec_ids = []
        hier_nums = []
        # iterate over each position and find the position ID and hierarchy number
        for position in position_list:
            try:
                position_id, position_id_spec, hier_num = position_id_mapping[position]
                ids.append(position_id)
                spec_ids.append(position_id_spec)
                hier_nums.append(hier_num)
            except:
                continue

        if len(ids) == 0:
            position_id = position_id_spec = hier_num = -1
        elif len(ids) == 1:
            position_id = ids[0]
            position_id_spec = spec_ids[0]
            hier_num = hier_nums[0]
        else:
            # assign the position in the higher hierarchy as the final position
            high_position_idx = hier_nums.index(min(hier_nums))
            position_id = ids[high_position_idx]
            position_id_spec = spec_ids[high_position_idx]
            hier_num = hier_nums[high_position_idx]

    return position_id, position_id_spec, hier_num

def assign_position_apply(row, position_id_mapping):
    split_position = row.Split_position
    try:
        position_id, position_id_spec, hier_num = position_id_mapping[split_position]
    except: position_id = position_id_spec = hier_num = -1

    return position_id, position_id_spec, hier_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ################################################################
    # Load datasets
    NFL_coach_record_filename = "../datasets/NFL_Coach_Data.csv"
    #################################################################

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)

    # Generate the list of positions for each coach
    position_lists = NFL_record_df.Position.str.split("[/;-]")
    position_lists = position_lists.apply(lambda x: [e.lower().strip() for e in x])
    NFL_record_df = NFL_record_df.assign(Position_list=position_lists)

    # Split rows if a coach has multiple titles
    total_rows = position_lists.apply(len).sum()
    name_arr = np.zeros((total_rows)).astype(str)
    year_arr = np.zeros((total_rows)).astype(int)
    team_arr = np.zeros((total_rows)).astype(str)
    position_arr = np.zeros((total_rows)).astype(str)
    i = 0
    for idx, row in NFL_record_df.iterrows():
        name = row.Name
        year = row.Year
        team = row.Team
        position_list = row.Position_list
        for position in position_list:
            name_arr[i] = name
            year_arr[i] = year
            team_arr[i] = team
            position_arr[i] = position
            i += 1

    NFL_record_split_df = pd.DataFrame({"Name":name_arr,
                                        "Year": year_arr, "Team":team_arr,
                                        "Split_position": position_arr})

    # iterate over each row in NFL record
    # Match the position name to the position IDs and hierarchy number.
    assigned_unique_positions = NFL_record_split_df.apply(assign_position_apply, \
            args=[simple_position_id_mapping], axis=1)

    NFL_record_split_df['final_position'], NFL_record_split_df['final_position_spec'], NFL_record_split_df['final_hier_num'] = zip(*assigned_unique_positions)

    # drop duplicates
    NFL_record_split_df = NFL_record_split_df.drop_duplicates(subset=["Name", "Year", "Team", "final_position", "final_hier_num"])

    # write to a separate csv file
    NFL_record_split_df.to_csv("../datasets/NFL_Coach_Data_final_position_expanded.csv", \
            index=False, encoding="utf-8-sig")


    # iterate over each row in NFL record
    # Match the position name to the position IDS and hierarchy number.
    # If more than one position exists, select the position in the higher hierarchy.
    assigned_unique_positions = NFL_record_df.apply(assign_unique_position_apply, \
            args=[simple_position_id_mapping], axis=1)

    NFL_record_df['final_position'], NFL_record_df['final_position_spec'], NFL_record_df['final_hier_num'] = zip(*assigned_unique_positions)

    # drop duplicates
    NFL_record_df = NFL_record_df.drop_duplicates(subset=["Name", "Year", "Team", "final_position", "final_hier_num"])

    # write to a separate csv file
    NFL_record_df.to_csv("../datasets/NFL_Coach_Data_final_position.csv", \
            index=False, encoding="utf-8-sig")

    ########################################################
    # Following codes are to explore the number of teams with missing positions
    # when only 8 position coach titles are considered
    ########################################################
    ### Include only 2002-2019 seasons
    NFL_coach_instances = NFL_record_split_df[NFL_record_split_df.final_position != "iHC"]
    NFL_coach_instances.reset_index(drop=True, inplace=True)

    # exclude coaches with no proper positions
    NFL_coach_instances = NFL_coach_instances[(NFL_coach_instances.final_position != -1) & (NFL_coach_instances.final_hier_num != -1)]
    NFL_coach_instances.reset_index(drop=True, inplace=True)

    NFL_coach_instances = NFL_coach_instances[(NFL_coach_instances.Year >= 2002) & (NFL_coach_instances.Year <= 2019)]
    NFL_coach_instances.reset_index(drop=True, inplace=True)

    # identify teams with missing position coach titles
    position_coaches = NFL_coach_instances[NFL_coach_instances.final_hier_num == 3]
    position_coaches.reset_index(drop=True, inplace=True)
    years = position_coaches.Year.unique()
    teams = position_coaches.Team.unique()
    missing_dict = {}
    l = []
    for y in years:
        for t in teams:
            season = position_coaches[(position_coaches.Year==y)&(position_coaches.Team==t)]
            missing = set(["QB","RB","OL","WR","TE","LB","DL","Sec"]) - set(season.final_position_spec.values)
            if len(missing) in missing_dict:
                missing_dict[len(missing)]+= 1
            else:
                missing_dict[len(missing)] = 1

            if len(missing) == 1:
                l.append(list(missing)[0])





    

