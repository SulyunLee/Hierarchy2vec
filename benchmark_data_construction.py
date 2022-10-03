'''
This script generates dataset and trains model for benchmark.
Benchmark: Fast, A., & Jensen, D. (2006). The NFL coaching network: analysis of the social network among professional football coaches.
'''
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from data_cleaning_func import *

def first_year_HC_feature(row, history_df, NFL_record_df):
    year = row.Year
    team = row.Team
    hc = hc_dict[(year, team)]
    # If the current year is 2002, look at the 2001 record
    prev_hc_record = NFL_record_df[(NFL_record_df.Name == hc) & (NFL_record_df.Team == team) & (NFL_record_df.Year == year-1)]
    if prev_hc_record.shape[0] == 0:
        return 1
    else:
        return 0

def find_playoff_history(prev_teams, team_record_df, total_experience, playoff_count):
    # iterrate over previous records between 1999 and 2001
    for idx, record in prev_teams.iterrows():
        t = record.ServingTeam
        min_year = int(record.StartYear)
        max_year = int(record.EndYear)
        for y in range(max(1999,min_year), min(2001, max_year)+1):
            total_experience += 1
            # search if the team and season exists in playoff data
            playoff_team = playoff_df[(playoff_df.Team.str.lower().str.contains(t)) & (playoff_df.Year == y)]
            if playoff_team.shape[0] > 0:
                playoff_count += 1

    return total_experience, playoff_count

def find_win_perc(prev_teams, team_record_df):
    # iterate over previous records from 2002
    total_experience = 0
    failure_count = 0
    for idx, record in prev_teams.iterrows():
        t = record.Team.replace(" (NFL)", "")
        y = record.Year
        total_experience += 1
        # TODO: edit codes 
        record = team_labels_df[(team_record_df.Team == t) & (team_record_df.Year == y)]
        if record.iloc[0].Win_Percentage < 0.5:
            failure_count += 1

    return total_experience, failure_count


def hc_failure_feature(row, NFL_coach_instances):
    year = row.Year
    team = row.Team
    hc = hc_dict[(year,team)]

    total_experience = 0
    # previous teams of HC from 2002 up to previous year
    prev_teams = NFL_coach_instances[(NFL_coach_instances.Name == hc) & (NFL_coach_instances.Year < year)]
    prev_teams.reset_index(drop=True, inplace=True)
    if prev_teams.shape[0] > 0:
        total_experience, failure_count = find_win_perc(prev_teams, team_labels_df)

    if total_experience == 0:
        failure = 0
    else:
        failure = round(failure_count / total_experience,2)

    return failure

def coord_failure_feature(row, NFL_coach_instances):
    year = row.Year
    team = row.Team
    coords = coord_dict[(year,team)]

    # previous teams of Coordinators between 1999 and 2001
    performance_list = []
    for coord in coords:
        total_experience = 0
        # previous teams of HC from 2002 up to previous year
        prev_teams = NFL_coach_instances[(NFL_coach_instances.Name == coord) & (NFL_coach_instances.Year < year)]
        prev_teams.reset_index(drop=True, inplace=True)
        if prev_teams.shape[0] > 0:
            total_experience, failure_count = find_win_perc(prev_teams, team_labels_df)

        if total_experience == 0:
            performance = 0
        else:
            performance = round(failure_count / total_experience,2)

        # append performance to the list
        performance_list.append(performance)

    if performance_list == []:
        return 0
    else:
        return np.mean(performance_list)

def hc_champ_mentors(row, NFL_coach_instances):
    hc = hc_dict[(row.Year, row.Team)]
    # select previous seasons of the head coach
    prev_teams = NFL_coach_instances[(NFL_coach_instances.Name == hc) & (NFL_coach_instances.Year < row.Year)]
    prev_teams.reset_index(drop=True, inplace=True)

    if prev_teams.shape[0] != 0:
        # select seasons where HC worked as either coordinators or position coaches
        prev_teams_notHC = prev_teams[prev_teams.final_hier_num != 1]
        prev_teams_notHC.reset_index(drop=True, inplace=True)

        # get the mentors (HC) while working in the previous teams
        mentors = set()
        for season in prev_teams_notHC[["Year","Team"]].values:
            year = season[0]
            team = season[1]
            mentor = hc_dict[(year, team)]
            mentors.add(mentor)

        if len(mentors) != 0:
            champ_mentors = 0
            for m in mentors:
                if m in champ_set:
                    champ_mentors += 1
            prop_champ_mentors = round(champ_mentors / len(mentors), 2)
        else: prop_champ_mentors = 0

    else:
        prop_champ_mentors = 0

    return prop_champ_mentors

def num_more1yr_hc_teams(row, history_df, NFL_coach_instances): 
    hc = hc_dict[(row.Year, row.Team)]

    count = 0
    # select previous seasons between 1999 and 2001 
    prev_teams = history_df[(history_df.Name == hc) & (history_df.StartYear <= 1999) & (history_df.EndYear >= 2001) & \
                            (history_df.NFL == 1) & (history_df.ServingTeam != row.Team.replace(" (NFL)","").lower())]
    prev_teams.reset_index(drop=True, inplace=True)
    if prev_teams.shape[0] != 0:
        for idx, record in prev_teams.iterrows():
            if record.EndYear - record.StartYear > 0:
                count += 1

    # select previous seasons of the head coach
    prev_teams = NFL_coach_instances[(NFL_coach_instances.Name == hc) & (NFL_coach_instances.Year < row.Year) & (NFL_coach_instances.Team != row.Team)]
    prev_teams.reset_index(drop=True, inplace=True)

    if prev_teams.shape[0] != 0:
        # iterate through previous teams and find out if the coach worked in the previous years
        for idx, record in prev_teams.iterrows():
            prev_year = record.Year
            prev_team = record.Team

            oneyr_before = NFL_coach_instances[(NFL_coach_instances.Name == hc) & (NFL_coach_instances.Team == prev_team) & \
                                                (NFL_coach_instances.Year == prev_year-1)]
            if oneyr_before.shape[0] != 0:
                count += 1

    return count

def generate_labels(row, label_df, label_name):
    labels = label_df[(label_df.Year == row.Year) & (label_df.Team == row.Team.replace(" (NFL)", ""))]
    if int(labels[label_name]) == 1:
        return 1
    else:
        return 0

if __name__ == "__main__":
    """
    This script generates a dataset for a benchmark model, Fast and Jensen 
    """

    #################################################################
    NFL_coach_record_filename = "data/NFL_Coach_Data_final_position.csv"
    history_record_filename = "data/all_coach_records_cleaned.csv"
    playoff_filename = "data/Playoff.csv"
    team_labels_filename = "data/team_labels.csv"

    NFL_record_df = pd.read_csv(NFL_coach_record_filename)
    history_df = pd.read_csv(history_record_filename)
    playoff_df = pd.read_csv(playoff_filename)
    team_labels_df = pd.read_csv(team_labels_filename)
    #################################################################

    # clean NFL coach record data
    NFL_coach_instances = clean_NFL_data(NFL_record_df, 2002, 2019)

    # clean NFL coaches' history record data
    nfl_coaches = NFL_record_df.Name.unique()
    history_df = clean_history_data(history_df, nfl_coaches)

    tqdm.pandas()

    # Initialize dataset - Team and Year pairs
    data = NFL_coach_instances[["Year","Team"]].drop_duplicates()
    data.reset_index(drop=True, inplace=True)


    # Generate dictionaries that indicate HCs and Coordinators
    hc_dict = {}
    coord_dict = {}
    for idx, row in data.iterrows():
        year = row.Year
        team = row.Team
        team_members = NFL_coach_instances[(NFL_coach_instances.Year == year) & \
                                        (NFL_coach_instances.Team == team)]
        hc = team_members[team_members.final_hier_num == 1].Name.values[0]
        coords = team_members[team_members.final_hier_num == 2].Name.values

        hc_dict[(year, team)] = hc
        coord_dict[(year, team)] = coords

    # Generate a dictionary that stores championship coaches per year
    champ_set = set()
    champ_record = playoff_df[(playoff_df.Year > 2001) & ((playoff_df.StageAchieved == "Conference Championship") |\
                            (playoff_df.StageAchieved == "Super Bowl"))]

    for idx, row in champ_record.iterrows():
        team = row.Team + " (NFL)"
        year = row.Year
        champ_set.add(hc_dict[(year, team)])

    # Generate feature sets    

    # 1. First-year HC?
    print("Feature 1: Whether the HC is the first-year coach")
    first_year_HC = data.progress_apply(first_year_HC_feature,\
                    args=[history_df, NFL_coach_instances], axis=1)
    data = data.assign(FirstYearHC = first_year_HC)

    # 2. Proportion of HC's previous seasons that made playoffs
    print("Feature 2: Proportion of HC's previous seasons with winning percentage less than 50%")
    hc_failures = data.progress_apply(hc_failure_feature,\
                            args=[NFL_coach_instances], axis=1)
    data = data.assign(HCFailure = hc_failures)

    # 3. Proportion of Coordinators' previous seasons that made playoffs
    print("Feature 3: Proportion of Coordinators' previous seasons with winning percentage less than 50%")
    coord_failures = data.progress_apply(coord_failure_feature,\
                            args=[NFL_coach_instances], axis=1)
    data = data.assign(CoordFailure = coord_failures)

    # 4. Proportion of HC's mentors who have won championship
    print("Feature 4: Proportion of HC's mentors who have won championships")
    hc_champ_mentors = data.progress_apply(hc_champ_mentors,\
                        args=[NFL_coach_instances], axis=1)
    data = data.assign(HCChampMentors = hc_champ_mentors)

    # 5. Number of HC's previous teams that worked for more than 1 year
    print("Feature 5: Number of HC's previous teams that worked for more than 1 year")
    num_more1yr_hc_teams = data.progress_apply(num_more1yr_hc_teams,\
                            args=[history_df, NFL_coach_instances], axis=1)
    data = data.assign(NumMore1yrHCTeams = num_more1yr_hc_teams)

    # Generate target variable
    print("Target generating...")
    target = data.progress_apply(generate_labels,\
                                args=[team_labels_df, "failure"], axis=1)
    data = data.assign(failure = target)

    data.to_csv("data/benchmark_data.csv", index=False, encoding="utf-8-sig")




















