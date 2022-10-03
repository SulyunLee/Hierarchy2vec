import pandas as pd
import numpy as np


if __name__ == "__main__":
    #################################################################
    # Load datasets
    team_labels_filename = "data/team_labels.csv"
    playoff_filename = "data/Playoff.csv"

    team_labels_df = pd.read_csv(team_labels_filename)
    playoff_df = pd.read_csv(playoff_filename)
    #################################################################

    playoff_label = np.zeros((team_labels_df.shape[0], 1))
    for idx, row in team_labels_df.iterrows():
        playoff_record = playoff_df[(playoff_df.Year == row.Year) & \
                                    (playoff_df.Team == row.Team)]
        if playoff_record.shape[0] > 0:
            playoff_label[idx] = 1

        idx += 1

    team_labels_df = team_labels_df.assign(playoff=playoff_label)
    team_labels_df.to_csv(team_labels_filename, index=False, encoding="utf-8-sig")

