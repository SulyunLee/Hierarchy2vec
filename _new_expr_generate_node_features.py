'''
This script generates node features for each employee 
'''
import pandas as pd
import numpy as np
import networkx as nx
import argparse
from tqdm import tqdm
from datetime import datetime

def get_tenure(row, personal_info_df):
    employee = row.Employee
    current_date = row.Date

    date = datetime.strptime(current_date, "%Y-%m-%d")
    hire_date = personal_info_df.loc[personal_info_df.employee_id == employee, "hire_date"].iloc[0]
    hire_date = datetime.strptime(hire_date, "%m/%d/%y")

    # take difference between current date and hire date
    delta = date - hire_date
    # return the tenure in years
    return round(delta.days / 365, 2)

def get_education_level(row, personal_info_df):
    employee = row.Employee
    highest_degree = personal_info_df.loc[personal_info_df.employee_id == employee, "highest_degree"].iloc[0]
    # greater highest degree -> higher value returned
    # Elementary school or below has value of 1 (lowest)
    # PhD has the value of 8 (highest)
    if highest_degree is np.nan:
        return 0
    else:
        value = 10 - int(highest_degree.split(' ')[0])
        return value

def get_job_rank(row, dept_affil_df):
    employee = row.Employee
    date = row.Date

    # get the paygrade for the employee
    dept_extract = dept_affil_df[(dept_affil_df.employee_id == employee) &\
                                (dept_affil_df.dt == date)]
    paygrade = dept_extract.paygrade.iloc[0]

    return paygrade
                                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_size', '--emb_size', default=32, type=int, help="Node embedding size")
    parser.add_argument('-bias', '--bias', default="unbiased", type=str, help="Unbiased or biased walk while learning node embeddings")
    parser.add_argument('-w', '--w', default=3, type=int, help="window size")

    args = parser.parse_args()
    emb_size = args.emb_size
    bias = args.bias
    w = args.w

    #################################################################
    # Load datasets
    personal_info_filename = "data/Company_experiment/employee_info_final.csv"
    hier_team_edgelist_fname = "data/Company_experiment/hier_team_edgelist_final.csv"
    dept_affil_fname = "data/Company_experiment/dept_affil_final.csv"
    paygrade_rankings_fname = "data/Company_experiment/paygrade_rankings.csv"
    node_embedding_fname = "data/Company_experiment/embeddings/cumulative_collab_G_node_embedding_size{}_{}_w{}_df.csv".format(emb_size, bias, w)

    personal_info_df = pd.read_csv(personal_info_filename)
    hier_team_edgelist = pd.read_csv(hier_team_edgelist_fname)
    dept_affil_df = pd.read_csv(dept_affil_fname)
    paygrade_rankings_df = pd.read_csv(paygrade_rankings_fname)
    node_embedding_df = pd.read_csv(node_embedding_fname)
    #################################################################

    # generate teams for each timestamp
    team_id = 0
    df_list = []
    for date in hier_team_edgelist.dt.unique()[1:]:
        tmp = hier_team_edgelist[hier_team_edgelist.dt == date].reset_index(drop=True)
        g = nx.from_pandas_edgelist(tmp)
        cc = nx.connected_components(g)
        num_cc = nx.number_connected_components(g)
        # iterate over each team
        for component in cc:
            sub_g = g.subgraph(component)
            df = pd.DataFrame({'Employee':sub_g.nodes(), 'Date':date, 'TeamID':team_id})
            df_list.append(df)
            team_id += 1

    data = pd.concat(df_list, axis=0).reset_index(drop=True)

    tqdm.pandas()

    print("* Individual features")
    # tenure - how long with the company
    print("1. Tenure - how long with the company")
    tenure = data.progress_apply(get_tenure, args = [personal_info_df], axis=1)
    # replace data with zero where hire date is after the current date.
    tenure[tenure < 0] = 0
    data = data.assign(Tenure = tenure)

    # education level
    print("2. Education level")
    education_level = data.progress_apply(get_education_level, args=[personal_info_df], axis=1)
    # higher values if education level is higher
    data = data.assign(EduLevel = education_level)

    # job rank (paygrade)
    print("Job rank (paygrade)")
    paygrade_m_map = list(zip(paygrade_rankings_df.paygrade_M, paygrade_rankings_df.ranking))
    paygrade_p_map = list(zip(paygrade_rankings_df.paygrade_P, paygrade_rankings_df.ranking))
    paygrade_map = dict(paygrade_m_map + paygrade_p_map)

    paygrade = data.progress_apply(get_job_rank, args=[dept_affil_df], axis=1)
    paygrade_rank = paygrade.map(paygrade_map)
    # higher values if paygrade rank is higher
    data = data.assign(JobRank = 20-paygrade_rank)

    
    print('* Collaboration features')
    node_emb_columns = node_embedding_df.columns[node_embedding_df.columns.str.contains("cumul_emb")].tolist()
    employee_collab_features = np.zeros((data.shape[0], len(node_emb_columns)))

    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        date = row.Date
        employee = row.Employee

        emb = node_embedding_df[(node_embedding_df.Node == employee) & \
                            (node_embedding_df.Week < date)]
        if emb.shape[0] != 0:
            emb = emb.sort_values('Week', ascending=False).iloc[0]
            emb = np.array(emb[node_emb_columns])

            employee_collab_features[idx,:] = emb

    data = pd.concat([data, pd.DataFrame(employee_collab_features,         
                                            index=data.index, columns=node_emb_columns)], axis=1)
    data.to_csv("data/Company_experiment/Company_Employee_Data_with_features_emb{}_{}_w{}.csv".format(emb_size, bias, w),\
        index=False)


    
