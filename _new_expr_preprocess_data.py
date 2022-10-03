'''
This script preprocesses given dataset for running experiments.
'''
import pandas as pd
import numpy as np
import networkx as nx

if __name__ == "__main__":

    #################################################################
    # Load datasets
    print("Reading datasets...")
    employee_info_fname = "data/Company_experiment/employee_personal_info.csv"
    dept_affil_fname = "data/Company_experiment/department_affiliation_combined.csv"
    message_collab_fname = "data/Company_experiment/combined_message_collab.csv"


    employee_info_df = pd.read_csv(employee_info_fname)
    dept_affil_df = pd.read_csv(dept_affil_fname)
    message_collab_df = pd.read_csv(message_collab_fname)
    #################################################################

    # extract employees who are in all three datasets
    emp1 = set(employee_info_df.employee_id.unique())
    emp2 = set(dept_affil_df.employee_id.unique()).union(set(dept_affil_df.leader_id.unique()))
    emp3 = set(message_collab_df.emp_a.unique()).union(set(message_collab_df.emp_b.unique()))

    common_emp = emp1.intersection(emp2, emp3)
    print("Final number of employees: {}".format(len(common_emp)))

    # select subset of datasets
    employee_info_df2 = employee_info_df[employee_info_df.employee_id.isin(common_emp)]
    employee_info_df2.reset_index(drop=True, inplace=True)

    dept_affil_df2 = dept_affil_df[(dept_affil_df.employee_id.isin(common_emp)) & \
                                    (dept_affil_df.leader_id.isin(common_emp))]
    dept_affil_df2.reset_index(drop=True, inplace=True)

    message_collab_df2 = message_collab_df[(message_collab_df.emp_a.isin(common_emp)) & \
                                    (message_collab_df.emp_b.isin(common_emp))]
    message_collab_df2.reset_index(drop=True, inplace=True)

    # form teams based on leader information
    # only keep three levels of hierarchy
    # weakly connected component?
    edgelist_lst = []
    for date in dept_affil_df2.dt.unique():
        time_dept = dept_affil_df2[dept_affil_df2.dt == date].reset_index(drop=True)
        team_cnt = 0
        nodes_list = []
        for dept in time_dept.deptid_4.unique():
            team = time_dept[time_dept.deptid_4 == dept]
            team_member = team.employee_id.unique()
            team = team[team.leader_id.isin(team_member)].reset_index(drop=True)

            # generate graph based on leader info.
            # Directed graph from leader to its subordinate
            g = nx.from_pandas_edgelist(team, source="leader_id", target="employee_id",\
                                        create_using=nx.DiGraph)
            
            roots = [n for n,d in g.in_degree() if d == 0] # leaders of teams
            if len(roots) == 0:
                break
            else:
                # extract root with the maximum depth
                root_depths = [max(nx.shortest_path_length(g, source=root).values()) for root in roots]
                max_root_depth_idx = root_depths.index(max(root_depths))

                    
                # truncate some branches that are over length 2 (3 hierarchies)
                if root_depths[max_root_depth_idx] > 1:
                    root = roots[max_root_depth_idx]
                    spl = nx.shortest_path_length(g, source=root)
                    # only keep the top 3 levels
                    include_nodes = [n for n in spl.keys() if spl[n] <= 2]
                    new_g = g.subgraph(include_nodes).copy()
                
                    edgelist = nx.to_pandas_edgelist(new_g)
                    edgelist.loc[:, 'dt'] = date
                    edgelist_lst.append(edgelist)

                    team_cnt += 1

        print("** Date {}".format(date))
        print("Number of teams: {}".format(team_cnt))

    # hierarchical team edgelist
    hier_team_edgelist = pd.concat(edgelist_lst, axis=0)
    hier_team_edgelist.reset_index(drop=True, inplace=True)
    hier_team_edgelist.to_csv("data/Company_experiment/hier_team_edgelist_final.csv", index=False)

    # department affiliation information
    dept_affil_df2.to_csv("data/Company_experiment/dept_affil_final.csv", index=False)
    # message collaboration network 
    message_collab_df2.to_csv("data/Company_experiment/message_collab_final.csv", index=False)

    # employee personal information
    employee_info_df2 = employee_info_df2[["employee_id", "gender", "yob", "highest_degree", "hire_date"]]
    employee_info_df2.to_csv("data/Company_experiment/employee_info_final.csv", index=False)






