import pandas as pd
import numpy as np
import networkx as nx

if __name__ == "__main__":
    # dept_affil_fname = "data/Company_experiment/202104-202106_department affiliation.xlsx"
    # dept_affil_df = pd.read_excel(dept_affil_fname)

    # first_month_employees = dept_affil_df[dept_affil_df.dt == "2021-04-01"].employee_id.unique()

    # print("2021-05-01")
    # second_month_employees = dept_affil_df[dept_affil_df.dt == "2021-05-01"].employee_id.unique()
    # newly_emerged = [emp for emp in second_month_employees if emp not in first_month_employees]
    # print("New: {}".format(len(newly_emerged)))
    # retired = [emp for emp in first_month_employees if emp not in second_month_employees]
    # print("Quit:{}".format(len(retired)))

    # print("2021-06-01")
    # third_month_employees = dept_affil_df[dept_affil_df.dt == "2021-06-01"].employee_id.unique()
    # newly_emerged = [emp for emp in third_month_employees if emp not in second_month_employees]
    # print("New: {}".format(len(newly_emerged)))
    # retired = [emp for emp in second_month_employees if emp not in third_month_employees]
    # print("Quit:{}".format(len(retired)))

    # import employee personal info dataset
    employee_info_filename = "data/Company_experiment/employee_personal_info.csv"
    employee_info_df = pd.read_csv(employee_info_filename)

    # import collaboration dataset
    # collaboration_fname = "data/Company_experiment/combined_collab_0329to0627.csv"
    # collab_df = pd.read_csv(collaboration_fname)

    # import department affiliation dataset
    dept_affil_fname = "data/Company_experiment/202104-202106_department affiliation.xlsx"
    dept_affil_df = pd.read_excel(dept_affil_fname)

    # extract full-time employees
    # fulltime_emply_ids = employee_info_df.employee_id.unique()

    # collab_df = collab_df[(collab_df.emp_a.isin(fulltime_emply_ids))|(collab_df.emp_b.isin(fulltime_emply_ids))]
    # collab_df.reset_index(drop=True, inplace=True)

    # dept_affil_df = dept_affil_df[dept_affil_df.employee_id.isin(fulltime_emply_ids)]
    # dept_affil_df.reset_index(drop=True, inplace=True)

    print("Generating teams...")
    # based on department ID
    # dept_num = 4
    # depts = dept_affil_df["deptid_{}".format(dept_num)].unique()
    # for date in dept_affil_df.dt.unique()[:1]:
    #     time_dept = dept_affil_df[dept_affil_df.dt == date]

    #     dept_ids = time_dept["deptid_{}".format(dept_num)].unique()
    #     team_count = 0
    #     for dept_id in dept_ids:
    #         team = time_dept[time_dept["deptid_{}".format(dept_num)] == dept_id]
    #         team.reset_index(drop=True, inplace=True)

    #         g = nx.from_pandas_edgelist(team, source="leader_id", target="employee_id", create_using=nx.DiGraph)
    #         root = [n for n,d in g.in_degree() if d == 0]
    #         print("Number of leaders: {}".format(len(root)))
    #         print("Number of components: {}".format(len(list(nx.weakly_connected_components(g)))))
    #         spl = nx.shortest_path_length(g, source=root[0])
    #         print("Max depth: {}".format(max(spl.values())))

    for date in dept_affil_df.dt.unique()[1:2]:
        time_dept = dept_affil_df[dept_affil_df.dt == date]
        time_dept.reset_index(drop=True, inplace=True)

        g = nx.from_pandas_edgelist(time_dept, source="leader_id", target="employee_id", \
                                    create_using=nx.DiGraph)

    roots = [n for n,d in g.in_degree() if d == 0] # leaders of teams
    new_g = g.copy()
    # exclude teams with less than 3 levels
    valid_teams = 0
    for r in roots:
        spl = nx.shortest_path_length(g, source=r)
        max_depth = max(spl.values())
        if max_depth > 1:
            valid_teams += 1
            exclude_members = [node for node in spl if spl[node] > 2]
            new_g.remove_nodes_from(exclude_members)
        else:
            new_g.remove_nodes_from(list(spl.keys()))

    # check the number of team members
    num_members_list = []
    components = nx.weakly_connected_component(new_g)
    for c in components:
        team = new_g.subgraph(c).copy()
        num_members_list.append(team.number_of_nodes())


    # check team leaders' report likes
    time_collab_df = collab_df[(pd.to_datetime(collab_df.dt) >= "2021-04-01") & \
                             (pd.to_datetime(collab_df.dt) < "2021-05-01")]
    new_leaders = [n for n,d in new_g.in_degree() if d == 0]
    likes_list = []
    for leader in new_leaders:
        likes = time_collab_df[(time_collab_df.emp_b == leader)].num_weekly_like
        likes_list.append(likes.sum())
        

        


