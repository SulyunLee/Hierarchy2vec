
import pandas as pd
import numpy as np
import networkx as nx

if __name__ == "__main__":
    # import employee personal info dataset
    employee_info_filename = "data/Company_experiment/employee_personal_info.csv"
    employee_info_df = pd.read_csv(employee_info_filename)

    # import department affiliation dataset
    dept_affil_fname = "data/Company_experiment/202104-202106_department affiliation.xlsx"
    dept_affil_df = pd.read_excel(dept_affil_fname)

    employees = dept_affil_df.employee_id.unique()

    # import message collaboration network
    message_collab_fname = "data/Company_experiment/combined_message_collab.csv"
    message_collab_df = pd.read_csv(message_collab_fname)

    for date in message_collab_df.dt.unique():
        collabs = message_collab_df[message_collab_df.dt <= date]
        collabs.reset_index(drop=True, inplace=True)

        g = nx.from_pandas_edgelist(collabs, source="emp_a", target="emp_b")
        not_covered = [x for x in employees if x not in g.nodes()]
        print("Not covered: {}".format(len(not_covered)))

        cc = nx.connected_components(g)
        comp_size = [g.subgraph(s).copy().number_of_nodes() for s in cc]
        print("Connected components: {}".format(cc))
        print("Component sizes: {}".format(comp_size))
