import pandas as pd

if __name__ == "__main__":
    
    ### Clean employee info
    # import employee personal info dataset
    employee_info_filename = "data/Company_experiment/employee_personal_info.csv"
    employee_info_df = pd.read_csv(employee_info_filename)

    # clean employee personal info
    employee_info_df.rename(columns={'工号': 'employee_id', '性别': 'gender', '出生年月':'yob',\
                                    '最高学历学历':'highest_degree', '员工类别':'employee_type',\
                                    '最近一次雇佣日期':'hire_date'}, inplace=True)
    m = {'03 - 硕士研究生': '03 - Master', '02 - 博士研究生':'02 - PhD', '04 - 本科':'04 - Bachelor', \
        '05 - 大专': '05 - Associate Degree', '08 - 初中': '08 - Junior High', '07 - 中专/技校': '07 - Vocational School',\
        '06 - 高中': '06 - High School', '09 - 小学及以下': '09 - Elementary School or Below'}
    employee_info_df.highest_degree = employee_info_df.highest_degree.map(m)

    employee_info_df.loc[employee_info_df.employee_type == '全职员工', 'employee_type'] = 'full-time'
    employee_info_df = employee_info_df.loc[employee_info_df.employee_type == "full-time",:].reset_index(drop=True)

    employee_info_df.to_csv(employee_info_filename, index=False)


    ### Clean collaboration info
    import_ind = True
    if import_ind:
        collab_combined_df = pd.read_csv("data/Company_experiment/combined_collab_0329to0627.csv")

    else:

        # combine collaboration data (March-June)
        str_list = ["03-29to04-11", "04-12to04-25", "04-26to05-09_new", "05-10to05-23", "05-24to06-06", \
                    "06-07to06-20", "06-27"]
        collab_df_list = []
        for e in str_list:
            fname = "data/Company_experiment/opera{}.csv".format(e)
            collab_df = pd.read_csv(fname)
            collab_df_list.append(collab_df)

        collab_combined_df = pd.concat(collab_df_list, axis=0)
        collab_combined_df = collab_combined_df.iloc[:, 1:]
        collab_combined_df.to_csv("data/Company_experiment/combined_collab_0329to0627.csv", index=False)
    
    # import additional collaboration data 
    import_ind = False
    if import_ind:
        collab_combined_df2 = pd.read_csv("data/Company_experiment/combined_collab_0704to0718.csv")
    else:
        # combine collaboration data (July-)
        str_list = ["2021-07-04", "2021-07-11", "2021-07-18"]
        collab_df_list = []
        for e in str_list:
            fname = "data/Company_experiment/{}.csv".format(e)
            collab_df = pd.read_csv(fname)
            collab_df_list.append(collab_df)
        
        collab_combined_df2 = pd.concat(collab_df_list, axis=0)
        collab_combined_df2.to_csv("data/Company_experiment/combined_collab_0704to0718.csv")

    collab_combined_final = pd.concat([collab_combined_df, collab_combined_df2], axis=0)
    collab_combined_final.to_csv("data/Company_experiment/combined_collab0329to0718.csv", index=False)

    collab_combined_final.loc[collab_combined_final.rel == '其他', "rel"] = "Other"
    collab_combined_final.loc[collab_combined_final.rel == '平级', "rel"] = "N+0"
    collab_combined_final.loc[collab_combined_final.rel == '自己', "rel"] = "Self"

    collab_combined_final = collab_combined_final.loc[collab_combined_final.rel != "Self", :].reset_index(drop=True)
    
    # extract full-time employees
    fulltime_emply_ids = employee_info_df.employee_id.unique()
    collab_combined_final = collab_combined_final[(collab_combined_final.emp_a.isin(fulltime_emply_ids))&\
                                                (collab_combined_final.emp_b.isin(fulltime_emply_ids))]
    collab_combined_final.reset_index(drop=True, inplace=True)

    # extract meeting network
    # collab_meeting_df = collab_combined_final[collab_combined_final.meet_num > 0]
    # collab_meeting_df.reset_index(drop=True, inplace=True)
    # collab_meeting_df.to_csv("data/Company_experiment/combined_meeting_collab.csv", index=False)

    # extract message network
    collab_message_df = collab_combined_final[collab_combined_final.send_num > 0]
    collab_message_df.reset_index(drop=True, inplace=True)
    collab_message_df.to_csv("data/Company_experiment/combined_message_collab.csv", index=False)


    ### Clean department affiliation data
    dept_affil_fname1 = "data/Company_experiment/202104-202106_department affiliation.xlsx"
    dept_affil_fname2 = "data/Company_experiment/202107-202109_department affiliation.xlsx"

    dept_affil_df1 = pd.read_excel(dept_affil_fname1)
    dept_affil_df2 = pd.read_excel(dept_affil_fname2)

    m = {'员工工号': 'employee_id', '部门id': 'department_id', '日期': 'dt', '职级': 'paygrade'}
    dept_affil_df2 = dept_affil_df2.rename(columns=m, inplace=True)

    cols = ['employee_id', 'department_id'] + ['deptid_{}'.format(i) for i in range(1, 13)] + \
            ['dt', 'leader_id', 'paygrade']
    dept_affil_final = pd.concat([dept_affil_df1[cols], dept_affil_df2[cols]], axis=0)

    # extract full-time employees
    dept_affil_final = dept_affil_final[dept_affil_final.employee_id.isin(fulltime_emply_ids)].reset_index(drop=True)
    dept_affil_final.to_csv("data/Company_experiment/department_affiliation_combined.csv", index=False)



