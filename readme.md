# Hierarchical Team Performance Prediction Project

Author: Sulyun Lee<br>
Email: sulyun-lee@uiowa.edu / 0tnfus1230@gmail.com<br>
Last modified: Sep 21, 2022<br>

## Data
There are two folders under ``data/`` directory, ``NFL_experiment/`` and ``Company_experiment/``, which contain datasets for the Chinese corporation teams and NFL coach teams experiments.

### NFL_experiment
1. ``NFL_Coach_Data_final_position.csv``
This dataset contains NFL coach information of all teams in seasons between 2002 and 2019. 

* Column information

|Column|Description|
|------|-----------|
|Position|Uncleaned texts of a coach's position name|
|Name|Name of a coach|
|Year|Year of a season|
|Team|Name of a team (there are 32 unique NFL teams)|
|Position_list|A list of a coach's positions extracted from *Position* column. The texts are cleaned by removing special characters and converting to lowercase.|
|final_position|Letters that indicate the position of a coach. If a coach held multiple positions in the *Position_list*, then the upper level of position is assigned to the coach. The letters are defined as following:<br> <ul><li>HC: Head Coach</li><li>iHC: Interim Head Coach</li><li>OC: Offensive Coordinator</li><li>DC: Defensive Coordinator</li><li>O: Offensive Position Coach</li><li>D: Defensive Position Coach</li><li>-1: Not a qualified coach</li></ul>|
|final_position_spec|Letters that indicate the specific position of a coach. Head coaches and coordinators have the same value with *final_position* column, but position coaches have the different value by indicating the specific position name.Some of the positions can be called as different names. The letters are defined as the following:<br><ul><li>DL: Defensive Line</li><li>LB: Linebacker</li><li>OL: Offensive Line</li><li>QB: Quarterback</li><li>RB: Running Back</li><li>Sec: Secondary (=Defensive Back, Cornerback, etc.)</li><li>TE: Tight End</li><li>WR: Wide Receiver</li></ul>|
|final_hier_num|Hierarchical number given based on the final position of a coach.  This column has the following values:<br><ul><li>1: Head coach</li><li>2: Coordinator (Offensive/Defensive)</li><li>3: Position coach (Offensive/Defensive)</li><li>-1: Not a qualified coach</li></ul>|

2. ``all_coach_records_cleaned.csv``
This dataset contains NFL coaches' previous NFL & college football coach experiences up to 2001.

* Column information

|Column|Description|
|------|-----------|
|Name|Name of a coach|
|StartYear|The starting year of serving a team|
|EndYear|The ending year of serving a team|
|ServingTeam|The name of a team|
|Position|Name of the position title|
|NFL|Indicator of whether a team is NFL team or not. <br><ul><li>1: NFL team</li><li>0: College football team</li></ul>|

3. ``Total_Win.csv``
This dataset contains win/loss records of NFL teams from 1997 to 2019.

* Column information

|Column|Description|
|------|-----------|
|Team|Name of a NFL team|
|Year|Year of the record|
|Total_Win|The number of wins in the season|
|Total_Lose|The number of loses in the season|

### Company_experiment
1. ``employee_info_final.csv``
This dataset contains personal information of full-time employees.

* Column information

|Column|Description|
|------|-----------|
|employee_id|Unique ID of an employee|
|gender|Gender of an employee|
|yob|Year of birth|
|highest_degree|Highest degree obtained|

2. ``dept_affil_final.csv``
This dataset contains department affiliation information of individual employees.

* Column information

|Column|Description|
|------|-----------|
|employee_id|ID of an employee|
|department_id|ID of an employee's affiliated department|
|leader_id|ID of an employee's leader|
|paygrade|Paygrade of an employee|

3. ``message_collab_final.csv``
This dataset contains collaborations between pairs of employees through messaging each other.

* Column information

|Column|Description|
|------|-----------|
|dt|Date of the collaboration|
|emp_a|ID of employee 1|
|emp_b|ID of employee 2|
|emp_a_deptid|ID of employee 1's department|
|emp_b_deptid|ID of employee 2's department|
|send_num|The number of messages sent between employee 1 and employee 2|
|rel|Hierarchical relationship between employee 1 and employee 2|

4. ``hier_team_edgelist_final.csv``
This dataset contains edgelist of hierarchical relationships among employees. In this edgelist, a pair of nodes represent the tie between a leader and its subordinate. The hierarchical information was inferred based on the leader information in the ``dept_affil_final.csv`` dataset.

* Column information

|Column|Description|
|------|-----------|
|source|Employee ID of a source node|
|target|Employee ID of a target node|
|dt|Date of the hierarchical relationship|

5. ``paygrade_rankings.csv``
This dataset contains the ranking information of paygrade column in the ``dept_affil_final.csv`` dataset. There are rankings from 1 to 21. This data is used to map the rankings based on employees' paygrade information to figure out the hierarchical collaborations during the node embedding process.

* Column information

|Column|Description|
|------|-----------|
|ranking|Ranking in numeric value. Lower numbers are higher rankings|
|paygrade_M|Paygrade indicator for management|
|paygrade_P|Paygrade indicator for professional|

------

## Generate node embeddings for collaboration features
Generate node embeddings of each coach using vanilla DeepWalk and variations of DeepWalk on cumulative collaboration networks.
The generated node embeddings represent the coaches' previous collaboration experiences.

* Network construction <br>
In a cumulative collaboration network, nodes are the team members and edges are the connectivity among team members. In NFL experiment, the connectivity is determined if a pair of nodes had ever worked together in the same team; and in Company experiment, the connectivity is determined if a pair of nodes exchanged messages at least one time. 

* Node embedding generation <br>
Node embeddings are learned based on the cumulative collaboration network to capture the individual's previous collaboration experiences. Node embeddings learned with vanilla DeepWalk approach (``unbiased``) treats all collaborations as equally important. Giving bias to the random walk allows for the consideration of network property, such as team structures and collaboration strength (``hierarchy``, ``strength``, ``recency``).

In NFL experiment, hierarchy is determined based on coaches' hierarchical position. For example, the random walk from a defensive coordinator to a head coach is given greater probability of traversing compared to a walk from a head coach to a defensive coordinator. In Company experiment, hierarchy is determined based on the paygrades of employees. Each employee is given a ranking baed on the paygrade and the random walk from a lower rank to a higher rank has a higher probability of traversing. For both experiments, the strength of collaborations is correlated with the number of previous collaborations and the recency of collaborations is inversely correlated with the time passed since the most recent collaboration.

* Required packages
```
gensim
pandas
numpy
networkx
tqdm
```

* Usage example <br>

*NFL experiment*
```
python node_embedding.py -emb_size 32 -window_size 3 -bias hierarchy 
```

*Company experiment*
```
python _new_expr_node_embedding.py -emb_size 32 -window_size 3 -bias hierarchy
```

* Input arguments
1. ```--emb_size```: the number of embedding dimensions for each node (```dtype:int```)
2. ```--window_size```: the size of the sliding window used in Skip-gram model. e.g., window size of 3 means 3 neighbors on each side of the target node (```dtype:int```)
3. ```--bias```: the bias type of the random walk during the DeepWalk. The available options are *unbiased*, *hierarchy*, *strength*, and *recency* (```dtype:str```)

* Output <br>
The saved csv file that contains the node embeddings of every node in each timestamp. The file is saved in ``data/NFL_experiment/embeddings/`` directory for the NFL experiment and ``data/Company_experiment/embeddings/`` directory for the company experiment.
The file contains the following columns:
    - ***Name***: Name of a node
    - ***Year*** (or ***Week***): The prediction year (or week) for using the node embedding (e.g., the year is 2005 when the node embeddings are learned based on the cumulative collaboration network up to year 2004.)
    - ***cumul_emb0***, ***cumul_emb1***, ...: The embedding vector in each dimension

------
## Generate node features
Generate the nodes'(team members') features with the combination of individual and collaboration features.

### NFL experiment
* Individual features <br>
1. ```TotalYearsInNFL```: the number of years that a coach has worked in NFL teams
2. ```Past5yrsWinningPerc_best```: The highest achieved winning percentage during the previous 5 seasons in NFL teams
3. ```Past5yrsWinningPerc_avg```: The average achieved winning percentage during the previous 5 seasons in NFL teams

* Collaboration features <br>
Node embeddings learned based on collaboration networks are used as collaboration features for coaches. To predict the team performance in year ```y```, the node embeddings generated based on collaborations up to year ```y-1``` are used as the collaboration features. 

### Company experiment
* Individual features <br>
1. ```Tenure```: the tenure of employees. i.e., the number of years since the hire date
2. ```EduLevel```: the education level of employees. Larger values indicate higher education
3. ```JobRank```: the rank of the job title based on the paygrade. Larger values indicate higher job rank

* Collaboration features <br>
Node embeddings learned based on collaboration networks are used as collaboration features for employees. To predict the team performance in week ```w```, the node embeddings generated based on collaborations up to week ```w-1``` are used as the collaboration features. 


* Required packages
```
pandas
numpy
argparse
tqdm
```

* Input arguments
1. ```--emb_size```: the number of embedding dimensions for each node (```dtype:int```)
2. ```--w```: the size of the sliding window used in Skip-gram model. e.g., window size of 3 means 3 neighbors on each side of the target node (```dtype:int```)
3. ```--bias```: the bias type of the random walk during the DeepWalk. The available options are *unbiased*, *hierarchy*, *strength*, and *recency* (```dtype:str```)

* Output <br>
For the NFL experiment, the saved csv file contains NFL coaches' individual and collaboration features. The file is stored under ```data/NFL_experiment/``` directory. The data file contains the following columns:
    - ***Position***: Position name of a coach
    - ***Name***: Name of a coach
    - ***Year***: Year of the season
    - ***Team***: Name of a coach's team
    - ***Position_list***: List of a coaches' position titles
    - ***final_position***: The final assigned position letters. Available letters are HC, DC, OC, D, and O.
    - ***final_position_spec***: If a coach's position is either defensive or offensive position coach, the specific position names are given. Defensive position coaches (D) include Secondary (Sec), Defensive Lines (DL), Linebackers (LB); and offensive position coaches (O) include Offensive Line (OL), Quarterbacks (QB), Running backs (RB), Tight Ends (TE), Wide Receivers (WR).
    - ***final_hier_num***: Numbers given to a coach based on the final position. This column has the following values: 1 for head coaches, 2 for offensive/defensive coordinators, and 3 for offensive/defensive position coaches.
    - ***TotalYearsInNFL***: Individual feature. Total number of years served as a NFL coach.
    - ***Past5yrsWinningPerc_best***: Individual feature. Best winning percentage that a coach achieved during the past 5 years.
    - ***Past5yrsWinningPerc_avg***: Individual feature. Average winning percentage that a coach achieved during the past 5 years.
    - ***HC***: Indicator whether a coach is a head coach.
    - ***Coord***: Indicator whether a coach is a coordinator (offensive/defensive).
    - ***cumul_emb0*** ~ ***cumul_emb31***: Collaboration feature. Node embedding dimensions from 0 to 31 learned based on the cumulative collaboration network.

For the Company experiment, the saved csv file contains employees' individual and collaboration features. The file is stored under ```data/Company_experiment/``` directory. The data file contains the following columns:
    - ***Employee***: Employee ID
    - ***Date***: Date of the team formation
    - ***TeamID***: ID of a team. Employees having the same team ID are in the same team.
    - ***Tenure***: Individual feature. Tenure of employees (how many years an employee worked in the company).
    - ***EduLevel***: Individual feature. Education level of an employee.
    - ***JobRank***: Individual feature. Rank of an employee's job position determined by paygrade.
    - ***cumul_emb0*** ~ ***cumul_emb31***: Collaboration feature. Node embedding dimensions from 0 to 31 learned based on the cumulative collaboration network.

* Usage example
```
python generate_coach_features.py -emb_size 32 -w 3 -bias hierarchy 
```
```
python _new_expr_generate_node_features.py -emb_size 32 -w 3 -bias hierarchy 
```

## Team embedding
### NFL experiment

* Hierarchy2vec (proposed method)
```
python hier_bias_attn_embedding.py -individual True -collab True -train_split_year 2015 -valid_split_year 2017 -hierarchy True -strength True -recency True -drop_rate 0.4 
```

* Hierarchy2vec AvgNetEmb (benchmark)
```
python hier_team_embedding.py -individual True -colalb True -train_split_year 2015 -valid_split_year 2017 -bias averaged -drop_rate 0.4
```

* Hierarchy2vec FlatAgg (benchmark)
```
python nonhier_bias_attn_embedding.py -individual True -collab True -train_split_year 2015 -valid_split_year 2017 -hierarchy True -strength True -recency True -drop_rate 0.4  
```

* OptMatch (benchmark)
```
python nonhier_optmatch_embedding.py -individual True -collab True -train_split_year 2015 -valid_split_year 2017 -hierarchy True -strength True -recency True -drop_rate 0.4  
```

* Fast and Jensen (benchmark)
```
python benchmark_modeling.py -train_split_year 2015 -valid_split_year 2017
```
***Note: the input data for Fast and Jensen is generated using a script benchmark_data_construction.py***









