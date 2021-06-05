import pandas as pd
import numpy as np
import json as js
import matplotlib.pyplot as plt
import seaborn as sns

### loading data
data = pd.read_csv('youtube_trending/USvideos.csv', sep = ',')

with open('youtube_trending/US_category_id.json') as f: 
    json_date = js.load(f)
    f.close()

print('-------------Processing Category Id-----------')
### mapping id to category
id_cat = {}
for i in range(len(json_date['items'])): 
    id_cat[json_date['items'][i]['id']] = json_date['items'][i]['snippet']['title']
    
print(len(data))
for i in range(len(data)):   
    id = data.loc[i, 'category_id']
    data.loc[i, 'category_id'] = id_cat[str(id)]
print(data['category_id'])

print('-------------Processing views-----------')
views = data['views']
quater_1 = views.quantile(0.25)
quater_3 = views.quantile(0.75)
view_part = []
for i in data['views']:
    if int(i) >= quater_3:
        view_part.append('high_view')
    elif int(i) <= quater_1:
        view_part.append('low_view')
    else:
        view_part.append('medium_view')
print(view_part[:20])

print('-------------Processing likes-----------')
favorable = []
for i in range(len(data)):
    if data.loc[i, 'likes'] >= data.loc[i, 'dislikes']:
        favorable.append('favorable')
    else:
        favorable.append('unfavorable')
print(favorable[0:20])

print('-------------Processing Comment_count-----------')
comment_counts = data['comment_count']
views = data['views']
comment_by_views = comment_counts / views
one = comment_by_views.quantile(0.25)
three = comment_by_views.quantile(0.75)
comment_part = []
for i in comment_by_views:
    if i >= three:
        comment_part.append('high_comment')
    elif i <= one:
        comment_part.append('low_comment')
    else:
        comment_part.append('medium_comment')
print(comment_part[0:10])

# replace the original data columns
data = data.drop(['views', 'likes', 'dislikes', 'comment_count'], axis = 1)
data.insert(0, 'views', view_part)
data.insert(0, 'like', favorable)
data.insert(0, 'comment_count', comment_part)
# delete the other data columns
data = data.drop(['video_id', 'trending_date', 'publish_time', 'video_error_or_removed', 'description', 'thumbnail_link', 'title', 'comments_disabled', 'ratings_disabled'], axis = 1)
print(data.head())

# change the data format (strings -> ints)
id2str = {}   # reversed mapping (int -> string), for following steps
str2id = {}   # mapping (string -> int)
id = 0
transaction = []
for i in range(len(data)):
    one = []
    for j in data.columns:
        if j == 'tags':
            str_arr = data.loc[i, j].split('|')
            for s in str_arr:
                if s in str2id:
                    one.append(str2id[s])
                else:
                    id2str[id] = s
                    str2id[s] = id
                    one.append(id)
                    id += 1
        else:
            if data.loc[i, j] in str2id:
                one.append(str2id[data.loc[i, j]])
            else:
                id2str[id] = data.loc[i, j]
                str2id[data.loc[i, j]] = id
                one.append(id)
                id += 1
    transaction.append(one)
    
print('------------Part of the processed data-----------')
print(transaction[:10])

import Orange as og
import orangecontrib.associate.fpgrowth as oaf

# frequent pattern mining
threshold = 0.3
print('------------Threshould = 0.3------------')
items = list(oaf.frequent_itemsets(transaction, threshold))
for i in items:
    freq_set = []
    abs_sup = i[1]
    for j in i[0]:
        freq_set.append(id2str[j])
    print(freq_set, abs_sup, round(float(abs_sup) / len(data), 2))
    
print('------------Threshould = 0.2------------')
threshold = 0.2 
items = list(oaf.frequent_itemsets(transaction, threshold))
for i in items:
    freq_set = []
    abs_sup = i[1]
    for j in i[0]:
        freq_set.append(id2str[j])
    print(freq_set, abs_sup, round(float(abs_sup) / len(data), 2))
    
# Rules
items = list(oaf.frequent_itemsets(transaction, 0.2))
rules = list(oaf.association_rules(dict(items), 0.2))
for i in rules:
    antecedent = []
    consequent = []
    for j in i[0]:
        antecedent.append(id2str[j])
    for j in i[1]:
        consequent.append(id2str[j])
    print(antecedent, "-", consequent, i[2], round(i[3],2))
print(f'Length of rules: {len(rules)}')


measure = list(oaf.rules_stats(oaf.association_rules(dict(items), 0.2), dict(oaf.frequent_itemsets(transaction, 0.2)), len(data)))
for i in measure:
    antecedent = []
    consequent = []
    for j in i[0]:
        antecedent.append(id2str[j])
    for j in i[1]:
        consequent.append(id2str[j])
    print(antecedent, "-", consequent, round(i[6], 2))
    
# Caclculate Kulc 
kulc = []
visit = [False for i in range(len(rules))]
for i in range(len(rules)):
    if visit[i] == True:
        continue
    visit[i] = True
    for j in range(len(rules)):
        if visit[j] == True:
            continue
        if rules[j][0] == rules[i][1] and rules[j][1] == rules[i][0]:
            one = []
            antecedent = []
            consequent = []
            for k in rules[i][0]:
                antecedent.append(id2str[k])
            for k in rules[i][1]:
                consequent.append(id2str[k])
            one.append(rules[i][0])
            one.append(rules[i][1])
            one.append((rules[i][3] + rules[j][3])/2)
            kulc.append(one)
            print(antecedent, "-", consequent, round((rules[i][3] + rules[j][3])/2, 2))
            visit[j] = True
            
            
#### visualization heatmap      
# Visualization
conf_matrix = []
rules_column = set()

for i in range(len(measure)):
    rules_column.add(measure[i][0])
# Condidence matrix (modified version of confusion matrix)
# calculate lift matrix
lift_matrix = []
for i in rules_column:
    one = []
    for j in rules_column:
        if i == j:
            one.append(1)
        else:
            flag = False
            for k in range(len(measure)):
                if measure[k][0] == i and measure[k][1] == j:
                    one.append(measure[k][6])
                    flag = True
            if flag == False:
                one.append(0)
    lift_matrix.append(one)

lift_pd = pd.DataFrame(lift_matrix, columns = rules_column_list, index = rules_column_list)
plt.figure(figsize=(11, 9),dpi=100)
sns.heatmap(data = lift_pd, annot = True, fmt = ".2f", cmap = "Purples")
plt.show()


# Calculate Kulc matrix
kulc_matrix = []
for i in rules_column:
    one = []
    for j in rules_column:
        if i == j:
            one.append(1)
        else:
            flag = False
            for k in range(len(kulc)):
                if kulc[k][0] == i and kulc[k][1] == j:
                    one.append(kulc[k][2])
                    flag = True
            if flag == False:
                one.append(0)
    kulc_matrix.append(one)
    
kulc_pd = pd.DataFrame(kulc_matrix, columns = rules_column_list, index = rules_column_list)
plt.figure(figsize=(11, 9),dpi=100)
sns.heatmap(data = kulc_pd, annot = True, fmt = ".2f", cmap = "Purples")
plt.show()
