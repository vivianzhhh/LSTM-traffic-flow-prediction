import pandas as pd
import numpy as np
import csv

col = ["datetime","lane3","lane4","agg","lane","observed"]
pre = "traffic_flow/pems_output ("
frames1 = []
frames2 = []
frames3 = []
shape = []

for i in range(0,53):
    df = pd.read_excel(open(pre+str(i)+').xlsx','rb'), names=col)
    if df.shape == (180,6):
        frames1.append(df)
    elif df.shape ==(144,6):
        frames2.append(df)
    else:
        frames3.append(df)        

df1 = pd.concat(frames1)
df2 = pd.concat(frames2)
df3 = pd.concat(frames3)
df = pd.concat([df1,df2,df3])

df['flow'] = df['lane3'] + df['lane4']
flow = df['flow'].tolist()
 
# 36 records/day
def chunks(l,n): # n = 36
    for i in range(0,len(l),n):
        yield l[i:i+n]

flow = list(chunks(flow,36))


# length of sequence : a
m = np.shape(flow)[0]
n = np.shape(flow)[1]

seq_length = [3,6,9,12]

for a in seq_length:

    input_flow = []
    for i in range(0,m):
        day_flow = flow[i]
        for j in range(0,n-a,3):
            record = day_flow[j:j+(a+1)]
            input_flow.append(record)
       
    train_data = []
    label = []     
    for i in range(0,(np.shape(input_flow)[0])):
        train_data.append(input_flow[i][0:a]) 
        label.append(input_flow[i][a])
    
    train_name = "train_data" + str(a) + ".csv"
    label_name = "label_data" + str(a) + ".csv"
    
    train = pd.DataFrame(train_data)
    train.to_csv(train_name,index = False,header = False)
    
    with open(label_name,"w") as f:
        wr = csv.writer(f,delimiter="\n")
        wr.writerow(label)   # -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

