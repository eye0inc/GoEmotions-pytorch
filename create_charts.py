import sys, os
from pprint import pp
#
# clean up data
#
for dataset in "train", "dev", "test":
    data = []
    fn = f"ckpt/original/bert-base-cased-goemotions-original/eval_{dataset}_results.txt"
#     print (fn)
    f = open(fn)
    for row in f.readlines():
        row = row.strip().split('=')
        print (row[0])
        i = row[0].rfind("_")
        print (i)
        metric = row[0][:i]
        epoch = int(row[0][i+1:])
        value = float(row[1])
        data.append((metric, epoch, value))
#         break
    data.sort()
    pp(data)
    data2 = {}
    cols = set()
    for m, e, v in data:
        if e not in data2:
            data2[e] = {}
        data2[e][m] = v
        cols.add(m)
    pp(data2)
    cols = list(cols)
    cols.sort()
    head = "epoch," + ",".join(cols) + ","
    f = open(f"{dataset}.csv", 'w')
    print (head, file=f)
    for k, v in data2.items():
        print (f"{k},", end="", file=f)
        for col  in cols:
            print (f"{v[col]},", end="", file=f)
        print ("", file=f)