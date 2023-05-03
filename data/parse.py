with open('sql.txt') as f:
    contents = f.read().split('\n')
f.close()

end = []

with open('matrix.csv','w') as file:
    for line in contents:
        line = line[33:]
        line = line[:-2]
        r = line.split('),(')
        for row in r:
            k = row.split(',\'')
            k = k[1][:-1]
            act = [float(x) for x in k.split(',')]
            end.append(act)
            file.write(k)
            file.write('\n')



