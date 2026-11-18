import csv
with open("C:\\Users\\adam.demjanovic\\Desktop\\scp_statements.csv", mode='r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        #print(row)
        if row[7] == 'Myocardial Infarction':
            print(row[0])
