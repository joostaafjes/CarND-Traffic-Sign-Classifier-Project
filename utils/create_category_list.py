import csv

category_ids = []
category_descs = []

with open('../signnames.csv', 'r') as signnames:
    sign_reader = csv.reader(signnames)
    for sign in sign_reader:
        cat_id = str(sign[0])
        if cat_id.isnumeric():
            # get meta data for cat
            category_ids.append(int(cat_id))
            category_descs.append(sign[1])

print(set(zip(category_ids, category_descs)))

for id, desc in zip(category_ids, category_descs):
    print(id, desc)
