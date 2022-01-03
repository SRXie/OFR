import csv

file = open("/checkpoint/siruixie/clevr_obj_test/output/obj_test/CLEVR_test_cases.csv")
reader = csv.reader(file)
lines= len(list(reader))

print(lines)
