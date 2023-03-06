import csv
import numpy as np
from matplotlib import pyplot as plt

from configs import *

patient_ids = []
with open('patients.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()

    for line in filecontents:
        patient = line[:-1]
        patient_ids.append(patient)

clinic_data_path = 'data/clinic/TCGA-clinic.csv';

with open(clinic_data_path, newline='') as csvfile:
    all_clinic_data = list(csv.reader(csvfile))[1:]

patients = []
overall_risk_score = []
for id in patient_ids:
    for row in all_clinic_data:
        if id in row:
            patients.append(row[0])
            overall_risk_score.append(row[-3])

overall_risk_score = np.array(overall_risk_score)
overall_risk_score = overall_risk_score.astype(float)

### Sanity check
for id in patient_ids:
    if id not in patients:
        print(id)

risk_mean = np.mean(overall_risk_score)
risk_dev = np.sqrt(np.var(overall_risk_score))

overall_risk_score = (overall_risk_score - risk_mean) / risk_dev

print(np.mean(overall_risk_score))
print(np.var(overall_risk_score))

plt.figure(1)
plt.plot(overall_risk_score, marker='o', markerfacecolor='blue', markersize=9)
plt.show()

if classification:
    for i, score in enumerate(overall_risk_score):
        if score < 0:
            overall_risk_score[i] = 1
        else:
            overall_risk_score[i] = 0

low_risk_patients = []
high_risk_patients = []

for i, id in enumerate(patients):
    if overall_risk_score[i] == 1:
        high_risk_patients.append(id)
    if overall_risk_score[i] == 0:
        low_risk_patients.append(id)


with open('test_patients.txt', 'w') as f:
    for id in patients[:val_set_size]:
        f.write(id)
        f.write('\n')

with open('train_patients.txt', 'w') as f:
    for id in patients[val_set_size:]:
        f.write(id)
        f.write('\n')

with open('high_risk_patients.txt', 'w') as f:
    for id in high_risk_patients:
        f.write(id)
        f.write('\n')

with open('low_risk_patients.txt', 'w') as f:
    for id in low_risk_patients:
        f.write(id)
        f.write('\n')


np.save("np/y_train", overall_risk_score[val_set_size:])
np.save("np/y_val", overall_risk_score[:val_set_size])

print(2)
print(3)