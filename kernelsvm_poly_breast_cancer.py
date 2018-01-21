# 2 - benign 4 - malignant ==> -1 and 1
from svmutil import *

# Read Data from Files:
print('Reading Data From Files.....')
test_indices = []
with open('breast-cancer-scale-test-indices.txt') as f:
    for line in f:
        line = line.rstrip()
        test_indices.append(int(line))


def test_class(index):
    if index in test_indices:
        return True
    else:
        return False

with open('breast-cancer_scale.txt') as f:
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    index = 1
    for line in f:
        line = line.rstrip()
        temp = line.split(' ')
        feature_row = []
        for i in range(1, 11):
            col, data = temp[i].split(':')
            feature_row.append(float(data))
        if test_class(index):
            X_test.append(feature_row)
            if temp[0] == '2':
                Y_test.append(-1.0)
            elif temp[0] == '4':
                Y_test.append(1.0)
        else:
            X_train.append(feature_row)
            if temp[0] == '2':
                Y_train.append(-1.0)
            elif temp[0] == '4':
                Y_train.append(1.0)
        index += 1

# C values: 0.1 1 10 100 1000


def train_classifier(y, x, c):
    prob = svm_problem(y, x)
    param = svm_parameter('-s 0 -t 1 -c '+str(c)+' -q')
    return svm_train(prob, param)

accuracy_list = []
C_list = [0.1, 1, 10, 100, 100]

print('Set of C values: '+str(C_list))
for C in C_list:
    # 5-fold cross validation for each C:
    accuracy = 0
    for i in range(5):
        indices_train = list(set(range(0, 500)) ^ set(range(100*i, 100*(i+1))))
        indices_test = list(range(100*i, 100*(i+1)))

        train_set_x = [X_train[i] for i in indices_train]
        train_set_y = [Y_train[i] for i in indices_train]
        val_set_x = [X_train[i] for i in indices_test]
        val_set_y = [Y_train[i] for i in indices_test]

        m = train_classifier(train_set_y, train_set_x, C)
        p_label, p_acc, p_val = svm_predict(val_set_y, val_set_x, m)
        ACC, MSE, SCC = evaluations(val_set_y, p_label)

        accuracy = accuracy + ACC
    accuracy_list.append(accuracy/5.0)

print('Accuracies (validation set): '+str(accuracy_list))

max_value = max(accuracy_list)
max_index = accuracy_list.index(max_value)

print('Best Value of C : '+str(C_list[max_index]))

C = C_list[max_index]

# Re-train classifier using this C value:

m = train_classifier(Y_train, X_train, C)

p_label, p_acc, p_val = svm_predict(Y_test, X_test, m)
ACC, MSE, SCC = evaluations(Y_test, p_label)

print('Accuracy with optimal C on training Data: '+str(ACC))
