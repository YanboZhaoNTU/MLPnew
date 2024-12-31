from u_mReadData import *
from u_base import *
import numpy as np
from u_evaluation import *
from L2 import *
from weight import *
from BRclass import *

from LP import *


datasnames = ["Yeast"]
num = 0
result = np.array([])
Y_tr = np.array([])
X_te = np.array([])
Y_te = np.array([])
w_save = np.array([])
X_tr_list = []
Y_tr_list = []
X_te_list = []
Y_te_list = []
rd = ReadData(datas=datasnames, genpath='data/')
X_train, Y_train, X_test, Y_test = rd.readData(0)
print(np.shape(X_train), np.shape(Y_train), np.shape(X_test), np.shape(Y_test))
# (1629, 103) (1629, 14) (788, 103) (788, 14)
BRt = BRclass()
LPt = LabelPowersetLogistic()

for h in range(3):
    X_tr_list = []
    Y_tr_list = []
    X_te_list = []
    Y_te_list = []
    for i in range(1629):
        j = random.randint(0, 1628)

        X_tr_list.append(X_train[j])
        Y_tr_list.append(Y_train[j])
    for i in range(1629):
        j = random.randint(0, 1628)
        X_te_list.append(X_train[j])
        Y_te_list.append(Y_train[j])

    X_tr = np.array(X_tr_list)

    Y_tr = np.array(Y_tr_list)
    X_te = np.array(X_te_list)
    Y_te = np.array(Y_te_list)
    LPt.fit(X_tr,Y_tr)


    result = LPt.predict(X_te)
print(result)

#    BRt.BRCLF_clear()

LPt.LP_train_BR_train(result, Y_te)
###############################################
####训练阶段结束了
###############################################
BRt.BRCLF_clear()
BRt.BRCLF_test_clear()
Y_tr = np.array([])
X_te = np.array([])
Y_te = np.array([])
X_tr_list = []
Y_tr_list = []
X_te_list = []
Y_te_list = []
test_clf_num = 0
test_result = np.array([])
for i in range(788):
    j = random.randint(0, 787)

    X_tr_list.append(X_test[j])
    Y_tr_list.append(Y_test[j])


for h in range(3):

    star = test_clf_num
    end = test_clf_num + 14
    test_clf_num = test_clf_num + 14
    X_tr = np.array(X_tr_list)
    test_result = LPt.test_fit(X_tr,h)
real_label = np.array(Y_tr_list)
final_result = LPt.BR_test_BRC_test(test_result)
correct_matrix = (final_result == real_label)
num_correct = np.sum(correct_matrix)
print(real_label)
print(final_result)
eva = evaluate(final_result,real_label)
print(eva)
accuracy = num_correct / (788*14)
print(accuracy)
