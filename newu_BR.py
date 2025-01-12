import numpy as np

from newData import *
from newBR import *

data = data()
data.readData()

rt = np.array([])
rte = np.array([])
X_train = data.getoriginTrainData()
Y_train = data.getoriginTrainLabel()
X_test = data.getoriginTestData()
Y_test = data.getoriginTestLabel()
num = 0
for i in range(3):
    BR = BRclass()
    BR.BRC_train(X_train,Y_train)
    tr_result = BR.BRC_test(X_train)
    BR.clear()
    te_result = BR.BRC_test(X_test)
    if num == 0:
        rt = tr_result
        rte = te_result
        num = 1
    else:
        rt = np.hstack([rt, tr_result])
        rte = np.hstack([rte, te_result])

BRE = BRclass()
BRE.train_L2(rt,Y_train)
result = BRE.test_L2(rte)
eva = evaluate(result, Y_test)
print(eva)