import numpy as np

# 初始化 X_tr 为一个空的 NumPy 数组
X_tr = np.array([])

# 模拟 X_train 的数据
X_train = np.array([10, 20, 30, 40])

# 动态添加值
for j in range(len(X_train)):
    X_tr = np.append(X_tr, X_train[j])  # 添加值
    print(f"当前 X_tr: {X_tr}")

print("最终 X_tr:", X_tr)
