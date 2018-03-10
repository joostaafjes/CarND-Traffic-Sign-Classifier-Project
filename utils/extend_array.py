import numpy as np
from math import ceil

category_ids=[0, 1, 2]

X_train = np.array([[[[0], [1]],
               [[2], [3]]],
              [[[1], [1]],
               [[2], [3]]],
              [[[2], [1]],
               [[2], [3]]],
              [[[3], [1]],
               [[2], [3]]]])
y_train = np.array([2, 1, 2, 0])

max_cat_cnt = 5
for cat_id in category_ids:
    index = np.where(y_train == cat_id)
    count_cat = index[0].size
    if count_cat >= max_cat_cnt:
        continue
    selection_x = X_train[np.where(y_train == cat_id)]
    selection_y = y_train[np.where(y_train == cat_id)]
    print(selection_x)
    # multiply
    add_x = np.repeat(selection_x, ceil((max_cat_cnt - count_cat) / count_cat) , axis=0)
    add_y = np.repeat(selection_y, ceil((max_cat_cnt - count_cat) / count_cat) , axis=0)
    print(add_x)
    X_train = np.concatenate((X_train, add_x[:(max_cat_cnt - count_cat), :, :, :]))
    y_train = np.concatenate((y_train, add_y[:(max_cat_cnt - count_cat)]))
    print("Cat %d" % cat_id)

print(X_train.shape, y_train.shape)
print('------------')
print((X_train))
print('------------')
print((y_train))
