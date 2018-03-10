import numpy as np


def get_from_category(x, y, category_id):
    x_filtered = np.array([])
    y_filtered = np.array([])

    for index in range(len(y)):
        if y[index] == category_id:
            x_filtered = np.append(x_filtered, x[index, :, :, :])
            y_filtered = np.append(y_filtered, y[index])

    x_filtered = np.reshape(x_filtered, (-1, x.shape[1], x.shape[2], x.shape[3]))

    return x_filtered, y_filtered

x = np.array([[[[0], [1]],
               [[2], [3]]],
              [[[1], [1]],
               [[2], [3]]],
              [[[2], [1]],
               [[2], [3]]],
              [[[3], [1]],
               [[2], [3]]]])
y = np.array([2, 3, 2, 3])

print(x.shape, y.shape)
print(x)
print(y)
print('-------------------')

# new_x, new_y = get_from_category(x, y, 2)

# new_x = np.select([y == 2], [x], default=-1)
# new_y = np.select([y == 2], [y], default=-1)
new_x = x[np.where(y == 2)]
new_y = y[np.where(y == 2)]

# print(new_x.shape, new_y.shape)
# print(new_y.shape)
print(new_x)
print(new_y)

