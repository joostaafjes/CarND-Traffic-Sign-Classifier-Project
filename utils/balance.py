import numpy as np


def resample_with_max_entries(x, y, max):
    x_resampled = np.array([])
    y_resampled = np.array([])

    cnt = np.zeros(len(y))

    for index in range(len(y)):
        if cnt[y[index]] < max:
            x_resampled = np.append(x_resampled, x[index, :, :, :])
            y_resampled = np.append(y_resampled, y[index])
        cnt[y[index]] += 1

    x_resampled = np.reshape(x_resampled, (-1, x.shape[1], x.shape[2], x.shape[3]))

    return x_resampled, y_resampled

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

new_x, new_y = resample_with_max_entries(x, y, 1)

print(new_x.shape, new_y.shape)
print(new_x)
print(new_y)
print('-------------------')
print(np.concatenate((x, new_x)))


