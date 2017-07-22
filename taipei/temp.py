import numpy as np

repeated_data = np.load('all_data.npy')
sparse_label_data = np.load('all_label.npy')

train_data, test_data = np.split(
    repeated_data, [repeated_data.shape[0] * 9 // 10])
np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
print('data saved')
train_label, test_label = np.split(
    sparse_label_data, [sparse_label_data.shape[0] * 9 // 10])
np.save('train_label.npy', train_label)
np.save('test_label.npy', test_label)
print('label saved')