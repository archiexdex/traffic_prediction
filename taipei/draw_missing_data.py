import json 
import matplotlib.pyplot as plt
import numpy as np

x = y = []

# with open('miss_len_data.json') as json_data:
#     data = json.load(json_data)
#     sum = 0
#     for i in data:
#         sum += data[i][0]  / 230976 * 100
#         print(i, data[i][0]  / 230976 * 100 )
#     print( sum / len(data) )

with open('fix_raw_data.json') as json_data:
    data = json.load(json_data)
    data = np.array(data['VP8GX00'])
    print(data.shape)
    x = list(map((lambda x: (x-data[0][4]) // 300), data[:,4]))
    y = list(map((lambda x: 0 if x[0] != 0 else 1), data))
    

# x = x[0:len(x)]
# y = y[0:len(y)]

plt.plot(x,y)
plt.ylim(0,2)
plt.show()
    