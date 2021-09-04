import random
import numpy as np
import re
def random_gen(num_samples, num_node):
    archs = []
    for i in range(num_samples):
        arch = []
        for j in range(num_node):
            arch += [np.random.randint(0, 7), np.random.randint(0, j+2), np.random.randint(0, 7), np.random.randint(0, j+2)]
        for j in range(num_node):
            arch += [np.random.randint(0, 7), np.random.randint(0, j+2), np.random.randint(0, 7), np.random.randint(0, j+2)]
        archs.append(arch)
    return np.array(archs)
# print(random_gen(2, 4))


# for tmp in range(1):
#     arch = []
#     for i in range(12):
#         arch += [random.randint(0, 3)]
#         for j in range(i):
#             arch += [random.randint(0,1)]

#     # print(arch)
#     test=re.sub("[|]|,",'',str(arch))
#     print(test)
#     num = 0
#     for i in range(12):
#         for j in range(i+1):
#             print(arch[num], end=' ')
#             num += 1
#         print('')
# # print(len(arch))