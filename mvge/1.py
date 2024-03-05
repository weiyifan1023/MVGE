import random
import numpy as np
from scipy.sparse import csr_matrix

list = [[20, 1], [16, 2], [10, 3], [5, 4]]
print(list)
random.shuffle(list)
print(list)

random.shuffle(list)
print(list)

f = lambda x: x == "1"
print(f('1'))

import numpy as np
from scipy.sparse import csr_matrix

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
a = csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
print("稀疏矩阵：", a)
csr_a = csr_matrix(a)
print("CSR矩阵：")
print(csr_a.data, csr_a.indptr, csr_a.indices)

import pdb

a = "aaa"
pdb.set_trace()
b = "bbb"
c = "ccc"
final = a + b + c
print(final)
print("graph: ")

diag_ele = np.full(3, 1)
graph = np.diag(diag_ele)
print(graph)
L = graph.tolist()
print(L)
if 2.5 > 361:
    print("nb")
else:
    print("not ")

max_node_num = 5
A = np.arange(95, 99).reshape(2, 2)  # 原始输入数组
print(A.size)
print("A:", A)
# 在数组A的边缘填充constant_values指定的数值
# （3,2）表示在A的第[0]轴填充（二维数组中，0轴表示行），即在0轴前面填充3个宽度的0，比如数组A中的95,96两个元素前面各填充了3个0；在后面填充2个0，比如数组A中的97,98两个元素后面各填充了2个0
# （2,3）表示在A的第[1]轴填充（二维数组中，1轴表示列），即在1轴前面填充2个宽度的0，后面填充3个宽度的0
B = np.pad(A, ((0, max_node_num - 2), (0, max_node_num - 2)), 'constant',
           constant_values=(0, 0))  # constant_values表示填充值，且(before，after)的填充值等于（0,0）
print("B:", B)
import torch

print("mask:")
x = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
mask = abs(x) == 1
print(mask)
variable_new = torch.tensor([200, 200, 200, 100])
# variable_new                              tensor([100, 100, 100])
y = x.masked_scatter_(mask, variable_new)
print(y)
# tensor([[ -2,  -1,   0],
#         [ -1,   0, 100],
#         [  0, 100, 100]])

input_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(0, len(input_list), 3):
    q,t,p = input_list[i:i+3]
    print(q,t,p)
end = input_list[-1]
print("end", end)
out_list = input_list[:end + 1]
print("output list:",out_list)
def fan(input_list):
    for i in range(len(input_list)):
        input_list[i] = 100
fan(input_list)
print(input_list)

str1 = "wyf"
str2 = "wyf"
if str1 in str2:
    print("lzy")

