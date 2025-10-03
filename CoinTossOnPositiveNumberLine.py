import math
from itertools import product

moves = list(product([0,1], repeat=10))
valid_moves = []

for E in moves:
    pos = 0
    for e in E:
        if e == 1 and pos < 3:
            pos += 1
        elif e == 0 and pos > 0:
            pos -= 1
    if pos == 2:
        valid_moves.append(E)

prob_di = {0: 'p',
             1: '(1-p)'}

result_di = {}

for E in valid_moves:
    sub_result_di = {'p':0,
                     '(1-p)':1}
    sub_result_li = []
    for e in E:
        sub_result_di[prob_di[e]] = sub_result_di.get(prob_di[e],0) + 1
    for key, value in sub_result_di.items():
        sub_result_li.append(f"{key}^{value}" if value > 1 else f"{key}")
    result_di["".join(sub_result_li)] = result_di.get("".join(sub_result_li),0) + 1

result_li = []
for key, value in result_di.items():
    result_li.append(f"{value}{key}" if value != 1 else f"{key}")

print(" + ".join(result_li))