from itertools import product
Map = list(map(lambda x: list(x), list(product([1, 0], repeat=3))))

for e in Map:
    e.append(0)
    e.insert(0,0)

result_di = {(0,0): "p",
             (0,1): "(1-p)",
             (1,0): "(1-q)",
             (1,1): "q"}

result_li = []

for e in Map:
    sub_result_di = {}
    sub_result_li = []
    for i in range(len(e)-1):
        sub_result_di[result_di[(e[i],e[i+1])]] = sub_result_di.get(result_di[(e[i],e[i+1])],0) + 1
    for key, value in sub_result_di.items():
        sub_result_li.append(f"{key}^{value}" if value > 1 else f"{key}")
    result_li.append(".".join(sub_result_li))

print(" + ".join(result_li))

