from itertools import product

P_H = {True: 0.6, False: 0.4}
P_F = {True: 0.3, False: 0.7}

P_D_given_H = {
(True): 0.9,
(False): 0.3
}

P_E_given_HF = {
(True, True): 0.4,
(True, False): 0.8,
(False, True): 0.1,
(False, False): 0.6,
}

P_C_given_D = {
True: 0.2,
False: 0.7
}

P_W_given_DE = {
(True, True): 0.95,
(True, False): 0.7,
(False, True): 0.6,
(False, False): 0.3,
}

P_T_given_C = {
True: 0.9,
False: 0.1
}

def joint_probability(H, F, D, E, C, W, T):
    return (
    P_H[H], 
    P_F[F],
    P_D_given_H[H] if D else (1 - P_D_given_H[H]),
    P_E_given_HF[(H, F)] if E else (1 - P_E_given_HF[(H, F)]),
    P_C_given_D[D] if C else (1 - P_C_given_D[D]),
    P_W_given_DE[(D, E)] if W else (1 - P_W_given_DE[(D, E)]),
    P_T_given_C[C] if T else (1 - P_T_given_C[C]))

example = {
'H': True,
'F': False,
'D': True,
'E': True,
'C': False,
'W': True,
'T': False
}

jp = joint_probability(**example)
print(f"Joint Probability: {jp:.6f}")

def marginal_W_given_H(health_value):
    total = 0
    w_true = 0
    for (F, D, E, C, W, T) in product([True, False], repeat=6):
        jp = joint_probability(health_value, F, D, E, C, W, T)
        total += jp
        if W:
            w_true += jp
    return w_true / total if total > 0 else 0

print(f"P(Weight normal = True | Health conscious = True): {marginal_W_given_H(True):.4f}")
print(f"P(Weight normal = True | Health conscious = False): {marginal_W_given_H(False):.4f}")