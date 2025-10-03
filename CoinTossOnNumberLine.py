import math

def coin_toss_problem(total_coins, final_position):
    heads = (total_coins + final_position)/2
    if heads != int(heads) or abs(final_position) > total_coins:
        return "Not possible"
    heads = int(heads)
    tails = total_coins - heads
    T = max(heads,tails)
    t = min(heads,tails)
    numerator = 1
    denominator = 1
    for i in range(total_coins,T,-1):
        numerator *= i
    for i in range(2,t+1,1):
        denominator *= i
    return f"{int(numerator/denominator)}*p^{heads}*(1-p)^{tails}"

    
if __name__ == "__main__":
    print(coin_toss_problem(10,0))
    print(coin_toss_problem(10,5))
    print(coin_toss_problem(20,10))
    print(coin_toss_problem(15,-5))