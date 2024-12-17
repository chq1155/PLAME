import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm import tqdm
# 通用参数设置
P_control = 1_000_000  # 招标控制价 (出价上限)
n_friends = 3  # 我方团队人数
n_opponents = 5  # 对手人数
n_random = 10  # 随机竞标者人数
sigma_random = 50_000  # 随机竞标者出价的标准差
sigma_opponent = 30_000  # 普通对手出价的标准差
epsilon = 5_000  # 成功的误差范围

# 生成随机对手和随机竞标者出价
def generate_bids(mean_price, num_bidders, sigma, P_control):
    """生成一个满足出价上限的价格集合"""
    bids = np.random.normal(mean_price, sigma, num_bidders)
    return np.round(np.clip(bids, 0, P_control), 2)

# 计算基准价
def calculate_baseline_price(P_control, k1, k2, k3, all_bids):
    """计算基准价"""
    control_price_component = P_control * k1 * k2
    mean_bids_component = np.mean(all_bids) * k3
    return round(control_price_component + mean_bids_component, 2)

# 计算中标概率
def success_probability(friend_bids, baseline_price, opponent_bids, random_bids, epsilon):
    """
    计算综合中标概率：
    1. 基于正态分布的接近程度概率
    2. 基于排名的相对概率
    将两者结合得到最终概率
    """
    # 获取我方最接近baseline_price的出价
    closest_friend_bid = min(friend_bids, key=lambda x: abs(x - baseline_price))
    
    # 1. 计算基于正态分布的概率部分
    difference = closest_friend_bid - baseline_price
    percentage_diff = (difference / baseline_price) * 100
    
    if difference > 0:  # 高于基准价
        deduction = percentage_diff * 1.0  # 每高1%扣1分
    else:  # 低于基准价
        deduction = abs(percentage_diff) * 0.5  # 每低1%扣0.5分
    
    # 使用正态分布计算概率
    normal_prob = norm.cdf(epsilon, loc=deduction, scale=sigma_random)
    
    # 2. 计算基于排名的概率部分
    all_competitor_bids = np.concatenate([opponent_bids, random_bids])
    
    def calculate_deduction(bid):
        diff = bid - baseline_price
        perc_diff = (diff / baseline_price) * 100
        return perc_diff * 1.0 if diff > 0 else abs(perc_diff) * 0.5
    
    # 计算我方扣分
    our_deduction = calculate_deduction(closest_friend_bid)
    
    # 计算所有竞争者扣分
    competitor_deductions = np.array([calculate_deduction(bid) for bid in all_competitor_bids])
    
    # 计算排名概率
    win_count = np.sum(our_deduction < competitor_deductions)
    tie_count = np.sum(our_deduction == competitor_deductions)
    ranking_prob = (win_count + 0.5 * tie_count) / len(all_competitor_bids)
    
    # 3. 结合两种概率
    # 使用几何平均数来结合两个概率，这样只有两个概率都较高时，最终概率才会较高
    final_prob = np.sqrt(normal_prob * ranking_prob)
    
    return final_prob

# 优化目标函数
def objective_function(friend_bids, P_control, k1, k2, k3, opponent_bids, random_bids, epsilon):
    """目标函数，最小化负的中标概率（即最大化中标概率）"""
    # 合并所有出价
    all_bids = np.concatenate([friend_bids, opponent_bids, random_bids])
    
    # 计算基准价
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, all_bids)
    
    # 计算中标概率
    prob = success_probability(friend_bids, baseline_price, opponent_bids, random_bids, epsilon)
    
    # 返回负概率，因为我们要最大化概率 (最小化负值)
    return -prob

# 修改后的优化函数
def adjust_bids_to_match_baseline(optimized_bids, baseline_price):
    """
    调整优化后的出价，使第一个出价最接近基准价
    """
    # 选择最接近 baseline_price 的出价
    closest_index = np.argmin(np.abs(optimized_bids - baseline_price))
    
    # 交换第一个出价和最接近基准价的出价
    optimized_bids[0], optimized_bids[closest_index] = optimized_bids[closest_index], optimized_bids[0]
    return optimized_bids

def version_1_given_k_values(P_control, k1, k2, k3, n_friends, n_opponents, n_random, sigma_opponent, sigma_random, epsilon):
    """给定 k1, k2, k3，模拟过程并计算胜率"""
    mean_price = P_control * 0.95
    opponent_bids = generate_bids(mean_price, n_opponents, sigma_opponent, P_control)
    random_bids = generate_bids(mean_price, n_random, sigma_random, P_control)
    
    # 优化我方团队的出价
    initial_bids = generate_bids(mean_price, n_friends, sigma_random, P_control)
    bounds = [(0, P_control) for _ in range(n_friends)]
    
    result = minimize(
        objective_function,
        initial_bids,
        args=(P_control, k1, k2, k3, opponent_bids, random_bids, epsilon),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    optimized_bids = np.round(result.x, 2)
    all_bids = np.concatenate([optimized_bids, opponent_bids, random_bids])
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, all_bids)
    optimized_bids = adjust_bids_to_match_baseline(optimized_bids, baseline_price)
    success_prob = success_probability(optimized_bids, baseline_price, opponent_bids, random_bids, epsilon)
    
    return {
        "optimized_bids": optimized_bids,
        "baseline_price": baseline_price,
        "success_probability": round(success_prob, 2),
        "opponent_bids": opponent_bids,
        "random_bids": random_bids
    }

def version_2_given_opponent_bids(P_control, opponent_bids, n_friends, n_random, sigma_random, epsilon):
    """给定对手的出价，模拟过程并计算胜率"""
    mean_price = P_control * 0.95
    random_bids = generate_bids(mean_price, n_random, sigma_random, P_control)
    
    # 随机生成 k1, k2, k3
    k1 = np.random.choice([0.95, 0.94, 0.93, 0.92, 0.91, 0.90])
    k2 = np.random.uniform(0.3, 0.6)
    k3 = 1 - k2
    
    # 优化我方团队的出价
    initial_bids = generate_bids(mean_price, n_friends, sigma_random, P_control)
    bounds = [(0, P_control) for _ in range(n_friends)]
    
    result = minimize(
        objective_function,
        initial_bids,
        args=(P_control, k1, k2, k3, opponent_bids, random_bids, epsilon),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    optimized_bids = np.round(result.x, 2)
    all_bids = np.concatenate([optimized_bids, opponent_bids, random_bids])
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, all_bids)
    optimized_bids = adjust_bids_to_match_baseline(optimized_bids, baseline_price)
    success_prob = success_probability(optimized_bids, baseline_price, opponent_bids, random_bids, epsilon)
    
    return {
        "optimized_bids": optimized_bids,
        "baseline_price": baseline_price,
        "success_probability": round(success_prob, 2),
        "k1": round(k1, 2),
        "k2": round(k2, 2),
        "k3": round(k3, 2),
        "random_bids": random_bids
    }

def version_3_given_opponent_and_k(P_control, opponent_bids, k1, k2, k3, n_friends, n_random, sigma_random, epsilon):
    """给定对手出价和 k1, k2, k3，模拟过程并计算胜率"""
    mean_price = P_control * 0.95
    random_bids = generate_bids(mean_price, n_random, sigma_random, P_control)
    
    # 优化我方团队的出价
    initial_bids = generate_bids(mean_price, n_friends, sigma_random, P_control)
    bounds = [(0, P_control) for _ in range(n_friends)]
    
    result = minimize(
        objective_function,
        initial_bids,
        args=(P_control, k1, k2, k3, opponent_bids, random_bids, epsilon),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    optimized_bids = np.round(result.x, 2)
    all_bids = np.concatenate([optimized_bids, opponent_bids, random_bids])
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, all_bids)
    optimized_bids = adjust_bids_to_match_baseline(optimized_bids, baseline_price)
    success_prob = success_probability(optimized_bids, baseline_price, opponent_bids, random_bids, epsilon)
    
    return {
        "optimized_bids": optimized_bids,
        "baseline_price": baseline_price,
        "success_probability": round(success_prob, 2),
        "random_bids": random_bids
    }

def version_4_randomized(P_control, n_friends, n_opponents, n_random, sigma_opponent, sigma_random, epsilon):
    """随机生成对手出价和 k1, k2, k3，模拟过程并计算胜率"""
    # Step 1: 随机生成对手和随机竞标者的出价
    mean_price = P_control * 0.95  # 假设大多数出价接近控制价的 95%
    opponent_bids = generate_bids(mean_price, n_opponents, sigma_opponent, P_control)
    random_bids = generate_bids(mean_price, n_random, sigma_random, P_control)
    
    # Step 2: 随机生成 k1, k2, k3
    k1 = np.random.choice([0.95, 0.94, 0.93, 0.92, 0.91, 0.90])
    k2 = np.random.uniform(0.3, 0.6)  # 确保 k2 在合理范围内
    k3 = 1 - k2  # k1 + k2 + k3 = 1
    
    # Step 3: 优化我方团队的出价
    initial_bids = generate_bids(mean_price, n_friends, sigma_random, P_control)  # 初始猜测
    bounds = [(0, P_control) for _ in range(n_friends)]  # 确保出价不超过 P_control
    
    result = minimize(
        objective_function,
        initial_bids,
        args=(P_control, k1, k2, k3, opponent_bids, random_bids, epsilon),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    # Step 4: 计算结果
    optimized_bids = np.round(result.x, 2)  # 我方团队优化后的出价
    all_bids = np.concatenate([optimized_bids, opponent_bids, random_bids])
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, all_bids)  # 基准价
    optimized_bids = adjust_bids_to_match_baseline(optimized_bids, baseline_price)
    success_prob = success_probability(optimized_bids, baseline_price, opponent_bids, random_bids, epsilon)  # 中标概率
    
    return {
        "optimized_bids": optimized_bids,  # 我方团队优化后的出价
        "baseline_price": baseline_price,  # 基准价
        "success_probability": round(success_prob, 2),  # 中标概率
        "k1": round(k1, 2),  # 随机生成的 k1
        "k2": round(k2, 2),  # 随机生成的 k2
        "k3": round(k3, 2),  # 随机生成的 k3
        "opponent_bids": opponent_bids,  # 对手的出价
        "random_bids": random_bids  # 随机竞标者的出价
    }

def version_5_randomized(P_control, n_friends, n_opponents, n_random, sigma_opponent, sigma_random, epsilon, num_trials=100):
    """纯随机应对所有可能的对手出价和 k1, k2, k3，寻找最优策略"""
    best_strategy = None
    max_success_prob = 0
    
    for _ in tqdm(range(num_trials)):
        # 随机生成 k1, k2, k3
        k1 = np.random.choice([0.95, 0.94, 0.93, 0.92, 0.91, 0.90])
        k2 = np.random.uniform(0.3, 0.6)
        k3 = 1 - k2
        
        # 随机生成对手和随机竞标者的出价
        mean_price = P_control * 0.95
        opponent_bids = generate_bids(mean_price, n_opponents, sigma_opponent, P_control)
        random_bids = generate_bids(mean_price, n_random, sigma_random, P_control)
        
        # 优化我方团队的出价
        initial_bids = generate_bids(mean_price, n_friends, sigma_random, P_control)
        bounds = [(0, P_control) for _ in range(n_friends)]
        
        result = minimize(
            objective_function,
            initial_bids,
            args=(P_control, k1, k2, k3, opponent_bids, random_bids, epsilon),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        optimized_bids = np.round(result.x, 2)
        all_bids = np.concatenate([optimized_bids, opponent_bids, random_bids])
        baseline_price = calculate_baseline_price(P_control, k1, k2, k3, all_bids)
        optimized_bids = adjust_bids_to_match_baseline(optimized_bids, baseline_price)
        success_prob = success_probability(optimized_bids, baseline_price, opponent_bids, random_bids, epsilon)
        
        # 更新最优策略
        if success_prob > max_success_prob:
            max_success_prob = success_prob
            best_strategy = {
                "optimized_bids": optimized_bids,
                "baseline_price": baseline_price,
                "success_probability": round(success_prob, 2),
                "k1": round(k1, 2),
                "k2": round(k2, 2),
                "k3": round(k3, 2),
                "opponent_bids": opponent_bids,
                "random_bids": random_bids
            }
    
    return best_strategy

def version_6_given_friends_bids(P_control, friend_bids, n_opponents, n_random, sigma_opponent, sigma_random, epsilon):
    """
    给定我方团队的出价，默认第一个是期望最大的报价，模拟过程并计算胜率。
    """
    mean_price = P_control * 0.95  # 假设大多数出价接近控制价的 95%
    
    # 随机生成对手和随机竞标者的出价
    opponent_bids = generate_bids(mean_price, n_opponents, sigma_opponent, P_control)
    random_bids = generate_bids(mean_price, n_random, sigma_random, P_control)
    
    # 随机生成 k1, k2, k3
    k1 = np.random.choice([0.95, 0.94, 0.93, 0.92, 0.91, 0.90])
    k2 = np.random.uniform(0.3, 0.6)
    k3 = 1 - k2
    
    # 合并所有出价
    all_bids = np.concatenate([friend_bids, opponent_bids, random_bids])
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, all_bids)
    
    # 计算中标概率
    success_prob = success_probability(friend_bids, baseline_price, opponent_bids, random_bids, epsilon)
    
    return {
        "friend_bids": friend_bids,  # 我方团队的出价
        "baseline_price": baseline_price,  # 基准价
        "success_probability": round(success_prob, 2),  # 中标概率
        "k1": round(k1, 2),  # 随机生成的 k1
        "k2": round(k2, 2),  # 随机生成的 k2
        "k3": round(k3, 2),  # 随机生成的 k3
        "opponent_bids": opponent_bids,  # 对手的出价
        "random_bids": random_bids  # 随机竞标者的出价
    }

# 主程序
def main():
    print("选择版本：")
    print("1. 给定 k1, k2, k3")
    print("2. 给定对手出价")
    print("3. 给定对手出价和 k1, k2, k3")
    print("4. 随机生成对手出价和 k1, k2, k3")
    print("5. 随机模拟多次，寻找最佳策略")
    print("6. 给定我方团队的出价")
    choice = int(input("请输入版本号（1/2/3/4/5/6）："))
    
    if choice == 1:
        k1 = float(input("请输入 k1 (如 0.95)："))
        k2 = float(input("请输入 k2 (如 0.4)："))
        k3 = 1 - k2
        n_friends = int(input("请输入我方团队人数："))
        n_opponents = int(input("请输入对手人数："))
        n_random = int(input("请输入随机竞标者人数："))
        result = version_1_given_k_values(P_control, k1, k2, k3, n_friends, n_opponents, n_random, sigma_opponent, sigma_random, epsilon)
    
    elif choice == 2:
        n_friends = int(input("请输入我方团队人数："))
        n_random = int(input("请输入随机竞标者人数："))
        opponent_bids = np.array([float(x) for x in input("请输入对手出价列表（用空格分隔）：").split()])
        result = version_2_given_opponent_bids(P_control, opponent_bids, n_friends, n_random, sigma_random, epsilon)
    
    elif choice == 3:
        k1 = float(input("请输入 k1 (如 0.95)："))
        k2 = float(input("请输入 k2 (如 0.4)："))
        k3 = 1 - k2
        n_friends = int(input("请输入我方团队人数："))
        n_random = int(input("请输入随机竞标者人数："))
        opponent_bids = np.array([float(x) for x in input("请输入对手出价列表（用空格分隔）：").split()])
        result = version_3_given_opponent_and_k(P_control, opponent_bids, k1, k2, k3, n_friends, n_random, sigma_random, epsilon)
    
    elif choice == 4:
        n_friends = int(input("请输入我方团队人数："))
        n_opponents = int(input("请输入对手人数："))
        n_random = int(input("请输入随机竞标者人数："))
        result = version_4_randomized(P_control, n_friends, n_opponents, n_random, sigma_opponent, sigma_random, epsilon)
    
    elif choice == 5:
        n_friends = int(input("请输入我方团队人数："))
        n_opponents = int(input("请输入对手人数："))
        n_random = int(input("请输入随机竞标者人数："))
        num_trials = int(input("请输入模拟次数："))
        result = version_5_randomized(P_control, n_friends, n_opponents, n_random, sigma_opponent, sigma_random, epsilon, num_trials)

    elif choice == 6:
        n_opponents = int(input("请输入对手人数："))
        n_random = int(input("请输入随机竞标者人数："))
        friend_bids = np.array([float(x) for x in input("请输入我方团队的报价列表（用空格分隔）：").split()])
        result = version_6_given_friends_bids(P_control, friend_bids, n_opponents, n_random, sigma_opponent, sigma_random, epsilon)
    
    else:
        print("无效的版本选择！")
        return
    
    print("结果：")
    print(result)

# 启动程序
if __name__ == "__main__":
    main()