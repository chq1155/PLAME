import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Parameters
P_control = 1_000_000  # 招标控制价 (出价上限)
k1_values = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90]  # 随机抽取的k1值
n_friends = 3  # 我方朋友团队人数
n_opponents = 5  # 对手人数
n_random = 10  # 随机竞标者人数
sigma_random = 50_000  # 随机竞标者出价的标准差
sigma_opponent = 30_000  # 普通对手出价的标准差
epsilon = 5_000  # 成功的误差范围
p_smart = 0.6  # 狡猾对手比例（60%狡猾，40%普通）

# Generate random bids
def generate_bids(mean_price, num_bidders, sigma, P_control):
    """生成一个满足出价上限的价格集合"""
    bids = np.random.normal(mean_price, sigma, num_bidders)
    return np.round(np.clip(bids, 0, P_control), 2)  # 限制出价不能超过 P_control，并保留两位小数

# Generate bids for "smart" opponents
def generate_smart_opponent_bids(P_control, k1, k2, k3, random_bids, n_smart, sigma, epsilon):
    """生成狡猾对手的出价"""
    smart_bids = []
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, random_bids)  # 他们预测的基准价
    for _ in range(n_smart):
        # 狡猾对手的出价接近他们预测的基准价（加上一定的随机性）
        bid = np.random.normal(baseline_price, sigma)
        smart_bids.append(min(bid, P_control))  # 确保不超过 P_control
    return np.round(smart_bids, 2)  # 保留两位小数

# Calculate baseline price
def calculate_baseline_price(P_control, k1, k2, k3, all_bids):
    """计算基准价"""
    control_price_component = P_control * k1 * k2
    mean_bids_component = np.mean(all_bids) * k3
    return round(control_price_component + mean_bids_component, 2)  # 保留两位小数

# Success probability
def success_probability(friend_bids, baseline_price, epsilon):
    """计算我方中标的概率"""
    closest_bid = min(friend_bids, key=lambda x: abs(x - baseline_price))
    prob = norm.cdf(epsilon, loc=abs(closest_bid - baseline_price), scale=sigma_random)
    return prob

# Objective function for optimization
def objective_function(friend_bids, P_control, k1, k2, k3, opponent_bids, random_bids, epsilon):
    """目标函数，最小化负的中标概率（即最大化中标概率）"""
    # Combine all bids
    all_bids = np.concatenate([friend_bids, opponent_bids, random_bids])
    
    # Calculate baseline price
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, all_bids)
    
    # Calculate success probability
    prob = success_probability(friend_bids, baseline_price, epsilon)
    
    # Return negative probability because we are minimizing
    return -prob

# Optimization
def optimize_bids(P_control, k1_values, n_friends, n_opponents, n_random, sigma_opponent, sigma_random, epsilon, p_smart):
    """优化我方团队的出价策略"""
    # Step 1: Generate opponent and random bids
    mean_price = P_control * 0.95  # 假设大多数出价接近控制价的95%
    random_bids = generate_bids(mean_price, n_random, sigma_random, P_control)
    
    # Split opponents into "smart" and "normal"
    n_smart = int(p_smart * n_opponents)
    n_normal = n_opponents - n_smart
    
    # Randomly select k1, k2, k3
    k1 = np.random.choice(k1_values)
    k2 = np.random.uniform(0.3, 0.6)  # 确保 k3 >= 0.4
    k3 = 1 - k2
    
    # Generate bids for normal opponents
    normal_opponent_bids = generate_bids(mean_price, n_normal, sigma_opponent, P_control)
    
    # Generate bids for smart opponents
    smart_opponent_bids = generate_smart_opponent_bids(P_control, k1, k2, k3, random_bids, n_smart, sigma_opponent, epsilon)
    
    # Combine opponent bids
    opponent_bids = np.concatenate([normal_opponent_bids, smart_opponent_bids])
    
    # Step 2: Optimize friend bids
    initial_bids = generate_bids(mean_price, n_friends, sigma_random, P_control)  # 初始猜测
    bounds = [(0, P_control) for _ in range(n_friends)]  # 确保出价不超过 P_control
    result = minimize(
        objective_function,
        initial_bids,
        args=(P_control, k1, k2, k3, opponent_bids, random_bids, epsilon),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    # Step 3: Return results
    optimized_bids = np.round(result.x, 2)  # 保留两位小数
    baseline_price = calculate_baseline_price(P_control, k1, k2, k3, np.concatenate([optimized_bids, opponent_bids, random_bids]))
    success_prob = success_probability(optimized_bids, baseline_price, epsilon)
    
    return {
        "optimized_bids": optimized_bids,
        "baseline_price": baseline_price,
        "success_probability": round(success_prob, 2),
        "k1": round(k1, 2),
        "k2": round(k2, 2),
        "k3": round(k3, 2),
        "opponent_bids": np.round(opponent_bids, 2),
        "random_bids": np.round(random_bids, 2)
    }

# Run optimization
result = optimize_bids(P_control, k1_values, n_friends, n_opponents, n_random, sigma_opponent, sigma_random, epsilon, p_smart)

# Print results
print("Optimized Friend Bids:", result["optimized_bids"])
print("Opponent Bids:", result["opponent_bids"])
print("Random Bids:", result["random_bids"])
print("Baseline Price:", result["baseline_price"])
print("Success Probability:", result["success_probability"])
print("k1, k2, k3:", result["k1"], result["k2"], result["k3"])