import numpy as np
from HelperFunctions import proj
from main import TRADING_FREQ
def ftl_average(history):
    # look at the history and choose the best in terms of average profit
    # history includes the past data
    # t is the current moment
    stock_profit_prods = []
    for i in range(history.shape[1]):
        # look at the ratios
        # look at the ratios
        current_stock = history[:, i]
        num_past_days = len(current_stock)
        sell_days = np.arange(start=TRADING_FREQ, stop=num_past_days, step=TRADING_FREQ)
        buy_days = np.arange(start=0, stop=num_past_days-1, step=TRADING_FREQ)
        stock_sell_values = current_stock[sell_days]
        stock_buy_values = current_stock[buy_days]
        ratio = np.divide(stock_sell_values, stock_buy_values)
        ratio_multip = np.mean(ratio)
        stock_profit_prods.append(ratio_multip)
    # now choose the best
    argmax_ind = np.argmax(np.array(stock_profit_prods))
    output = np.zeros(history.shape[1])
    output[argmax_ind] = 1
    return output

def ftl_max(history):
    # look at the history and choose the best in terms of number of times resulted in profit
    # history includes the past data
    # t is the current moment
    stock_profit_prods = []
    for i in range(history.shape[1]):
        # look at the ratios
        # look at the ratios
        current_stock = history[:, i]
        num_past_days = len(current_stock)
        sell_days = np.arange(start=TRADING_FREQ, stop=num_past_days, step=TRADING_FREQ)
        buy_days = np.arange(start=0, stop=num_past_days-1, step=TRADING_FREQ)
        stock_sell_values = current_stock[sell_days]
        stock_buy_values = current_stock[buy_days]
        ratio = np.divide(stock_sell_values, stock_buy_values)
        # take the ones above +1 which are perfect for you
        count_of_goods = np.sum(ratio > 1)
        stock_profit_prods.append(count_of_goods)
    # now choose the best
    argmax_ind = np.argmax(np.array(stock_profit_prods))
    output = np.zeros(history.shape[1])
    output[argmax_ind] = 1
    return output

def ftl_sum(history):
    # look at the history and choose the best
    # history includes the past data
    # t is the current moment
    stock_profit_prods = []
    for i in range(history.shape[1]):
        # look at the ratios
        # look at the ratios
        current_stock = history[:, i]
        num_past_days = len(current_stock)
        sell_days = np.arange(start=TRADING_FREQ, stop=num_past_days, step=TRADING_FREQ)
        buy_days = np.arange(start=0, stop=num_past_days-1, step=TRADING_FREQ)
        stock_sell_values = current_stock[sell_days]
        stock_buy_values = current_stock[buy_days]
        ratio = np.divide(stock_sell_values, stock_buy_values)
        # take the ones above +1 which are perfect for you
        ratio_sum = np.sum(ratio)
        stock_profit_prods.append(ratio_sum)
    # now choose the best
    argmax_ind = np.argmax(np.array(stock_profit_prods))
    output = np.zeros(history.shape[1])
    output[argmax_ind] = 1
    return output

def ftpl(history, eta):
    stock_profit_prods = []
    for i in range(history.shape[1]):
        current_stock = history[:, i]
        num_past_days = len(current_stock)
        sell_days = np.arange(start=TRADING_FREQ, stop=num_past_days, step=TRADING_FREQ)
        buy_days = np.arange(start=0, stop=num_past_days - 1, step=TRADING_FREQ)
        stock_sell_values = current_stock[sell_days]
        stock_buy_values = current_stock[buy_days]
        ratio = np.divide(stock_sell_values, stock_buy_values)
        # introduce noise using the eta
        noise_term = np.random.exponential(scale=1/eta, size=None)
        # add this noise to our profit ratio term
        ratio_sum = np.sum(ratio + noise_term)
        stock_profit_prods.append(ratio_sum)
    # now choose the best
    argmax_ind = np.argmax(np.array(stock_profit_prods))
    output = np.zeros(history.shape[1])
    output[argmax_ind] = 1
    return output

def mwa(prev_value, most_recent_value, mwa_weights, mwa_losses, eta):
    for i in range(len(mwa_weights)):
        ratio = most_recent_value[i] / prev_value[i]
        loss = 0 if ratio>1 else (1 - ratio)
        # introduce noise using the eta
        mwa_losses[i] += loss
        mwa_weights[i] *= np.exp(-eta * loss)
    # now find the probability of selection of each stock
    probs = mwa_weights /(np.sum(mwa_weights))
    argmax_ind = np.random.choice(len(mwa_weights), size=1, p=probs)
    output = np.zeros(len(mwa_weights))
    output[argmax_ind] = 1
    return output, mwa_weights, mwa_losses

def ogd(ratio, balance, ogd_weights, eta):
    # loss is defined as w_t*balance*(loss or profit ratio)
    grad = balance * ratio
    unprojected_weights = ogd_weights + eta * grad
    # now project it
    ogd_weights = proj(np.ones(len(ogd_weights)), unprojected_weights)
    return ogd_weights, ogd_weights

def ucb_stocks(history, sample_means, arm_counts, delta=0.01):
    output = np.zeros(history.shape[1])
    if history.shape[0] <= 5:  # t = 1
        output[history.shape[0] - 1] = 1
    else:
        ucb_factor = sample_means + np.sqrt(np.log(1/delta) / (arm_counts * 2))
        output[np.argmax(ucb_factor)] = 1
        arm_counts[np.argmax(ucb_factor)] += 1
    return output, arm_counts

def ucb_agents(t, sample_means, arm_counts, delta=0.01):
    output = np.zeros(len(sample_means))
    if t <= len(sample_means):  # t = 1
        output[t - 1] = 1
    else:
        ucb_factor = sample_means + np.sqrt(np.log(1/delta) / (arm_counts * 2))
        output[np.argmax(ucb_factor)] = 1
        arm_counts[np.argmax(ucb_factor)] += 1
    return output, arm_counts

def mwa_agents(agent_profits, mwa_weights, mwa_losses, eta):
    for i in range(len(mwa_weights)):  # over agents
        ratio = agent_profits[i]
        loss = 0 if ratio>1 else (1 - ratio)
        # introduce noise using the eta
        mwa_losses[i] += loss
        mwa_weights[i] *= np.exp(-eta * loss)
    # now find the probability of selection of each stock
    probs = mwa_weights /(np.sum(mwa_weights))
    argmax_ind = np.random.choice(len(mwa_weights), size=1, p=probs)
    output = np.zeros(len(mwa_weights))
    output[argmax_ind] = 1
    return output, mwa_weights, mwa_losses

def mwa_agents_plusplus(agent_profits, mwa_weights, mwa_losses, eta):
    for i in range(len(mwa_weights)):  # over agents
        ratio = agent_profits[i]
        loss = 0 if ratio>1 else (1 - ratio)
        # introduce noise using the eta
        mwa_losses[i] += loss
        mwa_weights[i] *= np.exp(-eta * loss)
    # now find the probability of selection of each stock
    probs = mwa_weights /(np.sum(mwa_weights))
    return probs, mwa_weights, mwa_losses
