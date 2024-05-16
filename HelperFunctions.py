import numpy as np
from alpha_vantage.timeseries import TimeSeries
import time
from main import TRADING_FREQ

def pull_daily_time_series_alpha_vantage(alpha_vantage_api_key, equity_name,output_size='full'):
    """
    Pulls daily time series data from alpha vantage
    :param alpha_vantage_api_key: API key taken from alpha vantage
    :param equity_name: which equity that the data will be taken from
    :param output_size: compact or fullsize
    :return: data, metadata
    """
    ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
    data, meta_data = ts.get_daily_adjusted(equity_name, outputsize=output_size)
    data.reset_index(level=0, inplace=True) # make the date index a column
    data['timestamp'] = data['date'].apply(lambda x: int(time.mktime(time.strptime(str(x), '%Y-%m-%d %H:%M:%S'))))
    data = data[['timestamp','1. open', '2. high', '3. low', '4. close', '5. adjusted close', '6. volume']]
    return data,meta_data

#used and up to date
def pull_daily_stocks(alpha_vantage_api_key, stocks, output_size = 'full', sample_size = 2000):
    """
    For different stocks, pulls them from the API and puts into equity_dataset
    :param alpha_vantage_api_key: API key
    :param stocks: which stocks do you want to keep track and invest
    :param output_size: full or compact - compact for last 100 samples
    :param sample_size: stocks have different amount of historical data - sample size will be used for fixing the length
    No return - But puts data into equity_dataset
    """
    close_values_dataset = {}
    out = np.zeros((sample_size, len(stocks)))
    count = 0
    for stock in stocks:
        data, meta_data = pull_daily_time_series_alpha_vantage(alpha_vantage_api_key, stock,output_size=output_size)
        closed_values_data = data['5. adjusted close']
        closed_values_data = closed_values_data[:sample_size]
        close_values_dataset[stock] = closed_values_data
        # reverse the dataset so that closer days will be at the end
        # this will intuitively ease the training procedure
        close_values_dataset[stock] = close_values_dataset[stock][::-1].reset_index(drop=True).values
        out[:, count] = close_values_dataset[stock]
        count+=1
    return out

def proj(a, y):
    # source http://www.mcduplessis.com/index.php/2016/08/22/fast-projection-onto-a-simplex-python
    l = y / a
    idx = np.argsort(l)
    d = len(l)

    evalpL = lambda k: np.sum(a[idx[k:]] * (y[idx[k:]] - l[idx[k]] * a[idx[k:]])) - 1

    def bisectsearch():
        idxL, idxH = 0, d - 1
        L = evalpL(idxL)
        H = evalpL(idxH)

        if L < 0:
            return idxL

        while (idxH - idxL) > 1:
            iMid = int((idxL + idxH) / 2)
            M = evalpL(iMid)

            if M > 0:
                idxL, L = iMid, M
            else:
                idxH, H = iMid, M

        return idxH

    k = bisectsearch()
    lam = (np.sum(a[idx[k:]] * y[idx[k:]]) - 1) / np.sum(a[idx[k:]])

    x = np.maximum(0, y - lam * a)
    return x

def get_regret_needed_terms(stock_values):
    # needed for regret calculation on stocks
    num_days = stock_values.shape[0]
    stock_means = np.zeros(stock_values.shape[1])
    stock_cumulative_ratios = np.zeros(stock_values.shape[1])

    sell_days = np.arange(start=TRADING_FREQ, stop=num_days, step=TRADING_FREQ)
    buy_days = np.arange(start=0, stop=num_days - 1, step=TRADING_FREQ)
    for i in range(stock_values.shape[1]): # traverse stocks
        stock_data = stock_values[:, i]


        profit_loss_ratios = np.divide(stock_data[sell_days], stock_data[buy_days])

        stock_means[i] = np.mean(profit_loss_ratios)
        stock_cumulative_ratios[i] = np.prod(profit_loss_ratios)

    best_arm_in_mean = np.argmax(stock_means)
    best_arms_mean = stock_means[best_arm_in_mean]
    best_stock_in_hindsight = np.argmax(stock_cumulative_ratios)

    ftl_hindsight_instantaneous_pro_loss = np.divide(stock_values[sell_days, best_stock_in_hindsight],
                                                                      stock_values[buy_days, best_stock_in_hindsight])
    ftl_hindsight_total_pro_loss = np.sum(ftl_hindsight_instantaneous_pro_loss)

    ucb_hindsight_total_pro_loss = best_arms_mean * (num_days-1)
    ucb_hindsight_instantaneous_pro_loss = np.divide(stock_values[sell_days, best_arm_in_mean],
                                                                      stock_values[buy_days, best_arm_in_mean])

    regret_needed_dict = {'hindsight_inst': ftl_hindsight_instantaneous_pro_loss,
                          'hindsight_total': ftl_hindsight_total_pro_loss,
                          'best_arm_inst': ucb_hindsight_instantaneous_pro_loss,
                          'best_arm_total': ucb_hindsight_total_pro_loss}
    return regret_needed_dict

def get_regret_needed_terms_for_agents(agent_pro_losses, num_days=1000):
    # needed for regret calculation on stocks
    agent_means = np.zeros(len(agent_pro_losses))
    agent_cumulative_ratios = np.zeros(len(agent_pro_losses))

    for i in range(len(agent_pro_losses)): # traverse agents
        agent_pro_loss = agent_pro_losses[i]

        agent_means[i] = np.mean(agent_pro_loss)
        agent_cumulative_ratios[i] = np.prod(agent_pro_loss)

    best_arm_in_mean = np.argmax(agent_means)
    best_arms_mean = agent_means[best_arm_in_mean]
    best_agent_in_hindsight = np.argmax(agent_cumulative_ratios)

    ftl_hindsight_instantaneous_pro_loss = agent_pro_losses[best_agent_in_hindsight]
    ftl_hindsight_total_pro_loss = np.sum(ftl_hindsight_instantaneous_pro_loss)

    ucb_hindsight_total_pro_loss = best_arms_mean * (num_days-1)
    ucb_hindsight_instantaneous_pro_loss = agent_pro_losses[best_arm_in_mean]

    regret_needed_dict = {'hindsight_inst': ftl_hindsight_instantaneous_pro_loss,
                          'hindsight_total': ftl_hindsight_total_pro_loss,
                          'best_arm_inst': ucb_hindsight_instantaneous_pro_loss,
                          'best_arm_total': ucb_hindsight_total_pro_loss}
    return regret_needed_dict

def calculate_regret(best_dict, alg, mode='hindsight'):
    if mode == 'hindsight':
        inst_pro_loss = best_dict['hindsight_inst']
        total_pro_loss = best_dict['hindsight_total']
    elif mode == 'bandit':
        inst_pro_loss = best_dict['best_arm_inst']
        total_pro_loss = best_dict['best_arm_total']

    # calculate algoritma's
    alg_inst_pro_loss = alg.all_profit_losses
    alg_total_pro_loss = np.sum(alg_inst_pro_loss)

    inst_regret = alg_inst_pro_loss - inst_pro_loss
    total_regret = alg_total_pro_loss  - total_pro_loss
    return inst_regret, total_regret