# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from HelperFunctions import *
from AgentClass import Agent
from VisualizationFunctions import *
from Models import *

TRADING_FREQ = 1  # we are opening and closing position every day
FTPL_LOW_NOISE_COFF = 0.2
FTPL_INTERMEDIATE_COFF = 0.75
FTPL_HIGH_COFF = 2.5

if __name__ == '__main__':
    alpha_vantage_api_key = 'something'
    output_format = 'pandas'
    simulation_size = 1000
    balance = 10000
    # determine which stocks you will be using
    stocks = ['AAPL','GOOGL','GOLD','SBUX','MSFT'] # limited to 5 calls in alphavantage api
    stock_values = pull_daily_stocks(alpha_vantage_api_key, stocks, output_size = 'full',sample_size=simulation_size)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
    regret_needed_dict = get_regret_needed_terms(stock_values)
    experimental_total_regrets = []
    experimental_inst_regrets = []

    # Agent 1
    ftl_average_agent = Agent(stocks, stock_values, ftl_average, alg_name='ftl',
                              simulation_size=simulation_size, balance=balance)
    ftl_average_agent.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, ftl_average_agent, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # visualize_profit_losses(ftl_average_agent.stock_profit_losses)
    # Agent 2
    ftl_max_agent = Agent(stocks, stock_values, ftl_max, alg_name='ftl',
                          simulation_size=simulation_size, balance=balance)
    ftl_max_agent.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, ftl_max_agent, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # Agent 3
    ftpl_low_noise = Agent(stocks, stock_values, ftpl, alg_name='ftpl',
                           simulation_size=simulation_size, balance=balance, ftpl_coff=FTPL_LOW_NOISE_COFF)
    ftpl_low_noise.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, ftpl_low_noise, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # Agent 4
    ftpl_interm_noise = Agent(stocks, stock_values, ftpl, alg_name='ftpl',
                              simulation_size=simulation_size, balance=balance,
                           ftpl_coff=FTPL_INTERMEDIATE_COFF)
    ftpl_interm_noise.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, ftpl_interm_noise, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # Agent 5
    ftpl_high_noise = Agent(stocks, stock_values, ftpl, alg_name='ftpl',
                            simulation_size=simulation_size, balance=balance,
                              ftpl_coff=FTPL_HIGH_COFF)
    ftpl_high_noise.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, ftpl_high_noise, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # Agent 6
    mwa_agent = Agent(stocks, stock_values, mwa, alg_name='mwa',
                            simulation_size=simulation_size, balance=balance)
    mwa_agent.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, mwa_agent, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # Agent 7
    ogd_agent = Agent(stocks, stock_values, ogd, alg_name='ogd',
                            simulation_size=simulation_size, balance=balance)
    ogd_agent.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, ogd_agent, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)
    # Agent 8
    ucb_agent = Agent(stocks, stock_values, ucb_stocks, alg_name='ucb',
                      simulation_size=simulation_size, balance=balance, ucb_delta=0.01)
    ucb_agent.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, ucb_agent, mode='bandit')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # Agent 9
    ucb_agent_high_delta = Agent(stocks, stock_values, ucb_stocks, alg_name='ucb',
                      simulation_size=simulation_size, balance=balance, ucb_delta=0.25)
    ucb_agent_high_delta.simulate_strategy()
    inst_reg, total_reg = calculate_regret(regret_needed_dict, ucb_agent_high_delta, mode='bandit')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    agent_list = [ftl_average_agent,
                  ftl_max_agent,
                  ftpl_low_noise,
                  ftpl_interm_noise,
                  ftpl_high_noise,
                  mwa_agent,
                  ogd_agent,
                  ucb_agent,
                  ucb_agent_high_delta
                  ]

    # Agent 10
    agents_profits = [agent_list[k].all_profit_losses for k in range(len(agent_list))]
    agents_weights = [agent_list[k].all_weights for k in range(len(agent_list))]
    # regret calculation for the following agents
    regret_needed_dict_for_agents = get_regret_needed_terms_for_agents(agents_profits)

    ucb_over_agents = Agent(stocks, stock_values, ucb_agents, alg_name='ucb_agents',
                                 simulation_size=simulation_size, balance=balance, ucb_delta=0.01,
                            agents_profits=agents_profits, agents_weights=agents_weights)
    ucb_over_agents.simulate_strategy()
    agent_list.append(ucb_over_agents)
    inst_reg, total_reg = calculate_regret(regret_needed_dict_for_agents, ucb_over_agents, mode='bandit')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # Agent 11
    mwa_over_agents = Agent(stocks, stock_values, mwa_agents, alg_name='mwa_agents',
                            simulation_size=simulation_size, balance=balance,
                            agents_profits=agents_profits, agents_weights=agents_weights)
    mwa_over_agents.simulate_strategy()
    agent_list.append(mwa_over_agents)
    inst_reg, total_reg = calculate_regret(regret_needed_dict_for_agents, mwa_over_agents, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)

    # Agent 12 - fantastic agent
    mwa_plusplus = Agent(stocks, stock_values, mwa_agents_plusplus, alg_name='mwa_agents_plusplus',
                            simulation_size=simulation_size, balance=balance,
                            agents_profits=agents_profits, agents_weights=agents_weights)
    mwa_plusplus.simulate_strategy()
    agent_list.append(mwa_plusplus)
    inst_reg, total_reg = calculate_regret(regret_needed_dict_for_agents, mwa_plusplus, mode='hindsight')
    experimental_total_regrets.append(total_reg)
    experimental_inst_regrets.append(inst_reg)


    balances = [i.balances for i in agent_list]
    investments_overstocks = [i.invested_stocks for i in agent_list[0:9]]
    investments_overagents = [i.invested_stocks for i in agent_list[9:]]
    labels = ['ftl_avg',
                    'ftl_max',
                    'ftpl_low',
                    'ftpl_med',
                    'ftpl_high',
                  'mwa',
                  'ogd',
                  'ucb',
                  'ucb_h',
                    'ucb_a',
                  'mwa_a',
                  'mwa_pp']

    colors = ['#EC407A', '#AB47BC', '#5C6BC0', '#42A5F5', '#26A69A', '#9CCC65', '#FFEE58',
              '#B71C1C', '#FF6F00', '#263238', '#90A4AE', '#EF9A9A']

    visualize_balances(balances, labels, colors, save_name='all_balances')
    visualize_total_regrets(experimental_total_regrets, labels, colors, save_name='total_regrets')
    visualize_instantaneous_regrets(experimental_inst_regrets, labels, colors, 'instantaneous_regrets')
    get_stock_distributions(investments_overstocks, labels[0:9],stocks, colors[0:9], num_stocks=5, num_agents=9, num_rows=3, num_columns=3, save_name='stock_dist')
    get_stock_distributions(investments_overagents, labels[9:], labels[0:9], colors[9:12], num_stocks=9, num_agents=3, num_rows=3,
                            num_columns=1, save_name='agent_dist')


    only_ftl_agents = [ftl_average_agent,
                  ftl_max_agent,
                  ftpl_low_noise,
                  ftpl_interm_noise,
                  ftpl_high_noise
                  ]
    labels = ['ftl_avg',
              'ftl_max',
              'ftpl_low',
              'ftpl_med',
              'ftpl_high']
    colors = ['#EC407A', '#AB47BC', '#5C6BC0', '#42A5F5', '#26A69A']
    balances = [i.balances for i in only_ftl_agents]
    visualize_balances(balances, labels, colors, save_name='only_ftl_balances')

    other_agents = [
                  mwa_agent,
                  ogd_agent,
                  ucb_agent,
                  ucb_agent_high_delta
                  ]
    labels = ['mwa',
                  'ogd',
                  'ucb',
                  'ucb_h']
    colors = ['#9CCC65', '#FFEE58',
              '#B71C1C', '#FF6F00']
    balances = [i.balances for i in other_agents]
    visualize_balances(balances, labels, colors, save_name='other_balances')

    agents_over_strategies = [
        ucb_over_agents,
        mwa_over_agents,
        mwa_plusplus
    ]
    labels = ['ucb_a',
                  'mwa_a',
                  'mwa_pp']
    colors = ['#263238', '#90A4AE', '#EF9A9A']
    balances = [i.balances for i in agents_over_strategies]
    visualize_balances(balances, labels, colors, save_name='agents_over_strategies')


    plt.show()
