import numpy as np
from Models import ucb_stocks, ucb_agents
class Agent():
    def __init__(self, stocks, stock_values, strategy, alg_name='ftl', simulation_size = 1000, balance=1000, ftpl_coff=None, ucb_delta=None,
                 agents_weights=None, agents_profits=None):

        self.balance = balance
        self.alg_name = alg_name
        self.stocks = stocks
        self.num_stocks = len(stocks)
        self.strategy = strategy
        self.stock_values = stock_values
        self.simulation_size = simulation_size
        self.open_pos_invest = np.zeros(self.num_stocks)  # this will hold the values of the stocks when the position is opened
        self.open_pos_weights = np.zeros(self.num_stocks) # this will hold the percentage of balance invested in the stocks
        self.ftpl_coff = ftpl_coff

        self.all_weights = np.zeros((self.simulation_size - 1, self.num_stocks))
        self.all_profit_losses = np.zeros((self.simulation_size - 1))
        self.invested_stocks = np.zeros((self.simulation_size - 1))

        if self.alg_name == 'mwa':
            self.mwa_weights = np.ones(self.num_stocks)
            self.mwa_losses = np.zeros(self.num_stocks)

        elif alg_name == 'ogd':
            self.ogd_weights = np.ones(self.num_stocks) / self.num_stocks

        elif alg_name =='ucb':
            self.delta = ucb_delta
            self.arm_sums = np.zeros(self.num_stocks)
            self.arm_counts = np.ones(self.num_stocks)

        elif alg_name == 'ucb_agents':
            self.arm_sums = np.zeros(len(agents_weights))
            self.arm_counts = np.ones(len(agents_weights))
            self.agents_weights = agents_weights
            self.agents_profits = agents_profits

        if (self.alg_name == 'mwa_agents') or (self.alg_name == 'mwa_agents_plusplus'):
            self.mwa_weights = np.ones(len(agents_weights))
            self.mwa_losses = np.zeros(len(agents_weights))
            self.agents_weights = agents_weights
            self.agents_profits = agents_profits


        self.stock_profit_losses = [] # list of lists
        for i in range(len(self.stocks)):
            self.stock_profit_losses.append([])
        self.balances = [self.balance]

    def open_positions(self, open_pos_invest, open_pos_weights):
        self.open_pos_weights = open_pos_weights
        self.open_pos_invest = open_pos_invest


    def close_positions(self, close_pos_invest):
        # close_pos_invest carries the values of the stocks at current time
        new_balance = 0
        for i in range(len(self.stocks)):
            # find the ratio of current price vs old price
            ratio = close_pos_invest[i] / self.open_pos_invest[i]
            # multiply this ratio with the money invested on that stock
            profit_loss = ratio * self.open_pos_weights[i] * self.balance
            self.stock_profit_losses[i].append((ratio - 1) * self.open_pos_weights[i] * self.balance)
            new_balance += profit_loss
        # zero out the stock since we closed the position
        self.open_pos_weights = np.zeros(self.num_stocks)
        self.open_pos_invest = np.zeros(self.num_stocks)
        # update balance
        self.balance = new_balance
        self.balances.append(new_balance)

    def simulate_strategy(self):
        last_played_arm = None
        prev_weights = np.zeros(self.num_stocks)
        for t in range(1, self.simulation_size):
            history = self.stock_values[:t, :]
            current_values = self.stock_values[t, :]
            # apply the strategy
            if t == 1:  # if we are just starting, start with opening positions
                if self.alg_name=='ftl':
                    weights = self.strategy(history)
                elif self.alg_name =='ftpl':
                    weights = self.strategy(history, self.ftpl_coff)
                elif self.alg_name=='mwa':  # for mwa - cannot make a prediction at t=1, return number weight
                    weights = np.ones(self.num_stocks)/self.num_stocks
                elif self.alg_name=='ogd':  # for mwa - cannot make a prediction at t=1, return number weight
                    weights = np.ones(self.num_stocks)/self.num_stocks
                elif self.alg_name == 'ucb':
                    sample_means = np.divide(self.arm_sums, self.arm_counts)
                    weights, self.arm_counts = ucb_stocks(history, sample_means, self.arm_counts, delta=self.delta)
                    last_played_arm = np.argmax(weights)
                elif self.alg_name == 'ucb_agents':
                    agents_current_profits = [self.agents_profits[k][t] for k in range(len(self.agents_profits))]
                    weights_current = [self.agents_weights[k][t] for k in range(len(self.agents_weights))]
                    sample_means = np.divide(self.arm_sums, self.arm_counts)
                    agent_weights, self.arm_counts = ucb_agents(t, sample_means, self.arm_counts, delta=0.01)
                    # now pick the agent
                    arm_idx = np.argmax(agent_weights)
                    weights = weights_current[arm_idx]
                    self.arm_sums[arm_idx] += agents_current_profits[arm_idx]
                elif (self.alg_name=='mwa_agents'):  # for mwa - cannot make a prediction at t=1, return number weight
                    agent_weights = np.ones(len(self.agents_weights))/len(self.agents_weights)
                    argmax_ind = np.random.choice(len(agent_weights), size=1, p=agent_weights)[0]
                    weights = self.agents_weights[argmax_ind][t]
                elif self.alg_name == 'mwa_agents_plusplus':
                    weights_current = [self.agents_weights[k][t] for k in range(len(self.agents_weights))]
                    agent_stock_matrix = np.array(weights_current)
                    agent_probs = np.ones(len(self.agents_weights)) / len(self.agents_weights)  # 5,
                    weights = np.matmul(agent_stock_matrix.T,
                                        np.reshape(agent_probs, (len(agent_probs), 1)))
                self.open_positions(current_values, weights)
            else:  # t is not 1
                prev_weights = np.copy(weights)
                # first close the position you have
                self.close_positions(current_values)
                # now open position again
                if self.alg_name == 'ftl':
                    weights = self.strategy(history)
                    self.invested_stocks[t - 1] = np.argmax(weights)
                elif self.alg_name == 'ftpl':
                    weights = self.strategy(history, self.ftpl_coff)
                    self.invested_stocks[t - 1] = np.argmax(weights)
                elif self.alg_name == 'mwa':
                    weights, self.mwa_weights, self.mwa_losses = self.strategy(history[-2, :], history[-1, :],
                                                                               self.mwa_weights,
                                                                               self.mwa_losses,
                                                                               eta=1 / np.sqrt(self.simulation_size))
                    self.invested_stocks[t - 1] = np.argmax(weights)
                elif self.alg_name=='ogd':  # for mwa - cannot make a prediction at t=1, return number weight
                    weights, self.ogd_weights = self.strategy(np.divide(history[-2, :], history[-1, :] - 1),
                                                              self.balance,
                                                              self.ogd_weights, eta=1 / np.sqrt(self.simulation_size))
                    self.invested_stocks[t - 1] = np.argmax(weights)
                elif self.alg_name =='ucb':
                    last_price = history[-1]
                    self.arm_sums[last_played_arm] += last_price[last_played_arm] / current_values[last_played_arm]
                    sample_means = np.divide(self.arm_sums, self.arm_counts)
                    weights, self.arm_counts = ucb_stocks(history, sample_means, self.arm_counts, delta=self.delta)
                    last_played_arm = np.argmax(weights)
                    self.invested_stocks[t - 1] = np.argmax(weights)
                elif self.alg_name =='ucb_agents':
                    # profit and weight at t-1
                    agents_current_profits = [self.agents_profits[k][t-1] for k in range(len(self.agents_profits))]
                    weights_current = [self.agents_weights[k][t-1] for k in range(len(self.agents_weights))]
                    sample_means = np.divide(self.arm_sums, self.arm_counts)
                    agent_weights, self.arm_counts = ucb_agents(t, sample_means, self.arm_counts, delta=0.01)
                    # now pick the agent
                    arm_idx = np.argmax(agent_weights)
                    weights = weights_current[arm_idx]
                    self.arm_sums[arm_idx] += agents_current_profits[arm_idx]
                    self.invested_stocks[t - 1] = np.argmax(agent_weights)

                elif self.alg_name == 'mwa_agents':
                    agents_current_profits = [self.agents_profits[k][t-1] for k in range(len(self.agents_profits))]
                    weights_current = [self.agents_weights[k][t-1] for k in range(len(self.agents_weights))]
                    agent_weights, self.mwa_weights, self.mwa_losses = self.strategy(agents_current_profits,
                                                                               self.mwa_weights,
                                                                               self.mwa_losses,
                                                                               eta=1 / np.sqrt(self.simulation_size))
                    arm_idx = np.argmax(agent_weights)
                    weights = weights_current[arm_idx]
                    self.invested_stocks[t - 1] = np.argmax(agent_weights)


                elif self.alg_name == 'mwa_agents_plusplus':
                    agents_current_profits = [self.agents_profits[k][t-1] for k in range(len(self.agents_profits))]
                    weights_current = [self.agents_weights[k][t-1] for k in range(len(self.agents_weights))]
                    agent_probs, self.mwa_weights, self.mwa_losses = self.strategy(agents_current_profits,
                                                                                     self.mwa_weights,
                                                                                     self.mwa_losses,
                                                                                     eta=1 / np.sqrt(
                                                                                         self.simulation_size))
                    agent_stock_matrix = np.array(weights_current)
                    weights = np.matmul(agent_stock_matrix.T,
                                        np.reshape(agent_probs, (len(agent_probs), 1)))
                    self.invested_stocks[t - 1] = np.argmax(agent_probs)


                self.open_positions(current_values, weights)
            # save
            self.all_weights[t -1] = np.squeeze(prev_weights)
            self.all_profit_losses[t-1] = np.dot(current_values/history[-1], weights)

        print(f"{self.alg_name}: ", self.balance)