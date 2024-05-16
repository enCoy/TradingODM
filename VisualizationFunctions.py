import matplotlib.pyplot as plt
import numpy as np

def visualize_profit_losses(profit_losses):
    # this is for a single agent
    plt.figure(figsize=(8, 6))
    plt.title("Profit loss curve")
    for i in range(len(profit_losses)):
        profit_loss_i = profit_losses[i]
        plt.plot(np.arange(len(profit_loss_i)), profit_loss_i, label=str(i + 1))
    plt.legend()
    plt.show()

def visualize_balances(balances, labels=None, colors=None, save_name=None):
    # this is for multiple agents
    plt.figure(figsize=(8, 6))
    plt.title("", fontname='Verdana', fontsize=16)
    plt.xlabel("Day", fontname='Verdana', fontsize=14)
    plt.ylabel("Current balance", fontname='Verdana', fontsize=14)
    plt.grid(visible=True)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    for i in range(len(balances)):
        balance = balances[i]
        plt.plot(np.arange(len(balance)), balance, label=labels[i], color=colors[i], alpha=0.8, linewidth=1.5)
    plt.axhline(y=10000, color='#455A64', linestyle='--', linewidth=1, label='', alpha=0.7)
    plt.text(len(balances[0]), 10000 - 10, f'Initial \nbalance')
    plt.legend()
    plt.savefig(save_name + '.png', format='png', dpi=600)

def visualize_total_regrets(total_regrets, labels, colors, save_name=None):
    plt.figure(figsize=(8, 6))
    plt.title("", fontname='Verdana', fontsize=16)
    plt.xlabel("Day", fontname='Verdana', fontsize=14)
    plt.ylabel("Regret", fontname='Verdana', fontsize=14)
    for j in range(len(total_regrets)):
        plt.bar(j, total_regrets[j], color=colors[j], alpha=0.8)
    plt.xticks(np.arange(len(total_regrets)), labels=labels, fontsize=14, fontname='Verdana', rotation=30)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid()
    plt.yticks(fontsize=16, fontname='Arial')
    plt.savefig(save_name + '.png', format='png', dpi=600)

def visualize_instantaneous_regrets(inst_regrets, labels, colors, save_name=None):
    # this is for multiple agents
    plt.figure(figsize=(8, 6))
    plt.title("", fontname='Verdana', fontsize=16)
    plt.xlabel("Day", fontname='Verdana', fontsize=14)
    plt.ylabel("Regret", fontname='Verdana', fontsize=14)
    plt.grid(visible=True)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    for i in range(len(inst_regrets)):
        regret = inst_regrets[i]
        plt.plot(np.arange(len(regret)), regret, label=labels[i], color=colors[i], alpha=0.5, linewidth=1.5)
    plt.legend()
    plt.savefig(save_name + '.png', format='png', dpi=600)

def get_stock_distributions(investments, alg_names, labels, colors, num_stocks = 5, num_agents = 12, num_rows = 3, num_columns = 4, save_name=None):
    fig, axs = plt.subplots(nrows=num_rows, ncols = num_columns, figsize=(8, 6))
    plt.setp(axs, xticks=[j for j in range(num_stocks)], xticklabels=labels)
    # fig.suptitle('Investment Distributions', fontsize=12, fontname="Verdana")
    for i in range(num_agents):
        investment_data = investments[i]
        # bar plot
        rwidth=0.8
        if axs.ndim > 1:
            axs[i // num_columns, int(i % num_columns)].hist(investment_data, bins=np.arange(num_stocks + 1), color=colors[i], alpha=0.8, density=True, edgecolor="dimgrey", rwidth=rwidth)
            axs[i // num_columns, int(i % num_columns)].set_xlabel('Stocks', fontname="Verdana")
            axs[i // num_columns, int(i % num_columns)].set_ylabel('Density', fontname="Verdana")

            axs[i // num_columns, int(i % num_columns)].spines['right'].set_visible(False)
            axs[i // num_columns, int(i % num_columns)].spines['top'].set_visible(False)

            axs[i // num_columns, int(i % num_columns)].set_xticks([j + 0.5 for j in range(num_stocks)])
            axs[i // num_columns, int(i % num_columns)].set_xticklabels(labels, rotation=30)
            axs[i // num_columns, int(i % num_columns)].set_title(alg_names[i], fontname='Arial')

        else:
            axs[i].hist(investment_data, bins=np.arange(num_stocks + 1),
                                                             color=colors[i], alpha=0.8, density=True,
                                                             edgecolor="dimgrey", rwidth=rwidth)
            axs[i].set_xlabel('Stocks', fontname="Verdana")
            axs[i].set_ylabel('Density', fontname="Verdana")

            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)

            axs[i].set_xticks([j + 0.5 for j in range(num_stocks)])
            axs[i].set_xticklabels(labels, rotation=30)
            axs[i].set_title(alg_names[i], fontname='Arial', fontweight='bold', color='dimgrey')


    for ax in fig.get_axes():
        ax.label_outer()
    plt.tight_layout()
    plt.savefig(save_name + '.png', format='png', dpi=600)