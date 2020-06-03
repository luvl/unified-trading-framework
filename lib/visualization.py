from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from bokeh.plotting import figure, output_file, show, save, output_notebook
from bokeh.layouts import column

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-white')
import seaborn as sns

from sklearn import metrics

class BacktestingVisualization:
    def __init__(self, df_filepath, title='Backtest', mode='static', window_size=8):
        self.folder = os.path.dirname(df_filepath)
        self.df = pd.read_pickle(df_filepath)
        self.mode = mode
        self.title = title
        self.window_size = window_size

        fig = plt.figure(figsize=(16,8))
        fig.suptitle(title)

        self.price_ax = plt.subplot2grid(
            (10, 1), (2, 0), rowspan=8, colspan=1
        )
        self.net_worth_ax = plt.subplot2grid(
            (10, 1), (0, 0), rowspan=2, colspan=1, sharex=self.price_ax
        )

        plt.subplots_adjust(
            left=0.11, bottom=0.24,
            right=0.90, top=0.90, wspace=0.2, hspace=0
        )

    def run(self):
        if self.mode == 'static':
            net_worth = self.df['Net_worth']
            dates = self.df['Date']
            trades = self.df['Trades']
            close_prices = self.df['Close']

            self.net_worth_ax.plot_date(
                dates,
                net_worth,
                '-',
                label='Net Worth'
            )
            self.net_worth_ax.legend()
            legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
            legend.get_frame().set_alpha(0.4)

            self.price_ax.plot_date(
                dates,
                close_prices,
                '-',
                label='Close price'
            )

            self.price_ax.legend()
            legend = self.price_ax.legend(loc=2, ncol=2, prop={'size': 8})
            legend.get_frame().set_alpha(0.4)

            buy_legend = False
            sell_legend = False

            for i in range(len(trades)):
                trade = trades.values[i]
                date = dates.values[i]
                close = close_prices.values[i]

                if trade['type'] == 'buy':
                    if not buy_legend: 
                        self.price_ax.scatter(date, close, marker="v", c="#6666ee", cmap="#00FF00", alpha=1, linewidths=1, label='Buy')
                        buy_legend = True
                    else:
                        self.price_ax.scatter(date, close, marker="v", c="#6666ee", cmap="#00FF00", alpha=1, linewidths=1)
                elif trade['type'] == 'sell':
                    if not sell_legend:
                        self.price_ax.scatter(date, close, marker="^", c="#ee6666", cmap="#FF0000", alpha=1, linewidths=1, label='Sell')
                        sell_legend = True
                    else:
                        self.price_ax.scatter(date, close, marker="^", c="#ee6666", cmap="#FF0000", alpha=1, linewidths=1)

            self.price_ax.legend()
            plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
            if not os.path.exists(self.folder+'/'+self.title+'.png'):
                plt.savefig(self.folder+'/'+self.title)


        elif self.mode == 'live':
            for i in range(self.window_size, len(self.df)-self.window_size):

                net_worth = self.df['Net_worth'][i-self.window_size:i]
                dates = self.df['Date'][i-self.window_size:i]
                trades = self.df['Trades'][i-self.window_size:i]
                close_prices = self.df['Close'][i-self.window_size:i]

                self._render_net_worth(dates, net_worth)
                self._render_price(dates, close_prices)
                self._render_trade(trades, dates, close_prices)

                self.net_worth_ax.legend()
                legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
                legend.get_frame().set_alpha(0.4)

                last_date = dates.values[-1]
                last_net_worth = net_worth.values[-1]

                self.net_worth_ax.annotate(
                    '{0:.2f}'.format(last_net_worth), 
                    (last_date, last_net_worth),
                    xytext=(last_date, last_net_worth),
                    bbox=dict(boxstyle='round',fc='w', ec='k', lw=1),
                    color="black",
                    fontsize="small"
                )

                self.price_ax.set_xticklabels(
                    dates, 
                    rotation=45,
                    horizontalalignment='right'
                )

                plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
                plt.pause(0.01)

        plt.ioff()
        plt.show()

    def _render_net_worth(self, dates, net_worth):
        self.net_worth_ax.clear()
        self.net_worth_ax.plot_date(
            dates,
            net_worth,
            '-',
            label='Net Worth'
        )
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = dates.values[-1]
        last_net_worth = net_worth.values[-1]

        self.net_worth_ax.annotate(
            '{0:.2f}'.format(last_net_worth), 
            (last_date, last_net_worth),
            xytext=(last_date, last_net_worth),
            bbox=dict(boxstyle='round',fc='w', ec='k', lw=1),
            color="black",
            fontsize="small"
        )

    def _render_price(self, dates, close_prices):
        self.price_ax.clear()

        last_date = dates.values[-1]
        last_close = close_prices.values[-1]

        self.price_ax.plot_date(
            dates,
            close_prices,
            '-',
            label='Close price'
        )

        self.price_ax.legend()
        legend = self.price_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        self.price_ax.annotate(
            '{0:.2f}'.format(last_close), 
            (last_date, last_close),
            xytext=(last_date, last_close),
            bbox=dict(boxstyle='round',fc='w', ec='k', lw=1),
            color="black",
            fontsize="small"
        )


    def _render_trade(self, trades, dates, close_prices):
        for i in range(len(trades)):
            trade = trades.values[i]
            date = dates.values[i]
            close = close_prices.values[i]

            if trade['type'] == 'buy':
                self.price_ax.scatter(date, close, marker="v", c="#6666ee", cmap="#00FF00", alpha=1, linewidths=5)
            elif trade['type'] == 'sell':
                self.price_ax.scatter(date, close, marker="^", c="#ee6666", cmap="#FF0000", alpha=1, linewidths=5)

            tc = '{0:.2f}'.format(trade['transaction_value'])
            self.price_ax.annotate(f'${tc}', (date, close),
                                    xytext=(date, close),
                                    fontsize=5,)


def plot_profit(folder, profit, profit2, values, actions, title='profit_curve', save_only=True):
    """
    Plot the q values history of the agent
    """
    if not os.path.exists(folder):
        os.mkdir(folder)

    output_file(folder + '/' + title + '.html')

    s1 = figure(title='Vectorized profits', plot_width=800, plot_height=600)
    x_axis = range(len(profit))
    p_axis = [i for i in profit]  # neutral (blue)
    s1.title.text_font_size = '20pt'
    s1.line(x_axis, p_axis, line_color="red", line_width=2)

    s2 = figure(title='Cumulative profits', plot_width=800, plot_height=600)
    x_axis = range(len(profit2))
    p_axis = [i for i in profit2]  # neutral (blue)
    s2.title.text_font_size = '20pt'
    s2.line(x_axis, p_axis, line_color="red", line_width=2)

    s3 = figure(title='Price', plot_width=800, plot_height=600)
    x_axis = range(len(values))
    v_axis = [v for v in values]  # neutral (blue)
    s3.title.text_font_size = '20pt'
    s3.line(x_axis, v_axis, line_color="cyan", line_width=2)

    s4 = figure(title='Actions', plot_width=800, plot_height=600)
    x_axis = range(len(actions))
    a_axis = [a for a in actions]  # neutral (blue)
    s4.title.text_font_size = '20pt'
    s4.line(x_axis, a_axis, line_color="blue", line_width=2)

    p = column(s1, s2, s3, s4)

    if save_only:
        save(p)
    else:
        show(p)

def plot_actions(folder, memory, long_actions, short_actions, save_only=True):
    if not os.path.exists(folder):
        os.mkdir(folder)

    title = 'agent_actions'
    output_file(folder + '/' + title + '.html')
    p = figure(plot_width=1000, plot_height=600)
    data_len = len(memory)

    x_axis = [x[0] for x in memory] # index/position
    y_axis = [x[1] for x in memory] # value
    x_axis_l = [x[0] for x in long_actions] # long index/position
    y_axis_l = [y[1] for y in long_actions] # long value
    x_axis_s = [x[0] for x in short_actions] # short index/position
    y_axis_s = [y[1] for y in short_actions] # short value

    p.line(x_axis, y_axis, line_width=2)
    p.scatter(x_axis_l, y_axis_l, marker="triangle",
            line_color="#6666ee", fill_color="#00FF00", fill_alpha=0.5, size=16, legend_label="Long postion")
    p.scatter(x_axis_s, y_axis_s, marker="inverted_triangle",
            line_color="#ee6666", fill_color="#FF0000", fill_alpha=0.5, size=16, legend_label="Short position")
    if save_only:
        save(p)
    else:
        print("Plotting long and short position within financial windows")
        show(p)

def plot_train_rewards(folder, rewards):
    if not os.path.exists(folder):
        os.mkdir(folder)

    title = 'train_rewards'
    output_file(folder + '/' + title + '.html')
    p = figure(plot_width=1000, plot_height=600)
    x_axis = range(len(rewards))
    y_axis = rewards
    p.line(x_axis, y_axis, line_width=2)
    print("Plotting train reward ...")
    show(p)

def visualize_heatmap_cf(y, y_hat, save_location=None):
    unique_labels = list(np.unique(y))
    matrix = metrics.confusion_matrix(y, y_hat, labels=unique_labels)

    fig, ax = plt.subplots()
    tick_marks = np.arange(len(unique_labels))
    sns.heatmap(pd.DataFrame(matrix, index=unique_labels, columns=unique_labels), annot=True, cmap="coolwarm" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(save_location+'/confusion_matrix')
