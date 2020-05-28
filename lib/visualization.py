from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from bokeh.plotting import figure, output_file, show, save, output_notebook
from bokeh.layouts import column

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

