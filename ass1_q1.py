import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

env = gym.make("FrozenLake-v0")
# Build and initialize lookup table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Initialize parameters
params = {'alpha': 0.1,
          'discount_factor': 0.99,
          'epsilon': 1,
          'min_epsilon': 0.25,
          'decay': 0.95,
          'train_episodes': 5000,
          'max_steps': 100}


actions_dict = {0: 'Left',
                1: 'Down',
                2: 'Right',
                3: 'Up'}

q_to_plot = [499,1999,4999]

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_q(Q):
    fig, ax = plt.subplots(figsize=(5, 10))
    row_labels = range(Q.shape[0])
    col_labels = range(Q.shape[1])
    im, cbar = heatmap(Q, row_labels, col_labels, ax,
                       cmap="winter", cbarlabel="Q(s,a) Value")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")

    fig.tight_layout()
    plt.show()


def q_learning(env, Q, params, actions_dict,q_to_plot, verbose=False):
    """

    :param env:
    :param Q:
    :param params:
    :param actions_dict:
    :return:
    """
    r_per_episode = []
    step_per_episode = []
    for episode in tqdm(range(params['train_episodes'])):
        current_reward = 0
        state = env.reset()
        for step in range(params['max_steps']):
            if verbose:
                env.render()
            rand = np.random.rand()
            if params['epsilon'] > rand:
                action = env.action_space.sample()
                # params['epsilon'] #todo: ask in forum about the update of epsilon
            else:
                action = np.argmax(Q[state, :])

            # get reward
            new_state, reward, terminal, info = env.step(action)
            current_reward += reward

            # updating the q-function using the bellman equation
            Q[state, action] = Q[state, action] + params['alpha'] * (
                    reward + params['discount_factor'] * np.max(Q[new_state, :] - Q[state, action])
            )

            state = new_state

            if terminal:
                break
        r_per_episode.append(current_reward)
        step_per_episode.append(step)
        params['epsilon'] = max(params['min_epsilon'], params['epsilon'] * params['decay'])
        if episode in q_to_plot:
            plot_q(Q)

    return r_per_episode, step_per_episode


x, y = q_learning(env=env, Q=Q, actions_dict=actions_dict, params=params,q_to_plot=q_to_plot, verbose=False)




