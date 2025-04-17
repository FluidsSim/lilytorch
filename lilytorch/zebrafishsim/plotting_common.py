
import matplotlib.pyplot as plt
import numpy as np
import os
import farms_pylog as pylog
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
import matplotlib
matplotlib.rc('font', **{"size":20})
plt.rcParams["figure.figsize"] = (10,10)

def plot_time_histories(
    time: np.array,
    state: np.array,
    **kwargs,
):
    """
    Plot time histories of a vector of states on a single plot
    time: array of times
    state: array of states with shape (niterations,nvar)
    kwargs: optional plotting properties (see below)
    """

    xlabel        = kwargs.pop('xlabel', "Time [s]")
    ylabel        = kwargs.pop('ylabel', "Activity [-]")
    title         = kwargs.pop('title', None)
    labels        = kwargs.pop('labels', None)
    colors        = kwargs.pop('colors', None)
    xlim          = kwargs.pop('xlim', [0,time[-1]])
    ylim          = kwargs.pop('ylim', None)
    offset        = kwargs.pop('offset', 0)
    savepath      = kwargs.pop('savepath', None)
    lw            = kwargs.pop('lw', 1.0)
    xticks        = kwargs.pop('xticks', None)
    yticks        = kwargs.pop('yticks', None)
    xticks_labels = kwargs.pop('xticks_labels', None)
    yticks_labels = kwargs.pop('xticks_labels', None)
    closefig      = kwargs.pop('closefig', True)

    n_signals = state.shape[1]

    if offset>0:
        ymin = np.min(state)-offset*(n_signals-1)
        ymax = np.max(state)
    elif offset<0:
        ymin = np.min(state)
        ymax = np.max(state)-offset*(n_signals-1)
    else:
        ymin = np.min(state)
        ymax = np.max(state)

    amp = np.abs(ymax-ymin)
    if not ylim: ylim = [ymin-0.01*amp, ymax+0.01*amp]

    if isinstance(colors, list) and len(colors)==n_signals: #
        colors = colors
    elif not isinstance(colors, list):
        colors = [colors for _ in range(n_signals)]
    else:
        raise Exception("Color list not a vecotor of the correct size!")

    if title:
        plt.figure(title)
    for (idx, vector) in enumerate(state.transpose()):
        if not labels:
            label = None
        else:
            label = labels[idx]
        plt.plot(time, vector-offset*idx, label=label, color=colors[idx], linewidth=lw)
    if labels:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(False)
    if xticks:
        plt.xticks(xticks, labels=xticks_labels)
    if yticks:
        plt.yticks(yticks, labels=yticks_labels)
    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()


def plot_time_histories_multiple_windows(
    time: np.array,
    state: np.array,
    **kwargs,
):
    """
    Plot time histories of a vector of states on multiple subplots
    time: array of times
    state: array of states with shape (niterations,nvar)
    kwargs: optional plotting properties (see below)
    """

    xlabel        = kwargs.pop('xlabel', "Time [s]")
    ylabel        = kwargs.pop('ylabel', "Activity [-]")
    title         = kwargs.pop('title', None)
    labels        = kwargs.pop('labels', None)
    colors        = kwargs.pop('colors', None)
    xlim          = kwargs.pop('xlim', [0,time[-1]])
    ylim          = kwargs.pop('ylim', None)
    savepath      = kwargs.pop('savepath', None)
    lw            = kwargs.pop('lw', 1.0)
    xticks        = kwargs.pop('xticks', None)
    yticks        = kwargs.pop('yticks', None)
    xticks_labels = kwargs.pop('xticks_labels', None)
    yticks_labels = kwargs.pop('xticks_labels', None)
    closefig      = kwargs.pop('closefig', True)


    if isinstance(colors, list) and len(colors)==state.shape[1]: #
        colors = colors
    elif not isinstance(colors, list):
        colors = [colors for _ in range(state.shape[1])]
    else:
        raise Exception("Color list not a vecotor of the correct size!")

    n=state.shape[1]
    if title:
        plt.figure(title)
    for (idx, vector) in enumerate(state.transpose()):
        if not labels:
            label = None
        else:
            label = labels[idx]
        plt.subplot(n,1,idx+1)
        plt.plot(time, vector, label=label, color=colors[idx], linewidth=lw)
        plt.ylabel(ylabel)
        plt.xlim(xlim)
        plt.ylim(ylim)
    if labels:
        plt.legend()
    plt.xlabel(xlabel)
    plt.grid(False)
    if xticks:
        plt.xticks(xticks, labels=xticks_labels)
    if yticks:
        plt.yticks(yticks, labels=yticks_labels)
    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    # plt.plot(results[:, 0], results[:, 1], 'r.')
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation='lanczos',
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])



def plot_left_right(times, state, left_idx, right_idx, cm="jet", offset=0.3):
    """
    plotting left and right states
    Inputs:
    - times: array of times
    - state: array of states with shape (niterations,nvar)
    - left_idx: index of the left nvars
    - right_idx: index of the right nvars
    - cm(optional): colormap
    - offset(optional): y-offset for each variable plot
    """

    n=len(left_idx)

    if cm=="jet":
        colors =  plt.cm.jet( np.linspace( 0, 1, n) ).tolist()
    elif cm=="Greens":
        colors =  plt.cm.Greens( np.linspace( 0, 1, n) ).tolist()
    else:
        colors = cm

    plt.subplot(2,1,1)
    plot_time_histories(
        times,
        state[:,left_idx],
        offset=-offset,
        colors=colors,
        ylabel="Left",
        )
    plt.subplot(2,1,2)
    plot_time_histories(
        times,
        state[:,right_idx],
        offset=offset,
        colors=colors,
        ylabel="Right"
        )


def plot_trajectory(controller, label=None, color=None, sim_fraction=1):

    head_positions = np.array(controller.links_positions)[:, 0, :]
    n_steps            = head_positions.shape[0]
    n_steps_considered = round(n_steps * sim_fraction)

    head_positions  = head_positions[-n_steps_considered:,:2]

    """Plot head positions"""
    plt.plot(head_positions[:-1, 0], head_positions[:-1, 1], label=label, color=color)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid(True)


def plot_positions(times, link_data):
    """Plot positions of the link_data data"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=['x', 'y', 'z'][i])
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.grid(True)

def save_figure(figure, dir='results', name=None, **kwargs):
    """ Save figure """
    for extension in kwargs.pop('extensions', ['pdf']):
        fig = figure.replace(' ', '_').replace('.', 'dot')
        if name is None:
            name = f'{dir}/{fig}.{extension}'
        else:
            name = f'{dir}/{fig}.{extension}'
        fig = plt.figure(figure)
        size = plt.rcParams.get('figure.figsize')
        fig.set_size_inches(0.7*size[0], 0.7*size[1], forward=True)
        fig.savefig(name, bbox_inches='tight')
        pylog.debug('Saving figure %s...', name)
        fig.set_size_inches(size[0], size[1], forward=True)

def save_figures(**kwargs):
    """Save_figures"""
    figures = [str(figure) for figure in plt.get_figlabels()]
    pylog.debug('Other files:\n    - %s', '\n    - '.join(figures))
    os.makedirs('./results/', exist_ok=True)
    for name in figures:
        save_figure(name, extensions=kwargs.pop('extensions', ['png']))

