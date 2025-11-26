import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib
matplotlib.rc('font', **{"size":20})
plt.rcParams["figure.figsize"] = (15,15)

def plot_time_histories(
    time: np.array,
    state: np.array,
    **kwargs,
):
    """
    Plot time histories of a vector of states
    time: array of times
    state: array of array of values
    kwargs: optional plotting properties
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

    ymin = np.min(state)
    ymax = np.max(state)
    if not ylim: ylim = [ymin-0.1*ymin, ymax+0.1*ymax]

    if title:
        plt.figure(title)
    for (idx, vector) in enumerate(state.transpose()):
        if not labels:
            label = None
        else:
            label = labels[idx]
        if not colors:
            color = None
        else:
            color = colors[idx]
        plt.plot(time, vector-offset*idx, label=label, color=color, linewidth=lw)
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

def plot_spk_train(
    pos,
    st,
    I='',
    color=None,
):
    """
    Plot spike train
    pos: array of positions
    st: array of arrays containing spike times
    I: optional xlim
    colors: optional uniform colour
    """
    for (i,spike_train) in enumerate(st):
        plt.plot(
            spike_train,
            pos[i]*np.ones_like(spike_train),
            marker='.',
            markersize=5,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle='None'
            )
        if len(I):
            plt.xlim(I)

def histogram_stair(
    data,
    bins,
    **kwargs,
):
    xlabel   = kwargs.pop('xlabel', None)
    ylabel   = kwargs.pop('ylabel', "# of occurrences")
    title    = kwargs.pop('title', None)
    xlim     = kwargs.pop('xlim', None)
    ylim     = kwargs.pop('ylim', None)
    savepath = kwargs.pop('savepath', None)
    closefig      = kwargs.pop('closefig', True)

    counts, bins = np.histogram(data, bins=bins)
    if title:
        plt.figure(title)
    plt.stairs(counts, bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()

def histogram_plot(
    fig,
    ax,
    x,
    y,
    **kwargs,
):
    xlabel   = kwargs.pop('xlabel', None)
    ylabel   = kwargs.pop('ylabel', "\# of occurrences")
    title    = kwargs.pop('title', None)
    color    = kwargs.pop('color', 'k')
    xlim     = kwargs.pop('xlim', None)
    ylim     = kwargs.pop('ylim', [np.min(y)-0.1*np.min(y), np.max(y)+0.1*np.max(y)+0.4])
    width    = kwargs.pop('width', 0.5)
    xticks   = kwargs.pop('xticks', None)
    savepath = kwargs.pop('savepath', None)
    closefig = kwargs.pop('closefig', True)
    yticks_labels = kwargs.pop('xticks_labels', [])

    xs = range(len(x))

    if title:
        ax.set_title(title)
    ax.bar(x, y, width, color=color)
    if xticks:
        plt.gca().set_xticks(xs)
        ax.set_xticklabels(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()
    for index, value in enumerate(y):
        plt.text(index, value,
                str(round(value,2)), ha = 'center')
    # ax.set_yticklabels(yticks_labels)


def stack_bar_chart(
    values,
    data,
    **kwargs,
):

    xlabel   = kwargs.pop('xlabel', None)
    width    = kwargs.pop('width', 0.5)
    xticks   = kwargs.pop('xticks', None)
    savepath = kwargs.pop('savepath', None)
    closefig = kwargs.pop('closefig', True)

    weight_counts = {
        "no conv": [d.count(-1) for d in data],
        "unclas": [d.count(0) for d in data],
        "lsw": [d.count(1) for d in data],
        "trot": [d.count(2) for d in data],
        "dsw": [d.count(3) for d in data],
        "pace": [d.count(4) for d in data]

    }

    fig, ax = plt.subplots()
    bottom = np.zeros(len(values))
    for boolean, weight_count in weight_counts.items():
        p = ax.bar(values, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    plt.ylim([0, max(bottom)+1])
    ax.set_title("Gait classification")
    ax.legend(loc="upper right")
    plt.xlabel(xlabel)
    if xticks:
        # plt.gca().set_xticks([0,len(x)])
        plt.gca().set_xticklabels(xticks)

    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()



def patch_plot(
    data,
    npixel_x = 1,
    npixel_y = 30,
    gap      = 5,
    y_resize = 1000,
    cmap     = "cividis",
    **kwargs
    ):

    """
    foot_values = matrix
    npixel_x = number of x pixels to color (choose 1 for time plot)
    npixel_y = number of y pixels to color
    gap = number of gap pixels
    cmap = color map
    labels = labels of each bar on the y-axis (default = range values)
    """

    if data.shape[1] > 3000: # resize is dataset is too large
        data = data[:,::int(data.shape[1]/y_resize)]

    xlabel   = kwargs.pop('xlabel', None)
    ylabel   = kwargs.pop('ylabel', None)
    clabel   = kwargs.pop('clabel', None)
    title    = kwargs.pop('title', None)
    labels   = kwargs.pop('labels', [])
    savepath = kwargs.pop('savepath', None)
    k        = kwargs.pop('gap_value', np.mean(data))
    xticks   = kwargs.pop('xticks', None)
    vmin     = kwargs.pop('vmin', -99.0)
    cback    = kwargs.pop('cback', 'white')
    dpi      = kwargs.pop('dpi', 600)
    clim     = kwargs.pop('clim', [np.min(data),np.max(data)])
    closefig      = kwargs.pop('closefig', True)
    colorbar_on = kwargs.pop('colorbar_on', False)

    cmap     = getattr(plt.cm, cmap)
    cmap.set_under(cback)

    n = len(data)

    ex_ = np.insert(data, range(1,n+1), k*np.ones(len(data[0])), axis=0)
    ex_ = np.repeat(ex_, npixel_x, axis=1)
    ex_ = np.repeat(ex_, np.tile([npixel_y,gap], n), axis=0)

    if title:
        plt.figure(title)

    plt.imshow(
        ex_,
        cmap = cmap,
        vmin = vmin
    )
    plt.clim(clim)

    if len(labels):
        plt.yticks([(npixel_y+gap)*i+npixel_y/2-0.5 for i in range(n)], labels)
    else:
        plt.gca().set_yticks([])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.gca().set_xticks([0,data.shape[1]])
        plt.gca().set_xticklabels(xticks)

    if colorbar_on:
        cbar = plt.colorbar()
        cbar.ax.set_title(clabel)

    if savepath:
        plt.savefig(savepath, dpi=dpi)


def plot2D(
    results,
    labels,
    n_data=300,
    log=False,
    cmap=None,
    sequence=None,
    interpolation=None,
    **kwargs
):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    savepath = kwargs.pop('savepath', None)
    closefig      = kwargs.pop('closefig', True)

    x=results[0]
    y=results[1]
    z=results[2]
    xnew = np.linspace(min(x), max(x), n_data)
    ynew = np.linspace(min(y), max(y), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (x, y), z,
        (grid_x, grid_y),
        method='nearest',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation=interpolation,
        norm=mpl.colors.LogNorm() if log else None
    )

    if cmap is not None:
        imgplot.set_cmap(cmap)
    cbar = plt.colorbar()
    cbar.set_label(labels[2])

    if sequence:
        sequence_interp = griddata(
            (x, y), sequence,
            (grid_x, grid_y),
            method='nearest',  # nearest, cubic
        )
        masked_data = np.ma.masked_where(sequence_interp==0, results_interp)
        plt.imshow(
            masked_data,
            extent=extent,
            aspect='auto',
            origin='lower',
            interpolation='none',
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm() if log else None
        )

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()

def fill_gait_diagram(
        ax,
        gait,
        color,
        alpha=0.1
):
    ax.fill_between(gait[0], [gait[1][0],gait[1][0]], [gait[1][1],gait[1][1]], alpha=alpha, color=color)

def gait_diagram(

):
    std = 0.2
    lsw = np.array( [
        [0.25-std, 0.25+std],
        [0.75-std, 0.75+std]
    ] )

    dsw = np.array( [
        [0.75-std, 0.75+std],
        [0.25-std, 0.25+std]
    ] )

    trot1 = np.array( [
        [0.5-std, 0.5+std],
        [0., 0.+std]
    ] )
    trot2 = np.array( [
        [0.5-std, 0.5+std],
        [1.-std, 1]
    ] )

    bound1 = np.array( [
        [0., 0.+std],
        [0.5-std, 0.5+std]
    ] )
    bound2 = np.array( [
        [1.-std, 1],
        [0.5-std, 0.5+std]
    ] )

    fig, ax = plt.subplots()
    plt.axis([0,1,0,1])

    fill_gait_diagram(trot1, color='g', alpha=0.5)
    fill_gait_diagram(trot2, color='g', alpha=0.5)
    fill_gait_diagram(bound1, color='y', alpha=0.5)
    fill_gait_diagram(bound2, color='y', alpha=0.5)

    fill_gait_diagram(lsw, color='r', alpha=0.5)
    fill_gait_diagram(dsw, color='b', alpha=0.5)
    plt.xlabel("homo")
    plt.ylabel("dia")

    n=10
    std_c=0.25
    x=np.linspace(0,1,n)
    ax.fill_between(x,x-0.5+std_c,x+0.5-std_c, alpha=1, color="k")
    ax.fill_between(x,x+0.5+std_c,1, alpha=1, color="k")
    ax.fill_between(x,0,x-0.5-std_c, alpha=1, color="k")
    ax.set_title(r"C=0.5 $\pm$ "+str(std_c))

def plot_errorbar(
    x,
    y,
    err,
    **kwargs,
):

    xlabel        = kwargs.pop('xlabel', None)
    ylabel        = kwargs.pop('ylabel', None)
    title         = kwargs.pop('title', None)
    color         = kwargs.pop('color', 'k')
    xlim          = kwargs.pop('xlim', None)
    ylim          = kwargs.pop('ylim', None)
    savepath      = kwargs.pop('savepath', None)
    xticks        = kwargs.pop('xticks', None)
    yticks        = kwargs.pop('yticks', None)
    xticks_labels = kwargs.pop('xticks_labels', None)
    yticks_labels = kwargs.pop('xticks_labels', None)
    marker        = kwargs.pop('marker', 'o')
    linestyle     = kwargs.pop('linestyle', None)
    label         = kwargs.pop('label', None)
    closefig      = kwargs.pop('closefig', True)

    plt.grid(False)
    if title:
        plt.figure(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if xticks:
        plt.xticks(xticks, labels=xticks_labels)
    if yticks:
        plt.yticks(yticks, labels=yticks_labels)

    plt.errorbar(
        x, y, err,
        ecolor=color,
        markerfacecolor=color,
        markeredgecolor=color,
        marker=marker,
        linestyle=linestyle,
        label=label
    )
    plt.legend()
    # for i in range(len(x)):
    #     plt.errorbar(x[i], y[i], err[i], ecolor=color, marker=marker, linestyle=linestyle)

    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()


def boxplot(
    data,
    **kwargs,
):

    xlabel        = kwargs.pop('xlabel', None)
    ylabel        = kwargs.pop('ylabel', None)
    title         = kwargs.pop('title', None)
    xlim          = kwargs.pop('xlim', None)
    ylim          = kwargs.pop('ylim', None)
    savepath      = kwargs.pop('savepath', None)
    xticks        = kwargs.pop('xticks', None)
    xticks_labels = kwargs.pop('xticks_labels', None)
    closefig      = kwargs.pop('closefig', True)

    plt.grid(False)
    if title:
        plt.figure(title)

    plt.boxplot(
        data,
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if xticks:
        plt.xticks(xticks, xticks_labels)

    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()

def plot_urdf_positions(
    net,
    iteration,
    **kwargs
):
    """
    plot the urdf position and contact points of the links in 3D
    """
    link_positions = np.asarray(net.data.sensors.links.urdf_positions())[iteration]
    xs=link_positions[:,0]
    ys=link_positions[:,1]
    zs=link_positions[:,2]
    ax = net.fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, marker="o", c="g")

    for i in range(4):
        p_c = net.data.sensors.contacts.position_all(i)[iteration]
        print(np.asarray(p_c))
        xc = p_c[0]
        yc = p_c[1]
        zc = p_c[2]

        ax.scatter(xc, yc, zc, marker="o", c="k")
        ax.set_xlim(-0.1,0.02)
        ax.set_ylim(-0.05,0.05)
        ax.set_zlim(-0.02,0.02)

###############################################################################
# FLUID SOLVER PLOTTING #######################################################
###############################################################################

def save_fig_to_dedicated_folder(
    save_path   : str,
    quantity_str: str,
    iteration   : int,
    type        : str = "png"
):
    ''' Save plot to dedicated folder '''

    # Create folder for the current plot
    target_folder = f'{save_path}/{quantity_str}'
    os.makedirs(target_folder, exist_ok=True)

    # Save plot
    target_file   = f'{target_folder}/{quantity_str}_{iteration}.{type}'
    plt.savefig(target_file)
    plt.close()

    return

def plot_composite_countour(X,Y,properties,iteration,save_path,name):
    plt.figure(figsize=(20,10))
    for prop in properties:
        d = prop[0].cpu()
        plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_only(u,extent,iteration,save_path,name,vmin,vmax):
    if vmin is None:
        limit = max(abs(u.min()), abs(u.max()))
        vmin = -limit
        vmax = limit
    plt.figure(figsize=(20,10))
    plt.imshow(
        u.T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu,
        interpolation=None
    )
    plt.colorbar()
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_composite(X,Y,u,properties,extent,iteration,save_path,name,vmin,vmax):
    if vmin is None:
        limit = max(abs(u.min()), abs(u.max())) #/2
        vmin = -limit
        vmax = limit
    plt.figure(figsize=(20,10))
    for prop in properties:
        d = prop[0].cpu()
        plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
    plt.imshow(
        u.detach().numpy().T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu,
        interpolation=None
    )
    plt.colorbar()
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_composite_quiver(X,Y,u,bodies,normal_x,normal_y,extent,iteration,save_path,name,vmin,vmax,subsample_n = 2**4, scale=None, body_contours = True):



    if vmin is None:
        limit = max(abs(u.min()), abs(u.max()))/2
        vmin = -limit
        vmax = limit
    if scale:
        scale=1/scale


    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    scale = 25 / max(x_range, y_range)  # Adjust 10 for overall size
    fig_width = x_range * scale
    fig_height = y_range * scale

    plt.figure(figsize=(5, 15))

    # plt.figure(figsize=(fig_width, fig_height))
    if body_contours:
        for i, body in enumerate(bodies):
            # d = body.sdf.cpu()
            # plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
            plt.plot(body.cnt_update[0][body.mask].cpu(), body.cnt_update[1][body.mask].cpu(), c="k",linewidth=0.5)
            # plt.scatter(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), c='k', s=0.1)

            # plt.fill(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), color="#000000")

            plt.plot(body.com_pos[0].cpu(), body.com_pos[1].cpu(), 'ro', markersize=2)

            # plt.plot(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), 'k',linewidth=0.5)

            # plt.plot(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), 'k',linewidth=0.5)
    plt.imshow(
        u.cpu().detach().numpy().T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu,
        interpolation=None,
        aspect='equal',
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(name)

    plt.colorbar()
    # q = plt.quiver(
    #     X[::subsample_n,::subsample_n].cpu(),
    #     Y[::subsample_n,::subsample_n].cpu(),
    #     normal_x[::subsample_n,::subsample_n].cpu(),
    #     normal_y[::subsample_n,::subsample_n].cpu(),
    #     color='g',
    #     scale=scale, scale_units='xy'
    # )
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)









def plot2d_imshow(X,Y,u,d,extent,iteration,save_path,name,vmin,vmax):
    if vmin is None:
        limit = max(abs(u.min()), abs(u.max()))/2
        vmin = -limit
        vmax = limit
    plt.figure(figsize=(20,10))
    ctr = plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
    plt.imshow(
        np.where(d<0,0,u).T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu
    )
    plt.colorbar()
    # plt.ylim([-0.25,0.25])
    # plt.xlim([0.58,1.13])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_quiver(X,Y,u,d,normal_x,normal_y,extent,iteration,save_path,name,vmin,vmax,subsample_n = 2**4, scale=None):

    if vmin is None:
        limit = max(abs(u.min()), abs(u.max()))/2
        vmin = -limit
        vmax = limit
    if scale:
        scale=1/scale
    plt.figure(figsize=(20,10))
    ctr = plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
    plt.imshow(
        u.T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu
    )
    plt.colorbar()
    q = plt.quiver(
        X[::subsample_n,::subsample_n].cpu(),
        Y[::subsample_n,::subsample_n].cpu(),
        normal_x[::subsample_n,::subsample_n].cpu(),
        normal_y[::subsample_n,::subsample_n].cpu(),
        color='g',
        scale=scale, scale_units='xy'
    )
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_simple(d,extent,iteration,save_path,name,vmin= -0.001,vmax= 0.001):
    plt.figure(figsize=(20,10))
    if vmin is None:
        limit = max(abs(d.min()), abs(d.max()))/2
        vmin = -limit
        vmax = limit
    plt.imshow(
        d.T,
        extent = extent,
        origin = "lower",
        vmin   = vmin,
        vmax   = vmax,
        cmap = cm.RdBu
    )
    plt.colorbar()
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)




def plot_ctrs(vars,bodies, extent, save_path, name, iteration,vmin= -0.001,vmax= 0.001, cmap = cm.get_cmap('viridis')):

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for i, body in enumerate(bodies):
        plt.scatter(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), c=vars[i].cpu(), cmap=cmap, norm=norm)
    plt.colorbar()
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)
