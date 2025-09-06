from __future__ import (
    division, print_function, absolute_import, unicode_literals)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from scipy.ndimage import gaussian_filter
from astropy.utils.console import ProgressBar


def dense_scatter(
        x, y, xscale='log', yscale='log', xlim=None, ylim=None,
        marker='o', c='w', markersize=1., edgecolor='k', edgewidth=0,
        color_smoothing_box=None, show_progress='notebook',
        ax=None, zorder=None, label=None, **kwargs):
    """
    Make scatter plots that handle overlapping data points better.

    This function attempts to make a dense scatter plot prettier by:
    + merging the 'marker edges' of overlapping data points;
    + running a median-filter to homogenize color for 'neighbouring'
      data points (size of the smoothing box specified by the user).

    Parameters
    ----------
    x, y : array_like
        x & y coordinates of the data points
    xscale, yscale : {'log', 'linear'}, optional
        Axes scales. Default is 'log' for both.
        They are also passed to the `~matplotlib.axes.Axes` object.
    xlim, ylim : None or 2-tuple, optional
        Axes ranges to show. Default is to use a range wider than the
        full data range by 10% on both sides.
        They are also passed to the `~matplotlib.axes.Axes` object.
    marker : marker style
        Default: 'o'
    c : color or array-like, optional
        Default: 'w'
    markersize : float, optional
        Default: 1
    edgecolor : color, optional
        Default: 'k'
    edgewidth : float, optional
        Default: 0
    color_smoothing_box : None or 2-tuple, optional
        If None, then no color smoothing will be performed.
        If a 2-tuple, then this parameter specifies the full width
        of the color smoothing box along X and Y direction, with its
        values interpreted as fractions of the ranges specified by
        'xlim' and 'ylim'.
    show_progress : {'notebook', True, False}, optional
        Whether to show the progress bar for color smoothing.
        If 'notebook', the progress bar will use the notebook widget
    ax : `~matplotlib.axes.Axes` object, optional
        The Axes object in which to draw the scatter plot.
    zorder : float, optional
    label : string, optional
        Text label to use in the legend
        (ignored if facecolor is not a scalar)
    **kwargs
        Keywords to be passed to `~matplotlib.pyplot.scatter`

    Returns
    -------
    ax : `~matplotlib.axes.Axes` object
        The Axes object in which contours are plotted.
    """

    # get plotting axis object if needed
    if ax is None:
        ax = plt.gca()

    # check and set axis scales
    if xscale not in ('log', 'linear'):
        raise ValueError("Unsupported 'xscale': " + xscale)
    ax.set_xscale(xscale)
    if yscale not in ('log', 'linear'):
        raise ValueError("Unsupported 'yscale': " + yscale)
    ax.set_yscale(yscale)

    # determine axis ranges if needed
    if xlim is None:
        xmax, xmin = np.nanmax(x), np.nanmin(x)
        if xscale == 'log':
            xoverscan = (xmax / xmin) ** 0.1
            xlim = (xmin / xoverscan, xmax * xoverscan)
        else:
            xoverscan = (xmax - xmin) * 0.1
            xlim = (xmin - xoverscan, xmax + xoverscan)
    ax.set_xlim(xlim)
    if ylim is None:
        ymax, ymin = np.nanmax(y), np.nanmin(y)
        if yscale == 'log':
            yoverscan = (ymax / ymin) ** 0.1
            ylim = (ymin / yoverscan, ymax * yoverscan)
        else:
            yoverscan = (ymax - ymin) * 0.1
            ylim = (ymin - yoverscan, ymax + yoverscan)
    ax.set_ylim(ylim)

    # color smoothing
    if (color_smoothing_box is not None) and (np.size(c) > 1):
        if xscale == 'log':
            x_ = np.log10(x)
            xbw = \
                color_smoothing_box[0] * np.log10(xlim[1] / xlim[0])
        else:
            x_ = x
            xbw = \
                color_smoothing_box[0] * (xlim[1] - xlim[0])
        if yscale == 'log':
            y_ = np.log10(y)
            ybw = \
                color_smoothing_box[1] * np.log10(ylim[1] / ylim[0])
        elif yscale == 'linear':
            y_ = y
            ybw = \
                color_smoothing_box[1] * (ylim[1] - ylim[0])
        newc = []
        if show_progress == 'notebook':
            bar = ProgressBar(range(len(x)), ipython_widget=True)
        elif show_progress:
            bar = ProgressBar(range(len(x)))
        else:
            bar = None
        for (x0_, y0_) in zip(x_, y_):
            newc.append(np.nanmedian(
                c[(x_ > x0_ - xbw/2) & (x_ < x0_ + xbw/2) &
                  (y_ > y0_ - ybw/2) & (y_ < y0_ + ybw/2)]))
            if show_progress:
                bar.update()
    else:
        newc = c

    # make scatter plot
    if edgewidth == 0:
        ax.scatter(
            x, y, marker=marker, c=newc, s=markersize**2,
            linewidths=0, zorder=zorder, **kwargs)
    else:
        ax.scatter(
            x, y, marker=marker, c=edgecolor,
            s=(markersize+edgewidth)**2,
            linewidths=0, zorder=zorder, **kwargs)
        ax.scatter(
            x, y, marker=marker, c=newc,
            s=(markersize-edgewidth)**2,
            linewidths=0, zorder=zorder, **kwargs)

    # add legend entry
    if label is not None:
        if np.size(c) > 1:
            print("Unable to add legend entry: `c` is not a scalar")
        else:
            ax.plot(
                [], [], marker=marker, mfc=c, mec=edgecolor,
                ms=markersize, mew=edgewidth, ls='', label=label)

    return ax


def density_contour(
        x, y, weights=None,
        xscale='log', yscale='log', xlim=None, ylim=None,
        overscan=(0.15, 0.15), binwidth=(0.01, 0.01), smoothbins=(5, 5),
        levels=(0.393, 0.865, 0.989), alphas=(0.75, 0.50, 0.25),
        color='k', contour_type='contourf', ax=None, **contourkw):
    """
    Generate data density contours.

    Parameters
    ----------
    x, y : array_like
        x & y coordinates of the data points
    weights : array_like, optional
        Statistical weight on each data point.
        To be passed to `~numpy.histogram2d`
    xscale, yscale : {'log', 'linear'}, optional
        Axes scales. Default is 'log' for both.
        They are also passed to the `~matplotlib.axes.Axes` object.
    xlim, ylim : array_like, optional
        Range to calculate and generate contour.
        Default is to use a range wider than the full data range
        by a factor of 'overscan' on both sides.
        They are also passed to the `~matplotlib.axes.Axes` object.
    overscan : array_like (length=2), optional
        Fraction by which 'xlim' and 'ylim' are wider than the full
        data range. Default is 0.15 for both axes, meaning 15% wider
        than the full data range on both sides.
    binwidth : array_like (length=2), optional
        Bin widths for generating the 2D histogram.
        Default is 0.01 for both axes, meaning 1% times 'xlim' and
        'ylim' (i.e., a 100 x 100 histogram will be generated).
    smoothbins : array_like (length=2), optional
        Number of bins to smooth over along both axes.
        To be passed to `~scipy.ndimage.gaussian_filter`
    levels : array_like, optional
        Contour levels to be plotted, specified as levels in CDF.
        Default is (0.393, 0.865, 0.989), which corresponds
        to the integral of a 2D normal distribution within 1-sigma,
        2-sigma, and 3-sigma range (i.e., Mahalanobis distance).
    alphas : array_like, optional
        Transparency of the contours. Default is (0.75, 0.50, 0.25).
        The length of 'alphas' should match that of 'levels'.
    color : mpl color, optional
        Base color of the contours. Default: 'k'
    contour_type : {'contour', 'contourf'}, optional
        Contour drawing function to call
    ax : `~matplotlib.axes.Axes` object, optional
        The Axes object to plot contours in.
    **contourkw
        Keywords to be passed to the contour drawing function
        (see keyword "contour_type")

    Returns
    -------
    ax : `~matplotlib.axes.Axes` object
        The Axes object in which contours are plotted.
    """

    # get plotting axis object if needed
    if ax is None:
        ax = plt.gca()

    # check and set axis scales
    if xscale not in ('log', 'linear'):
        raise ValueError("Unsupported 'xscale': " + xscale)
    ax.set_xscale(xscale)
    if yscale not in ('log', 'linear'):
        raise ValueError("Unsupported 'yscale': " + yscale)
    ax.set_yscale(yscale)

    # find full axis ranges for the 2D histogram
    if xlim is None:
        xmax, xmin = np.nanmax(x), np.nanmin(x)
        if xscale == 'log':
            xoverscan = (xmax / xmin) ** overscan[0]
            xlim = (xmin / xoverscan, xmax * xoverscan)
        else:
            xoverscan = (xmax - xmin) * overscan[0]
            xlim = (xmin - xoverscan, xmax + xoverscan)
    ax.set_xlim(xlim)
    if ylim is None:
        ymax, ymin = np.nanmax(y), np.nanmin(y)
        if yscale == 'log':
            yoverscan = (ymax / ymin) ** overscan[1]
            ylim = (ymin / yoverscan, ymax * yoverscan)
        else:
            yoverscan = (ymax - ymin) * overscan[1]
            ylim = (ymin - yoverscan, ymax + yoverscan)
    ax.set_ylim(ylim)

    # generate bins for the 2D histogram
    if xscale == 'log':
        xbwlog = binwidth[0] * np.log10(xlim[1] / xlim[0])
        xbinslog = np.arange(
            np.log10(xlim[0]), np.log10(xlim[1])+xbwlog, xbwlog)
        xmids = 10**(xbinslog[:-1] + 0.5*xbwlog)
    else:
        xbwlin = binwidth[0] * (xlim[1] - xlim[0])
        xbinslin = np.arange(
            xlim[0], xlim[1]+xbwlin, xbwlin)
        xmids = xbinslin[:-1] + 0.5*xbwlin
    if yscale == 'log':
        ybwlog = binwidth[1] * np.log10(ylim[1] / ylim[0])
        ybinslog = np.arange(
            np.log10(ylim[0]), np.log10(ylim[1])+ybwlog, ybwlog)
        ymids = 10**(ybinslog[:-1] + 0.5*ybwlog)
    else:
        ybwlin = binwidth[1] * (ylim[1] - ylim[0])
        ybinslin = np.arange(
            ylim[0], ylim[1]+ybwlin, ybwlin)
        ymids = ybinslin[:-1] + 0.5*ybwlin

    # generate 2D histogram
    if xscale == 'log' and yscale == 'log':
        hist, _, _ = np.histogram2d(
            np.log10(x), np.log10(y), weights=weights,
            bins=[xbinslog, ybinslog])
    elif xscale == 'log' and yscale == 'linear':
        hist, _, _ = np.histogram2d(
            np.log10(x), y, weights=weights,
            bins=[xbinslog, ybinslin])
    elif xscale == 'linear' and yscale == 'log':
        hist, _, _ = np.histogram2d(
            x, np.log10(y), weights=weights,
            bins=[xbinslin, ybinslog])
    else:
        hist, _, _ = np.histogram2d(
            x, y, weights=weights,
            bins=[xbinslin, ybinslin])

    # smooth 2D histogram
    pdf = gaussian_filter(hist, smoothbins).T

    # calculate cumulative density distribution (CDF)
    cdf = np.zeros_like(pdf).ravel()
    for i, density in enumerate(pdf.ravel()):
        cdf[i] = pdf[pdf >= density].sum()
    cdf = (cdf/cdf.max()).reshape(pdf.shape)

    # plot contourf
    if contour_type == 'contour':
        contourfunc = ax.contour
        contourlevels = levels
    elif contour_type == 'contourf':
        contourfunc = ax.contourf
        contourlevels = np.hstack([[0], levels])
    else:
        raise ValueError(
            "'contour_type' should be either 'contour' or 'contourf'")
    contourfunc(
        xmids, ymids, cdf, contourlevels,
        colors=[mplcolors.to_rgba(color, a) for a in alphas],
        **contourkw)

    return ax


def minimal_bar_plot(
        seq, percent=[16., 50., 84.], pos=None,
        colors=None, labels=None, labelloc='up', labelpad=0.1,
        minimal=True, ax=None, barkw={}, labelkw={}, **symkw):
    """
    Generate bar plot, in the spirit of minimalism.

    Parameters
    ----------
    seq : sequence of array_like
        Samples to be represented by the bar plot.
    percent : array_like of floats between 0 and 100, optional
        Percentiles to compute, default: [16., 50., 84.]
    pos : array_like of floats, optional
        Positions at which to plot the bars for each sample.
        Default: np.arange(len(seq)) + 1
    colors : array_like of colors, optional
        Colors to use for each sample, default: 'k'
    labels : array_like of strings, optional
        Labels for each sample, default is no label.
    labelloc : {'up', 'down'}, optional
        Locations to put label relative to bar, default: 'up'
    labelpad : float, optional
        Pad (in data unit) between labels and bars, default: 0.1
    minimal : bool, optional
        Adjust plot design in the spirit of minimalism (default: True)
    ax : `~matplotlib.axes.Axes` object, optional
        Overplot onto the provided axis object.
        If not available, a new axis will be created.
    barkw : dict, optional
        Keyword arguments to control the behavior of the bars.
        Will be passed to `~matplotlib.axes.Axes.hlines`.
    labelkw : dict, optional
        Keyword arguments to control the behavior of the labels.
        Will be passed to `~matplotlib.axes.Axes.text`.
    **symkw
        Keyword arguments to control the behavior of the symbols.
        Will be passed to `~matplotlib.axes.Axes.plot`.

    Returns
    -------
    ax : `~matplotlib.axes.Axes` object
        The Axes object in which the bar plot is created.
    """

    percentiles = np.array([np.percentile(x, percent) for x in seq])
    nsample, nlim = percentiles.shape

    if pos is None:
        pos = np.arange(nsample) + 1
    else:
        pos = np.atleast_1d(pos)
    if colors is None:
        colors = ['k'] * nsample
    else:
        colors = np.atleast_1d(colors)
    if labels is None:
        labels = [''] * nsample
    else:
        labels = np.atleast_1d(labels)
    if labelloc == 'up':
        ha = 'center'
        va = 'bottom'
        dy = labelpad
    elif labelloc == 'down':
        ha = 'center'
        va = 'top'
        dy = -labelpad
    else:
        raise ValueError(
            "`labelloc` must be either 'top' or 'bottom'")

    if ax is None:
        ax = plt.gca()
    lw = 0.
    ibar = -1
    for ibar in range(nlim // 2):
        lw += plt.rcParams['lines.linewidth']
        ax.hlines(
            pos, percentiles[:, ibar], percentiles[:, -ibar-1],
            colors=colors, linewidth=lw, **barkw)
    if nlim % 2 == 1:
        for isample in range(nsample):
            ax.plot(
                percentiles[isample, ibar+1], pos[isample],
                marker='o', ms=lw+2, mfc='w',
                mew=lw, mec=colors[isample],
                lw=0.0, **symkw)
            ax.text(
                percentiles[isample, ibar+1], pos[isample] + dy,
                labels[isample], ha=ha, va=va,
                color=colors[isample], **labelkw)
    else:
        for isample in range(nsample):
            ax.text(
                percentiles[isample, ibar:ibar+2].mean(),
                pos[isample] + dy,
                labels[isample], ha=ha, va=va,
                color=colors[isample], **labelkw)

    if minimal:
        ax.tick_params(
            axis='both', left='off', top='off', right='off',
            labelleft='off', labeltop='off', labelright='off')
        for side in ['top', 'left', 'right']:
            ax.spines[side].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_smart_bounds(True)

    return ax
