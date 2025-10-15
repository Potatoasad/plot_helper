import os
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import style
import jax.numpy as jnp
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from matplotlib.lines import Line2D
import scipy
from .makeCorner import getBounds

style.use(os.path.dirname(os.path.realpath(__file__))+'/plotting.mplstyle')

default_pallete = sns.color_palette('Dark2', 20)

def makedict(x):
    return {k : jnp.array(x[k].values) for k in x.columns}

def make_custom_legend(ax, names, colors):
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(colors))]
    ax.legend(custom_lines, names)
    return ax

def get_means_and_sigs(loc, scale, a=0, b=1):
    a_l, b_l = (a - loc) / scale, (b - loc) / scale
    mu_s = scipy.stats.truncnorm.mean(loc=loc, scale=scale, a=a_l, b=b_l)
    sigma_s = scipy.stats.truncnorm.std(loc=loc, scale=scale, a=a_l, b=b_l)
    return mu_s, sigma_s

def set_multicolor_title(ax, title_strings, title_colors, font_size=20, font_weight='bold'):
    """
    Sets a multicolor title on a given axis, with each string equally spaced horizontally.

    Parameters:
    - ax: The axis object to which the title will be added.
    - title_strings: A list of strings, each to be displayed in a different color.
    - title_colors: A list of colors corresponding to each string.
    - font_size: Font size of the title text.
    """
    if len(title_strings) != len(title_colors):
        raise ValueError("title_strings and title_colors must have the same length.")

    # Clear any existing title
    ax.set_title("")

    # Get number of strings
    n_strings = len(title_strings)
    
    # Set the position for the title in axis coordinates (x from 0 to 1)
    title_y_pos = 1.05  # Slightly above the plot (in axis coordinates)
    
    # Divide the space equally for each string
    if n_strings == 1:
        x_positions = [1/2 for i in range(1, n_strings + 1)]
    elif n_strings == 2:
        x_positions = [0.33-0.1, 0.66+0.1]
    else:
        x_positions = [(i-(i/n_strings)) / (n_strings) for i in range(1, n_strings + 1)]
    
    # Add each string at its corresponding position
    for x_pos, text, color in zip(x_positions, title_strings, title_colors):
        ax.text(x_pos, title_y_pos, text, fontsize=font_size, color=color,
                ha='center', va='bottom', transform=ax.transAxes)

def add_contours(ax, x,y,a,b, color=default_pallete[0], linestyles='solid', 
                linewidths=2, quantiles=[0.9, 0.5], fill=True):
    from truncnormkde import BoundedKDE, compute_bandwidth
    from scipy import interpolate
    X = jnp.stack([x,y], axis=-1)
    
    bw = compute_bandwidth(X)
    bkde = BoundedKDE(a, b, bw);
    gridsize = 100
    
    xs,ys = jnp.linspace(a[0],b[0],gridsize), jnp.linspace(a[1],b[1],gridsize)
    x_2d, y_2d = jnp.meshgrid(xs,ys)
    X_grid = jnp.stack([x_2d, y_2d],axis=-1)
    
    KDE = BoundedKDE(a=a, b=b, bandwidth=bw)
    computed_values = KDE(X_grid, X)

    dx = (b[0] - a[0])/gridsize;
    dy = (b[1] - a[1])/gridsize;
    ts = np.linspace(0,computed_values.max(),gridsize)
    quantils = ((computed_values[:,:,None] > ts[None, None, :]) * computed_values[:,:,None]).sum(axis=(0,1)) * dx * dy
    
    f = interpolate.interp1d(quantils, ts)
    t_contours = f(np.array(quantiles))
    
    quantile_plot = ax.contour(x_2d, y_2d, computed_values, colors=[color]*len(t_contours), 
                                levels=t_contours, linewidths=[linewidths]*len(t_contours))

    if fill:
        cmax = computed_values.max()
        for i, t_contour in enumerate(t_contours[::-1]):  # Reverse to fill from outer to inner quantile
                ax.contourf(x_2d, y_2d, computed_values, levels=[t_contour, cmax], colors=[color], alpha=quantiles[i])

        # Fill the region outside the last quantile with transparency
        cont = ax.contourf(x_2d, y_2d, computed_values, levels=[0, t_contours[0]], colors=[color], alpha=0)

def make_corner_plot(all_data : List,
                     model_labels : List[str],
                     variables : List[str],
                     variable_labels : Optional[List[str]] = None,
                     limits : Optional[List[Tuple]] = None,
                     nbins=20,
                     cp=default_pallete,
                     colors=[default_pallete[1], default_pallete[2], 'black'],
                     linestyles=['solid', 'solid', 'dashed'], alpha=0.02, 
                     kde=False, scatter=True, kde_kwargs={}, scatter_kwargs={},
                     legend_x_position=None, legend_y_position=None, CI_fontsize=15,
                     boundary_bias=False, fill=True, quantiles=[0.9, 0.5], legend=True, 
                     boundaries={},
                     truth=None):

    if variable_labels is None:
        variable_labels = variables

    labels = variable_labels

    if limits is None:
        limits = [None for _ in range(len(variables))]

    for k,var in enumerate(variables):
        the_min = min([np.min(data[var]) for data in all_data])
        the_max = max([np.max(data[var]) for data in all_data])
        if limits[k] is None:
            limits[k] = (the_min, the_max)

     # Make figure 
    nVars = len(variables)
    fig, axes = plt.subplots(nVars, nVars, figsize=(nVars*3,nVars*3))

    one_d_plot = False
    #print(type(axes))
    if type(axes) != type(np.array([])): #assume its a 1d plot
        one_d_plot = True
    #print(one_d_plot, nVars)

    # First, plot 1D histograms along diagonal
    for i, var in enumerate(variables):
        
        #print(f'Plotting 1d hist for {var}')

        # Get appropriate axis
        if one_d_plot:
            ax = axes
        else:
            ax = axes[i, i]
        ax.set_rasterization_zorder(2)

        # Get limits 
        min_lim = limits[i][0]
        max_lim = limits[i][1]
        
        # Get label
        label = labels[i]
        
        # Cycle through data sets
        CIs = [];
        text_colors = [];
        for j,data in enumerate(all_data): 
            
            if var in data.keys():

                # Fetch data for this variable
                samples = np.asarray(data[var])
                
                # Plot histogram
                ax.hist(samples, bins=np.linspace(min_lim,max_lim,nbins), histtype='step', 
                            edgecolor=colors[j], lw=2, linestyle=linestyles[j], density=True, 
                            zorder=2)
                CIs.append(r"${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(*getBounds(samples)))
                text_colors.append(colors[j])
        
        #ax.set_title("  ".join(CIs), fontsize=CI_fontsize)
        set_multicolor_title(ax, CIs, text_colors)
        ax.xaxis.grid(True,which='major',ls=':',color='grey',alpha=0.5)
        ax.yaxis.grid(True,which='major',ls=':',color='grey',alpha=0.5)
        ax.set_xlim(min_lim,max_lim)
        ax.set_yticklabels([])
        ax.tick_params(labelsize=18)
        if truth is not None:
            ax.axvline(truth[var], color=colors[j])

        # x-axis + tick labels depend on which specific axis 
        if i==(nVars-1):
            ax.set_xlabel(label,fontsize=20)
        else: 
            ax.set_xticklabels([])

    # Then, fill in 2D histograms
    for i_row in np.arange(nVars): 
        for i_col in np.arange(nVars):
            if one_d_plot:
                continue

            # Get appropriate axis
            ax = axes[i_row, i_col]
            ax.set_rasterization_zorder(2)

            # Limits
            min_lim_x = limits[i_col][0]
            max_lim_x = limits[i_col][1]
            min_lim_y = limits[i_row][0]
            max_lim_y = limits[i_row][1]
            
            # Labels
            label_x = labels[i_col]
            label_y = labels[i_row]
            
            # Make plots right of diagonal invisible
            if i_col>i_row: 
                ax.axis('off')

            # Add 2D hist to left of diagonal     
            elif i_row>i_col:
                #print(f'Plotting 2d hist for {variables[i_col]} and {variables[i_row]}')
                for j,data in enumerate(all_data):
                    if variables[i_col] in data.keys() and variables[i_row] in data.keys():
                        # Plot histogram
                        if kde:
                            if not boundary_bias:
                                quantiles_sorted = quantiles.copy()
                                quantiles_sorted.sort()
                                if quantiles_sorted[-1] != 1:
                                    quantiles_sorted_fill = quantiles_sorted + [1]
                                else:
                                    quantiles_sorted_fill = quantiles_sorted.copy()
                                g = sns.kdeplot(data, x=variables[i_col], y=variables[i_row], 
                                                ax=ax, color=colors[j], 
                                                linestyles=linestyles[j],
                                                linewidths=2,
                                                levels=quantiles_sorted_fill,
                                                fill=fill,
                                                zorder=2, **kde_kwargs)
                                if fill:
                                    g = sns.kdeplot(data, x=variables[i_col], y=variables[i_row], 
                                                    ax=ax, color=colors[j], 
                                                    linestyles=linestyles[j], 
                                                    linewidths=2,
                                                    levels=quantiles_sorted,
                                                    fill=False,
                                                    zorder=2, **{k:v for k,v in kde_kwargs.items() if k != 'alpha'})
                                g.set(xlabel=None); g.set(ylabel=None);
                            else:
                                quantiles_sorted = quantiles.copy()
                                quantiles_sorted.sort(reverse=True)
                                x_lims = boundaries.get(variables[i_col], [min_lim_x, max_lim_x])
                                y_lims = boundaries.get(variables[i_row], [min_lim_y, max_lim_y])
                                add_contours(ax, x=data[variables[i_col]], 
                                                 y=data[variables[i_row]], 
                                                 a=np.array([x_lims[0], y_lims[0]]), 
                                                 b=np.array([x_lims[1], y_lims[1]]),
                                                 color=colors[j], linestyles=linestyles[j],
                                                 linewidths=2, quantiles=quantiles_sorted, fill=fill)
                                ax.set_xlabel(None); ax.set_ylabel(None);

                        if scatter:
                            if 'alpha' in scatter_kwargs.keys():
                                alpha_ = scatter_kwargs['alpha']
                            else:
                                alpha_ = alpha
                            ax.scatter(data[variables[i_col]], data[variables[i_row]], 
                                            color=colors[j], alpha=alpha_,
                                            zorder=2, **{k:v for k,v in scatter_kwargs.items() if k != 'alpha'})

                        if truth is not None:
                            ax.scatter([truth[variables[i_col]]], [truth[variables[i_row]]], 
                                            color=colors[j], alpha=1,
                                            zorder=2, **{k:v for k,v in scatter_kwargs.items() if k != 'alpha'})

                        
                ax.xaxis.grid(True,which='major',ls=':',color='grey',alpha=0.5)
                ax.yaxis.grid(True,which='major',ls=':',color='grey',alpha=0.5)
                ax.set_xlim(min_lim_x,max_lim_x)
                ax.set_ylim(min_lim_y,max_lim_y)
                ax.tick_params(labelsize=18)

                # labels depend on which specific axis
                if i_row==nVars-1:
                    ax.set_xlabel(label_x,fontsize=20)
                if i_col==0: 
                    ax.set_ylabel(label_y,fontsize=20)

                # ticks also depend on which specific axis
                if i_col!=0 and i_row!=nVars-1:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                elif i_col!=0 and i_row==nVars-1: 
                    ax.set_yticklabels([])
                elif i_col==0 and i_row!=nVars-1: 
                    ax.set_xticklabels([])
                    
    # Legend
    if legend and not one_d_plot:
        handles = [Line2D([], [], color=colors[i],ls=linestyles[i],label=model_labels[i]) for i in range(len(model_labels))]
        if legend_x_position is None:
            legend_x_position = min([axes.shape[-1]-1, 4]);
        if legend_y_position is None:
            legend_y_position = 1;
        axes[legend_y_position, legend_x_position].legend(handles=handles, loc='upper left', fontsize=20)

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig, axes


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
#from truncatedgaussianmixtures import fit_gmm, jl, fit_kde, jlarray

def make_2D_comparison(posterior_samples, 
                       posterior_samples_no_190517=None, 
                       variables=['chi_1', 'chi_2'],
                       variable_labels=[r'$\chi_1$', r'$\chi_2$'],
                       a=[0,0], b=[1,1], model_labels = ['All Events', 'w/o GW190517'],
                       quantiles=[0.9,0.5,0.1], legend_location='upper right', 
                       bandwidth_scale=1, legend=True, legend_face_color=(1,1,1,0.1)):
    from truncatedgaussianmixtures import fit_gmm, jl, fit_kde, jlarray

    x_name = variables[0]; y_name = variables[1];
    fit_eta = fit_kde(posterior_samples[[x_name]], [a[0]], [b[0]], bandwidth_scale=bandwidth_scale);
    fit_sigma = fit_kde(posterior_samples[[y_name]], [a[1]], [b[1]], bandwidth_scale=bandwidth_scale);

    if posterior_samples_no_190517 is not None:
        fit_eta_2 = fit_kde(posterior_samples_no_190517[[x_name]], [a[0]], [b[0]]);
        fit_sigma_2 = fit_kde(posterior_samples_no_190517[[y_name]], [a[1]], [b[1]]);
    
    # Create a figure
    fig = plt.figure(figsize=(6,6),dpi=200)
    
    # Define the grid spec with different widths and heights for the histograms and scatter plot
    gs = gridspec.GridSpec(2, 2, 
                           width_ratios=[0.8, 0.2],  # Make the diagonal (histogram) on the right narrower
                           height_ratios=[0.2, 0.8]) # Make the diagonal (histogram) on the top shorter
    
    # Scatter plot (off-diagonal, bottom-left corner)
    ax_scatter = fig.add_subplot(gs[1, 0])
    
    add_contours(ax_scatter, posterior_samples[x_name].values, 
                 posterior_samples[y_name].values, 
                 a=np.array(a), b=np.array(b), quantiles=quantiles, 
                color=default_pallete[1])
    
    # X-axis histogram (top-left corner)
    eta_0s = np.linspace(a[0],b[0],100)
    pdf_etas = fit_eta.pdf(eta_0s.reshape(100,1))
    
    ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_hist_x.plot(eta_0s, pdf_etas, color=default_pallete[1])
    ax_hist_x.fill_between(eta_0s, pdf_etas, alpha=0.1, color=default_pallete[1])
    
    # Y-axis histogram (right, rotated 90 degrees)
    sigma_0s = np.linspace(a[1],b[1],100)
    pdf_sigmas = fit_sigma.pdf(sigma_0s.reshape(100,1))
    
    ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    ax_hist_y.plot(pdf_sigmas, sigma_0s, color=default_pallete[1])
    ax_hist_y.fill_betweenx(sigma_0s, pdf_sigmas, alpha=0.1, color=default_pallete[1])

    if posterior_samples_no_190517 is not None:
        add_contours(ax_scatter, posterior_samples_no_190517[x_name].values, 
                     posterior_samples_no_190517[y_name].values, 
                     a=np.array(a), b=np.array(b), quantiles=quantiles, 
                    color=default_pallete[2])
        
        # X-axis histogram (top-left corner)
        pdf_etas2 = fit_eta_2.pdf(eta_0s.reshape(100,1))
        
        #ax_hist_x = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
        ax_hist_x.plot(eta_0s, pdf_etas2, color=default_pallete[2])
        ax_hist_x.fill_between(eta_0s, pdf_etas2, alpha=0.1, color=default_pallete[2])
        
        # Y-axis histogram (right, rotated 90 degrees)
        pdf_sigmas2 = fit_sigma_2.pdf(sigma_0s.reshape(100,1))
        
        #ax_hist_y = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
        ax_hist_y.plot(pdf_sigmas2, sigma_0s, color=default_pallete[2])
        ax_hist_y.fill_betweenx(sigma_0s, pdf_sigmas2, alpha=0.1, color=default_pallete[2])
        
        # Adjust labels
        ax_hist_x.set_ylim(0,max([max(pdf_etas), max(pdf_etas2)])*1.1)
        ax_hist_y.set_xlim(1e-2,max([max(pdf_sigmas), max(pdf_sigmas2)])*1.1)
    else:
        ax_hist_x.set_ylim(0,max([max(pdf_etas)])*1.1)
        ax_hist_y.set_xlim(1e-2,max([max(pdf_sigmas)])*1.1)
    
    plt.setp(ax_hist_x.get_xticklabels(), visible=False)  # Hide x-axis tick labels on the top histogram
    plt.setp(ax_hist_y.get_yticklabels(), visible=False)  # Hide y-axis tick labels on the right histogram
    plt.setp(ax_hist_x.get_yticklabels(), visible=False)    # Rotate x-tick labels on the right histogram
    plt.setp(ax_hist_y.get_xticklabels(), visible=False)    # Rotate x-tick labels on the right histogram
    
    ax_scatter.set_xlabel(variable_labels[0])
    ax_scatter.set_ylabel(variable_labels[1])

    if posterior_samples_no_190517 is None:
        N_plots = 1
    else:
        N_plots = 2
    if legend:
        handles = [Line2D([], [], color=default_pallete[i+1],ls='solid',label=model_labels[i]) for i in range(N_plots)]
        thelegend = ax_scatter.legend(handles=handles, loc=legend_location, fontsize=15, frameon=True)
        thelegend.get_frame().set_facecolor(legend_face_color)
        thelegend.get_frame().set_edgecolor('none')
    # Tighten the layout
    plt.tight_layout()
    plt.show()

    return fig