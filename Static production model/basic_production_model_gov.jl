
using PyPlot
export @L_str, Axes3D, ColorMap, Figure, LaTeXString, PyPlot, acorr, annotate, arrow, art3D, autoscale, autumn, axhline, axhspan, axis, axvline, axvspan, bar, bar3D, barbs, barh, bone, box, boxplot, broken_barh, cla, clabel, clf, clim, cohere, colorbar, colors, contour, contour3D, contourf, contourf3D, cool, copper, csd, delaxes, disconnect, draw, errorbar, eventplot, figaspect, figimage, figlegend, figtext, figure, fill_between, fill_betweenx, findobj, flag, gca, gcf, gci, get_cmap, get_cmaps, get_current_fig_manager, get_figlabels, get_fignums, get_plot_commands, getindex, ginput, gray, grid, hexbin, hist, hist2D, hlines, hold, hot, hsv, imread, imsave, imshow, ioff, ion, ishold, jet, keys, latexstring, legend, locator_params, loglog, margins, matplotlib, matshow, mesh, minorticks_off, minorticks_on, over, pause, pcolor, pcolormesh, pie, pink, plot, plot3D, plot_date, plot_surface, plot_trisurf, plot_wireframe, plotfile, plt, polar, prism, psd, pygui, quiver, quiverkey, rc, rc_context, rcdefaults, register_cmap, rgrids, savefig, sca, scatter, scatter3D, sci, semilogx, semilogy, set_cmap, setindex!, setp, show, specgram, spectral, spring, spy, stackplot, stem, step, streamplot, subplot, subplot2grid, subplot_tool, subplots, subplots_adjust, summer, suptitle, surf, table, text, text2D, text3D, thetagrids, tick_params, ticklabel_format, tight_layout, title, tricontour, tricontourf, tripcolor, triplot, twinx, twiny, using3D, vlines, waitforbuttonpress, winter, withfig, xkcd, xlabel, xlim, xscale, xticks, ylabel, ylim, yscale, yticks, zlabel, zlim, zscale, zticks
using LaTeXStrings, KernelDensity
using Parameters, CSV, Random, QuantEcon
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim, Interpolations

@with_kw mutable struct Para

    α::Array{Float64, 1} = [0.3, 0.4]  # 
    β_1::Float64 = 0.3   # discount factor
    β_2::Float64 = 0.6
    Kbar::Float64 = 20
    Tbar::Float64 = 30
    G::Float64 = 3
    # Taxes
    tauc::Array{Float64, 1} = [0.0, 0.0]
    tauw::Float64 = 0.0
    taur::Float64 = 0.0

    #u = (L_1, K_1) -> (L_1^β_1*K_1^(1-β_1))^α*((Lbar-L_1)^β_2*(Kbar-K_1)^(1-β_2))^(1-α)
end

function markets(x, para)
    @unpack α, β_1, β_2, Kbar, Tbar, G, tauc, tauw, taur = para
    q = zeros(2)

    q[1] = 1.0
    q[2] = x[1]
    w = x[2]
    r = x[3]
    # One tax adjusts to clear markets
    taur = x[4]

    # Consumer prices and total income
    p = q.*(1.0 .+ tauc)
    wn = w.*(1.0 .- tauw)
    rn = r.*(1.0 .- taur)
    Ybarn = wn*Tbar + rn*Kbar

    out = similar(x)

    out[1] = α[1]*Ybarn/p[1]+ G - (β_1/w)^(β_1)*((1-β_1)/r)^(1-β_1)*q[1]*(α[1]*Ybarn/p[1]+G) 
    out[2] = 1.0 - (β_2/w)^β_2*((1-β_2)/r)^(1-β_2)q[2]
    out[3] = (β_1/w)*q[1]*(α[1]*Ybarn/p[1]+G) + (β_2/w)*q[2]*α[2]*Ybarn/p[2] + (1-α[1]-α[2])/wn*Ybarn - Tbar
    out[4] = (q[1]*G - tauc[1]/(1.0+tauc[1])*α[1]*Ybarn- tauc[2]/(1.0+tauc[2])*α[2]*Ybarn-
            tauw*w*(Tbar-(1.0-α[1]-α[2])/wn*Ybarn)-taur*r*Kbar)
    
    X_1 = α[1]*Ybarn/p[1]
    X_2 = α[2]*Ybarn/p[2]
    l = (1-α[1]-α[2])*Ybarn/wn
    u = X_1^(α[1])*X_2^(α[2])l^(1-α[1]-α[2])

    return out, q, w, r, p, wn, rn, Ybarn, X_1, X_2, l, u, G
end
    # Market equations

para = Para(Kbar=10)
f(x) = markets(x, para)[1]
x0 = [0.5; 0.5; 0.5; 0.5]
res = nlsolve(f, x0)
 out, q, w, r, p, wn, rn, Ybarn, X_1, X_2, l, u, G = markets(res.zero, para)
@show X_1, G, q[1]
@show X_2, q[2]
@show u
# Calculate other economic variables


