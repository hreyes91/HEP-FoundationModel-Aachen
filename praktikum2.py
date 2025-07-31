__all__ = ['pd', 'np', 'plt', 'pprint', 'color', 're', 'tex2uni', 'integrate', 'curve_fit', 'c', 'stats', 'CubicSpline', 'u', 'ufloat', 'split', 'ev', 'std', 'tag', 'weighted_mean', 'plots_path', 'latex_path', 'plot', 'add_value', 'format_number', 'table', 'uprint', 'linreg', 'linreg2', 'N_gaussian', 'fit_N_gaussian', 'general_func', 'general_fit', 'plot_general_fit', 'sqrt', 'pi', 'abs', 'sin', 'cos', 'tan', 'arccos', 'arcsin', 'arctan', 'exp', 'log', 'deg2rad', 'rad2deg', 'vectorize', 'un', 'unp', 'flat', 'data_path', 'residual_plot', 'fn', 'nonlinfit', 'language', 'Iterable', 'plotter']

from numpy import sqrt,pi,abs,sin,cos,tan,arccos,arcsin,arctan,exp,log,deg2rad,rad2deg,vectorize
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.labelsize": "large",
        "text.usetex": False,
        "font.size": 13,
    }
)

from praktikum.analyse import lineare_regression
from praktikum import cassy

import lmfit

from collections.abc import Iterable

from pprint import pprint

import ansi.color as color

import tabulate as __tabulate__
__tabulate__.PRESERVE_WHITESPACE = True
from tabulate import tabulate

import re

from typing import List, Tuple

from pylatexenc.latex2text import LatexNodes2Text

def tex2uni(x, all=True):
    """
    string: str     latex notation in it will interpreted to unicode
    all   : bool    if True the whole string will be searched for latex notation, which can have unexpexted effects, if False only text in enclosed by $...$ will be converted
    """
    exp_map = str.maketrans("0123456789-","⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    def replace_exponent(x):
        match = re.search(r"\^\{\+?(-?\d+)\}", x)
        if match:
            exp = match.group(1).translate(exp_map)
            x = re.sub(r"\^\{\+?(-?\d+)\}", exp, x)
            x = replace_exponent(x)
        return x 
    
    x = replace_exponent(x)
    l2t = LatexNodes2Text().latex_to_text
    if all:
        return l2t(x)
    return re.sub(r"(\$.+?\$)", lambda x: l2t(x.group(1)), x)

import os

import scipy.integrate as integrate
from scipy.optimize import curve_fit
import scipy.constants as c
import scipy.stats as stats
import scipy.odr as odr
from scipy.interpolate import CubicSpline

import itertools

flat = lambda x: list(itertools.chain.from_iterable(x))

import uncertainties as un
from uncertainties import ufloat, UFloat
import uncertainties.umath as umath
import uncertainties.unumpy as unp

un.UFloat.__float__ = lambda self: self.n
un.Variable.__float__ = lambda self: self.n

def u(mu, sigma, tag=None):
    return np.vectorize(lambda m, s, tag: un.ufloat(m, s, tag=tag))(mu, sigma, tag)

ev = np.vectorize(
    lambda x: x.n if type(x) == un.Variable or type(x) == un.UFloat else x,
    otypes=[float],
)  # ev(1+/-0.1)=1, ev(1)=[1]
std = np.vectorize(
    lambda x: x.s if type(x) == un.Variable or type(x) == un.UFloat else 0,
    otypes=[float],
)  # std(1+/-0.1)=0.1, std(1)=[0]
split = lambda X: (ev(X), std(X))
scale_std = lambda x, a, tag=None: np.vectorize(u(ev(x), a * std(x), tag=tag if tag else x.tag))
tag = lambda x, tag: u(*split(x), tag=tag)
weighted_mean = lambda x: u(
    np.sum(ev(x) / std(x) ** 2) / np.sum(1 / std(x) ** 2),
    np.sqrt(1 / np.sum(1 / std(x) ** 2)),
)  # sum(mu_i / sigma_i**2) / sum(1/sigma_i**2)

plots_path = "./plots/"
latex_path = "./latex/"
data_path = "./data/"
language = "de" # de/en
 
def plot(f):
    def wrapper(*args, **kwargs_):
        savefig, show, legend, close, subplot = None, None, None, False, False
        passed_kwargs = {}
        
        if "ax" in kwargs_:
            plt.sca(kwargs_["ax"])
            kwargs_.pop("ax")
            subplot = True
            
        if "figsize" in kwargs_:
            plt.figure(figsize=kwargs_["figsize"])
            kwargs_.pop("figsize")

        for key, value in kwargs_.items():
            if key == "xlim":
                plt.xlim(*value)
            elif key == "ylim":
                plt.ylim(*value)
            elif key == "xscale":
                plt.xscale(value)
            elif key == "yscale":
                plt.yscale(value)
            elif key == "xlabel":
                plt.xlabel(value)
            elif key == "ylabel":
                plt.ylabel(value)
            elif key == "title":
                plt.title(value)
            elif key == "bftitle":
                plt.title(r"$\mathbf{" + value.replace(" ", r"\ ") + r"}$")
            elif key == "suptitle":
                plt.suptitle(value)
            elif key == "bfsuptitle":
                plt.suptitle(r"$\mathbf{" + value.replace(" ", r"\ ") + r"}$")
            elif key == "grid":
                plt.grid(linestyle="--", alpha=0.5)
            elif key == "legend":
                legend = value
            elif key in ("savefig", "filename"):
                savefig = value
            elif key == "show":
                show = value
            elif key == "close":
                close = value
            else:
                passed_kwargs[key] = value

        res = f(*args, **passed_kwargs)

        if legend is True:
            plt.legend()
        elif legend is not None:
            plt.legend(loc=legend)

        plt.tight_layout()
        if savefig:
            plt.savefig(plots_path + savefig)
            close=True

        if (show is True) or (show is None and subplot is False):
            plt.show()
            close=True
        
        if close:
            plt.close()

        return res
    
    return wrapper

plotter = lambda **kwargs: plot(lambda: ...)(**{**kwargs, "show": False, "close": False})

def pretty_exponent(x):
    exp_map = str.maketrans("0123456789-","⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    
    # latex to unicode
    match = re.search(r"\^\{\+?(-?\d+)\}", x)
    if match:
        exp = match.group(1).translate(exp_map)
        x = re.sub(r"\^\{\+?(-?\d+)\}", exp, x)
        x = pretty_exponent(x)
        
    # e-13 to unicode 
    match = re.search(r"e\+?0*(-?\d+)", x)
    if match:
        exp = match.group(1).translate(exp_map)
        x = re.sub(r"e\+?0*(-?\d+)", exp, x)
        x = pretty_exponent(x)
    
    return x 

def format_number(x, fmt=None, udigits=2, fdigits=3, fformat="nice", latex=False):
    if fmt is not None:
        return x.format(fmt)
    elif isinstance(x,str): 
        return x
    elif isinstance(x, un.UFloat) or isinstance(x, un.Variable):
        res = f"{x:.3}"
        return "$"+res+"$" if latex else res
    elif isinstance(x, (float, np.floating)):
        return pretty_exponent(f"{x:.3}")
    else:
        return str(x)

fn = format_number

def uprint(
    x,
    name=None,
    title_name=None,
    add_val=True,
    add_file=True,
    digits=2,
    show=True,
    show_latex=False,
    bracket_notation=True,
):
    if type(x) == np.ndarray and x.size == 1:
        x = x.item()
    if add_val and name:
        add_value(name, x, show=True, udigits=2, fdigits=5)
    error_dict = x.error_components()
    key_values, key_str, values = [], [], []

    for key, value in error_dict.items():
        if key.tag:
            key_str.append(key.tag)
        else:
            key_str.append("<no tag>")
        key_values.append(key)
        values.append(value)

    sorted_indices = np.argsort(np.abs(values))[::-1]
    key_values = np.array(key_values)[sorted_indices]
    key_str = np.array(key_str)[sorted_indices]
    values = np.array(values)[sorted_indices]

    rel_error = np.abs(np.divide(std(key_values), ev(key_values)) * 100)  # %

    if type(x) == un.core.AffineScalarFunc:
        print((color.fg.boldred(f"<Error components for {title_name if title_name else name} = {format_number(x,udigits=digits)}, rel_error = {x.s/x.n*100:.{digits}g}%>")))

    table(
        [key_values, rel_error, values],
        [color.fg.bold(x) for x in ["value", r"rel. error in \%", f"propagated error"]],
        index_column=[color.fg.bold("variable"), *key_str],
        udigits=digits,
        show=show,
    )

    if add_file:
        table(
            [key_str, key_values, rel_error, values],
            ["Variable", "Wert", r"Rel. Fehler in \%", f"Propagierter Fehler"],
            caption=rf"Fehlerquellen für ${title_name if title_name else name}=${format_number(x,udigits=3, latex=True)}, rel. Fehler $={x.s/x.n*100:.{digits}g}\%$.",
            udigits=digits,
            show=False,
            show_latex=show_latex,
            filename=name + ".tex" if add_file else None,
        )

def residual_plot(x, y, f, fit, logx=False, logy=False, normalize=False, xlabel="x", ylabel="y", filename=None, suptitle=None, confidence_interval=None, xlim=None, bfsuptitle=None):
    def __confidence_interval__(x, f, fit, axis, sigma, alpha=0.3):
        y = [f(x,*fit) for x in x]
        axis.fill_between(x, ev(y)-sigma*std(y), ev(y)+sigma*std(y), alpha=alpha, fc="lightgreen", ec="green")
        
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.errorbar(ev(x), ev(y), xerr=std(x), yerr=std(y), c="r", fmt="o", markersize=2, capsize=2,)
    
    if logx is False:
        diff = max(ev(x)) - min(ev(x))
        low, high = min(ev(x)) - 0.1*diff, max(ev(x)) + 0.1*diff
        if xlim is not None: low, high= xlim
        x_fit = np.linspace(low,high, 1000)
    else: 
        low, high = min(ev(x))/2, max(ev(x))*2
        if xlim is not None: low,high=xlim
        x_fit = np.geomspace(low,high, 1000)
    
    ax1.set_xlim(low,high)
    ax2.set_xlim(low,high)
    
    ax1.plot(x_fit, [f(x,*ev(fit)) for x in x_fit], c="g")
    
    if confidence_interval is not None:
        if isinstance(confidence_interval, Iterable):
            for i in range(len(confidence_interval)):
                __confidence_interval__(x_fit,f,fit,ax1,confidence_interval[i],0.3/(1+i))
        else:
            __confidence_interval__(x_fit,f,fit,ax1,float(confidence_interval),0.3)
            
            

    ax2.axhline(y=0, c="black", linestyle="--")
    
    residuals = np.subtract(y,[f(x,*ev(fit)) for x in x])
    chiq = np.sum((ev(residuals) / std(residuals)) ** 2)
    p = stats.chi2.sf(chiq, len(x) - 2)
    residuals = residuals / (std(residuals) if normalize else 1)
    
    ax2.errorbar(ev(x), ev(residuals), yerr=std(residuals), fmt="o", c="r", markersize=3, capsize=2)

    sup = r"$\mathbf{"+bfsuptitle.replace(" ", r"\ ")+"}$\n" if bfsuptitle else suptitle + "\n" if suptitle else ""
    
    title = (sup if sup else "") + f"{', '.join([abc+'='+fn(x,latex=True) for abc,x in zip('abcdefghijk',fit)])}\nχ²/dof = {int(chiq) if np.isfinite(chiq) else str(chiq)}/{len(x)-len(fit)} = {format_number(chiq/(len(x)-len(fit)),fdigits=3,latex=True)}, p={format_number(p,fdigits=3,latex=True)}"
    ax1.set_title(title)
    
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_ylabel(f"Residuum{r' in sigma' if normalize else ''}")
    
    if logx:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    if logy:
        ax1.set_yscale("log")

    plt.tight_layout()

    if filename:
        fig.savefig(plots_path + filename)
    plt.show()

def nonlinfit(f,x,y,p0,maxit=100,report=False):
    data = odr.RealData(ev(x), ev(y), 
                        sx = std(x) if np.any(std(x)!=0) else None,
                        sy = std(y) if np.any(std(y)!=0) else None,)
    model = odr.Model(lambda beta, x: f(x, *beta),)
    odr_instance = odr.ODR(data, model, beta0=ev(p0),maxit=maxit,)
    output = odr_instance.run()
    if report: output.pprint()

    return u(output.beta, output.sd_beta)

def linreg(
    X: List[float],
    Y: List[UFloat],
    xlabel="x",
    ylabel="y",
    filename=None,
    suptitle=None,
    bfsuptitle=None,
    show=True,
    logx=False,
    logy=False,
    normalize=False,
    confidence_interval=False,
    xlim=None
) -> Tuple[UFloat, UFloat]:
    """
    Fits y = ax + b, in the case that only y has uncertainies. Also creates a plot of the fit and residuals, that includes the chi^2-value and the easier to interpret p-value.
    """
    X, Y = np.asarray(ev(X)), np.asarray(Y)
    
    a, ea, b, eb, _, _ = lineare_regression(X, ev(Y), std(Y))
    a, b = ufloat(a, ea), ufloat(b, eb)
        
    if show: residual_plot(X, Y, lambda x,a,b: a*x+b, [a,b], logx=logx, logy=logy,
                           normalize=normalize, xlabel=xlabel, ylabel=ylabel, filename=filename, 
                           suptitle=suptitle, confidence_interval=confidence_interval, xlim=xlim,
                           bfsuptitle=bfsuptitle)

    return a, b


def linreg2(X, Y, xlabel="x", ylabel="y", filename=None, suptitle=None, show=True, confidence_interval=None, logx=False, logy=False, normalize=False, bfsuptitle=None, xlim=None, report=False):
    """
    Fits y = ax + b, in the case that both variables have uncertainies. Also creates a plot of the fit and residuals, that includes the chi^2-value and the easier to interpret p-value.
    """    
    X, Y = np.asarray(X), np.asarray(Y)
    
    a,b = linreg(ev(X), Y, show=False)
    xerr, yerr = std(X), std(Y)
    x, y = ev(X), ev(Y)
    

    def linear_model(params, x):
        a = params["a"]
        b = params["b"]
        return a * x + b

    params = lmfit.Parameters()
    params.add("a", value=ev(a))
    params.add("b", value=ev(b))

    def residual(params):
        a = params["a"].value
        model = linear_model(params, x)
        total_error = np.sqrt(yerr**2 + (a * xerr) ** 2)
        return (y - model) / total_error

    minimizer = lmfit.Minimizer(residual, params, nan_policy="omit", scale_covar=False)
    result = minimizer.minimize(method="leastsq")
    
    if report: lmfit.report_fit(result)

    if not result.success:
        raise RuntimeError(f"Fit failed: {result.message}")

    a = result.params["a"].value
    b = result.params["b"].value
    cov = result.covar

    a, b = un.correlated_values((a, b), cov)
    
    if show: residual_plot(X, Y, lambda x,a,b: a*x+b, [a,b], logx=logx, logy=logy,
                           normalize=normalize, xlabel=xlabel, ylabel=ylabel, filename=filename, 
                           suptitle=suptitle, confidence_interval=confidence_interval, xlim=xlim,
                           bfsuptitle=bfsuptitle,)
    return a, b


__gaussian__ = lambda x, mu, sigma, A: A * unp.exp(-((x - mu) ** 2) / (2 * sigma**2))

@np.vectorize
def N_gaussian(x, *fit):
    assert len(fit) % 3 == 0, f"fit has the wrong amount of parameters ({len(fit)})"
    N = int(len(fit) / 3)
    return np.sum([__gaussian__(x, *fit[3 * i : 3 * (i + 1)]) for i in range(N)])

def fit_N_gaussian(
    x,
    y,
    p0,
    label="",
    maxfev=2000,
    diff=80,
    mask=None,
    plot=False,
    scale_sigma=1,
    **kwargs,
):
    mu = p0[: -1 if len(p0) == 3 else -3 : 3]
    low, high = int(max(0, min(mu) - diff)), int(min(max(mu) + diff, len(x - 1)))
    mask = mask if mask else np.s_[low:high]

    fit, cov = curve_fit(N_gaussian, x[mask], y[mask], p0, maxfev=maxfev)

    label = "_" + label if label else ""
    labels = np.array(
        [
            [f"mu_{i}{label}", f"sigma_{i}{label}", f"A_{i}{label}"]
            for i in range(1, int(len(p0) / 3) + 1)
        ]
    ).flatten()

    err = [scale_sigma * sqrt(cov[i, i]) for i in range(len(cov))]
    if np.all(np.isfinite(err)):
        fit = u(fit, err, labels)
    else:
        fit = u(fit, 0, labels)

    if plot:
        plot_N_gaussian_fit(x, y, fit, **kwargs)

    return fit

def __general_func__(x, *args):
    return (
        N_gaussian(x, *args)
        if len(args) % 3 == 0
        else (
            N_gaussian(x, *(args[:-1])) + args[-1]
            if len(args) % 3 == 1
            else N_gaussian(x, *(args[:-2])) + x * args[-2] + args[-1]
        )
    )

@np.vectorize
def general_func(x, fit):
    """
    N Gauss curves with either a gaussian or constant background
    fit: {"gauss": [μ_1, σ_1, A_1, ...], "bg": [μ, σ, A] | const | ax + b}
    """
    args = fit["gauss"]
    if "bg" in fit.keys():
        args = np.append(args, fit["bg"]).flatten()
    return __general_func__(x, *args)


