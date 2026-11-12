import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker
from scipy.special import gamma as gamma_function
from scipy.special import polygamma, hyp1f1
from scipy.special import psi as digamma
import pickle
import sys, os
sys.path.insert(1, '/Users/emersonuhlig/Library/Mobile Documents/com~apple~CloudDocs/Documents/school/graduate/research/python files/charged specie adsorption dynamics')
from numerical_solution_object import load

rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.latex.preamble'] = r"""
\usepackage{amsfonts, amsmath, amssymb, amsthm, bm, cancel, upgreek}
\renewcommand{\d}{\partial} % dee for partial differential operator
\newcommand{\eps}{\varepsilon} % the superior epsilon
\newcommand{\mb}[1]{\mathbf{#1}} % math bold (different than bm)
\newcommand{\til}[1]{\widetilde{#1}} % wide tilde
\newcommand{\wh}[1]{\widehat{#1}} % wide hat
\newcommand{\nt}[1]{\emph{\texttt{#1}}} % new term 
\newcommand{\der}{\text{d}} % differential operator
\newcommand{\ord}[1]{\mathcal{O}\left( #1 \right)}
"""
rcParams['font.size'] = 16
rcParams['legend.fontsize'] = 12
rcParams['legend.frameon'] = False
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.dpi'] = 1200
rcParams['figure.constrained_layout.use']
rcParams['figure.subplot.wspace'] = 0.5  # Horizontal space between subplots
rcParams['figure.subplot.hspace'] = 0.4  # Vertical space between subplots

figname = 'fig{fignum:1d}.png'
data_file_name = 'data/eps_{}_Da_{}_Dn_{}.nso'


def e_by_e0(s):
    prefactor = 1/(1 + (1/s)**2)
    re_part = 1
    im_part = (1/s)
    out = prefactor * (re_part + 1j*im_part)
    return out

def isotherm_pi(t, obj):
    Gamma = obj.gamma_p(t)
    Zeta = np.array([obj.potential(0, ti) for ti in t])
    out = np.log(1 - Gamma) - Gamma * Zeta
    return out

def c_short_time(x, t, eps, Da, alpha, upbeta=1):
    z = -alpha * x**2 / (4 * t)
    
    # Gamma function values
    G_1_3 = gamma_function(1/3)
    G_5_6 = gamma_function(5/6)
    
    # Confluent hypergeometric functions
    H1 = hyp1f1(1/6, 1/2, z)
    H2 = hyp1f1(2/3, 3/2, z)
    
    # Combine terms
    prefactor = (2 * upbeta)**(2/3) * Da * G_1_3 / (3 * upbeta * Da)**(2/3)
    term1 = t**(-1/6) * H1 / G_5_6
    term2 = np.sqrt(alpha * x**2 / t**(4/3)) * H2 / G_1_3
    lo_term = eps**(2/3) * prefactor * (term1 - term2)
    out = 1 - lo_term
    return out

def c(x, t, eps, Da, alpha, upbeta=1):
    Y_diff = digamma(4/3) - digamma(1/3)
    prefactor = (4/upbeta)**(1/3) * (Y_diff * np.sqrt(alpha) / 3)
    exponential = np.exp(-alpha * x**2 / (4 * t)) / np.sqrt(np.pi * t)
    lo_term = prefactor * exponential
    out = 1 - eps**(2/3) * lo_term
    return out

epsilons = [1e-2, 1e-3, 1e-4]
Damkohlers = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
negative_diff_coefs = [1, 2, 5, 10, 0.5, 0.1, 0.01]

fdim = (4,4) # width, height
fdim2 = (4,4)
options = {'figsize': fdim}
option2 = {'figsize': fdim2}

def make_fig4(show=False):
    plt.figure(4)
    fig, (ax1, ax2) = plt.subplots(1, 2, num=1, **options)
    t = np.logspace(-6, 3, num=1000)
    t_data = np.logspace(-6, 3, 25)
    df = data_file_name.format(str(1e-2), str(1), str(1))
    obj = load(df)

    Da = 1
    eps = 1e-2
    geq = 4**(1/3) * eps**(2/3)

    ax1.semilogx(t, (1 - np.exp(-3 * Da * t))**(1/3), '-k', label='asymptotics')
    ax1.semilogx(t_data, obj.gamma_p(t_data)/geq, 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')

    df = data_file_name.format(str(1e-4), str(1), str(1))
    obj = load(df)
    eps = 1e-4
    geq = 4**(1/3) * eps**(2/3)

    ax2.semilogx(t, (1 - np.exp(-3 * Da * t))**(1/3), '-k', label='asymptotics')
    ax2.semilogx(t_data, obj.gamma_p(t_data)/geq, 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')

    ax1.set_ylabel(r'$\Gamma / \Gamma_\text{eq}$', rotation=90, labelpad=5)
    ax1.set_xlabel(r'$t$'+ '\n(a)')
    ax2.set_ylabel(r'$\Gamma/\Gamma_\text{eq}$', rotation=90, labelpad=5)
    ax2.set_xlabel(r'$t$'+ '\n(b)')
    ax2.set_xlim([1e-6, 1e3])
    ax2.set_ylim([-0.03, 1.03])
    ax1.set_xlim([1e-6, 1e3])
    ax1.set_ylim([-0.03, 1.03])

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    label1 = r'$\varepsilon=10^{-2}$'
    label2 = r'$\varepsilon=10^{-4}$'
    label1xy = (10**-3, 0.75)
    textxy1 = (10**-3, 0.75)
    text = ax1.annotate(label1, label1xy, xytext=textxy1, ha='center')
    text = ax2.annotate(label2, label1xy, xytext=textxy1, ha='center')
    text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.0))

    nticks = 9
    maj_loc = ticker.LogLocator(numticks=nticks)
    min_loc = ticker.LogLocator(subs='all', numticks=9)
    ax1.xaxis.set_major_locator(maj_loc)
    ax1.xaxis.set_minor_locator(min_loc)
    ax2.xaxis.set_major_locator(maj_loc)
    ax2.xaxis.set_minor_locator(min_loc)

    for i, labels in enumerate(zip(ax1.get_xticklabels(), ax2.get_xticklabels())): 
        label1, label2 = labels
        label1.set_visible(False)
        label2.set_visible(False)
        if (i - 1) % 3 == 0:
            label1.set_visible(True)
            label2.set_visible(True)


    if show: plt.show()
    else: plt.savefig(figname.format(fignum=4))
    return None

def make_fig5(show=False):
    plt.figure(5)
    fig, (ax1, ax2) = plt.subplots(1, 2, num=3, **options)
    t = np.logspace(-6, 3, num=1000)
    t_data = np.logspace(-6, 3, 35)
    eps = 1e-4
    geq = 4**(1/3) * eps**(2/3)
    Da = 1
    df = data_file_name.format(str(1e-4), str(Da), str(1))
    obj = load(df)
    ax1.loglog(t_data, obj.gamma_p(t_data)/geq, 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')
    ax1.loglog(t, (1 - np.exp(-3 * Da * t))**(1/3), '-k', label=r'$t = \ord{1}$')
    ax1.loglog(t, 2 * eps * np.log(1 + Da * t / (2 * eps) )/geq, '--k', label=r'$t = \ord{\eps}$')

    ax1.set_ylabel(r'$\Gamma / \Gamma_\text{eq}$', rotation=90, labelpad=5)
    ax1.set_xlabel(r'$t$' + '\n(a)')
    ax1.set_xlim([1e-6, 1e3])
    ax1.set_ylim([1e-4, 2])
    ax1.set_box_aspect(1)

    Da = 0.1
    df = data_file_name.format(str(1e-4), str(Da), str(1))
    obj = load(df)

    ax2.loglog(t_data, obj.gamma_p(t_data)/geq, 'ok', label='numerical', markerfacecolor="None", markeredgecolor='black')
    ax2.loglog(t, (1 - np.exp(-3 * Da * t))**(1/3), '-k', label=r'$t = \ord{1}$')
    ax2.loglog(t, 2 * eps * np.log(1 + Da * t /(2 * eps))/geq, '--k', label=r'$t = \ord{\eps}$')

    ax2.set_ylabel(r'$\Gamma / \Gamma_\text{eq}$', rotation=90, labelpad=5)
    ax2.set_xlabel(r'$t$'+ '\n(b)')
    ax2.set_xlim([1e-6, 1e3])
    ax2.set_ylim([1e-4, 2])
    ax2.set_box_aspect(1)
        
    nticks = 9
    maj_loc = ticker.LogLocator(numticks=nticks)
    min_loc = ticker.LogLocator(subs='all', numticks=9)
    ax1.xaxis.set_major_locator(maj_loc)
    ax1.xaxis.set_minor_locator(min_loc)
    ax2.xaxis.set_major_locator(maj_loc)
    ax2.xaxis.set_minor_locator(min_loc)

    for i, labels in enumerate(zip(ax1.get_xticklabels(), ax2.get_xticklabels())): 
        label1, label2 = labels
        label1.set_visible(False)
        label2.set_visible(False)
        if (i - 1) % 3 == 0:
            label1.set_visible(True)
            label2.set_visible(True)

    ax2.legend(bbox_to_anchor=(1,1))


    if show: plt.show()
    else: plt.savefig(figname.format(fignum=5))
    return None

def make_fig6(show=False):
    plt.figure(6)
    fig, (ax1, ax2) = plt.subplots(1, 2, num=6, **options)
    x = np.linspace(0, 5, 500)
    eps = 1e-4
    Da = 1e2
    Dn1 = 1
    Dn2 = 10
    alpha1 = (1 + Dn1)/ (2 * Dn1)
    alpha2 = (1 + Dn2)/ (2 * Dn2)
    df1 = data_file_name.format(str(eps), str(Da), str(Dn1))
    df2 = data_file_name.format(str(eps), str(Da), str(Dn2))
    obj1 = load(df1)
    obj2 = load(df2)
    t = np.logspace(-3, 0, 5)
    for ti in t:
        mantissa, exponent = f'{ti:1.1e}'.split('e')
        fixed_exponent = f'{exponent[0]}' + f'{exponent[-1]}'
        if fixed_exponent == '+0' or fixed_exponent == '-0':
            formatted_time = r'$t={}$'.format(mantissa)
        elif '+' in fixed_exponent:
            formatted_time = r'$t={} \times 10^{{{}}}$'.format(mantissa, fixed_exponent[-1])
        else:
            formatted_time = r'$t={} \times 10^{{{}}}$'.format(mantissa, fixed_exponent)
        ax1.plot(x, obj1.c_p(x, ti)-1, '.', label=formatted_time)
        ax1.plot(x, c(x, ti, eps, Da, alpha1)-1, '-k')
        ax2.plot(x, obj2.c_p(x, ti)-1, '.', label=formatted_time)
        if ti == t[-1]:
            ax2.plot(x, c(x, ti, eps, Da, alpha2)-1, '-k', label='asymptotics')
        else:
            ax2.plot(x, c(x, ti, eps, Da, alpha2)-1, '-k')

    ax1.set_ylabel(r'$c_+(x,t)-1$', rotation=90, labelpad=5)
    ax1.set_xlabel(r'$x$'+ '\n(a)')
    ax2.set_ylabel(r'$c_+(x,t)-1$', rotation=90, labelpad=5)
    ax2.set_xlabel(r'$x$'+ '\n(b)')
    ax1.set_ylim([-0.005, 0.0001])
    ax2.set_ylim([-0.005, 0.0001])
    ax1.set_xlim([0, 5])
    ax2.set_xlim([0, 5])
    ax2.legend(bbox_to_anchor=(0.95,1))
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    yticks = np.array([-40, -30, -20, -10, 1]) * eps  
    ax1.set_yticks(yticks)
    ax2.set_yticks(yticks)
    tlabs = [fr"${int(y/eps)}\varepsilon$" for y in yticks]
    tlabs[-1] = r'$\varepsilon$'
    ax1.set_yticklabels(tlabs)
    ax2.set_yticklabels(tlabs) 

    label1 = r'$\alpha=1$'
    label2 = r'$\alpha=\textstyle{\frac{11}{20}}$'
    label1xy = (3, -0.0025)
    textxy1 = (3, -0.0025)
    text = ax1.annotate(label1, label1xy, xytext=textxy1, ha='center')
    text = ax2.annotate(label2, label1xy, xytext=textxy1, ha='center')
    text.set_bbox(dict(facecolor='white', edgecolor='white', alpha=0.0))

    if show: plt.show()
    else: plt.savefig(figname.format(fignum=6))
    return None

def make_fig7(show=False):
    plt.figure(7)
    fig, (ax1, ax2) = plt.subplots(1, 2, num=2, **options)
    t = np.logspace(-6, 3, num=1000)
    t_data = np.logspace(-12, 4, 1000)

    eps=1e-4
    geq = 4**(1/3) * eps**(2/3)
    dalist = [1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
    labellist = [r'$Da = 10^{-1}$', 
                 r'$Da = 1$',
                 r'$Da = 10$',
                 r'$Da = 10^{2}$',
                 r'$Da = 10^{3}$',
                 r'$Da = 10^{4}$',
                 r'$Da = 10^{5}$',
                 r'$Da = 10^{6}$'
                 ]
    for i, Da in enumerate(dalist):
        df = data_file_name.format(str(1e-4), str(Da), str(1))
        obj = load(df)
        ax1.semilogx(t_data, obj.gamma_p(t_data)/geq, label=labellist[i])
        ax2.semilogx(Da*t_data, obj.gamma_p(t_data)/geq, label=labellist[i])

    ax2.semilogx(t, (1 - np.exp(-3*t))**(1/3), ':k', label='asymptotics')

    ax1.set_ylabel(r'$\Gamma / \Gamma_\text{eq}$', rotation=90, labelpad=5)
    ax1.set_xlabel(r'$t$'+ '\n(a)')
    ax2.set_ylabel(r'$\Gamma/\Gamma_\text{eq}$', rotation=90, labelpad=5)
    ax2.set_xlabel(r'$t/\uptau_\text{ads}$'+ '\n(b)')

    ax2.set_xlim([1e-6, 1e3])
    ax2.set_ylim([0, 1.01])
    ax1.set_xlim([1e-6, 1e3])
    ax1.set_ylim([0, 1.01])

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)

    nticks = 9
    maj_loc = ticker.LogLocator(numticks=nticks)
    min_loc = ticker.LogLocator(subs='all', numticks=9)
    ax1.xaxis.set_major_locator(maj_loc)
    ax1.xaxis.set_minor_locator(min_loc)
    ax2.xaxis.set_major_locator(maj_loc)
    ax2.xaxis.set_minor_locator(min_loc)

    for i, labels in enumerate(zip(ax1.get_xticklabels(), ax2.get_xticklabels())): 
        label1, label2 = labels
        label1.set_visible(False)
        label2.set_visible(False)
        if (i - 1) % 3 == 0:
            label1.set_visible(True)
            label2.set_visible(True)

    ax2.legend(bbox_to_anchor=(1.05,1))

    if show: plt.show()
    else: plt.savefig(figname.format(fignum=7))
    return None

def make_fig8(show=False):
    plt.figure(8)
    fig, ax1 = plt.subplots(1, 1, num=5, figsize=(2,2))
    t = np.logspace(-6, 3, num=1000)
    t_data = np.logspace(-6, 3, num=25)
    Da = 1e2
    eps = 1e-4
    df = data_file_name.format(str(eps), str(Da), str(1))
    obj = load(df)
    pi_eq = (4)**(1/3) * eps**(2/3)
    ax1.semilogx(t, -(1 - np.exp(-3 * Da * t))**(1/3), '-k', label=r'asymptotic')
    ax1.semilogx(t_data, isotherm_pi(t_data, obj)/pi_eq, 'ok', markerfacecolor="None", markeredgecolor='black', label='numerics')
    ax1.set_ylabel(r'$-\Pi^*/\Pi_\text{eq}^*$', rotation=90, labelpad=5)
    ax1.set_xlabel(r'$t$')
    ax1.set_xlim([1e-6, 1e3])
    ax1.set_ylim([-1.01, 0])
    ax1.set_box_aspect(1)
    nticks = 9
    maj_loc = ticker.LogLocator(numticks=nticks)
    min_loc = ticker.LogLocator(subs='all', numticks=9)
    ax1.xaxis.set_major_locator(maj_loc)
    ax1.xaxis.set_minor_locator(min_loc)

    for i, labels in enumerate(ax1.get_xticklabels()): 
        label1 = labels
        label1.set_visible(False)
        if (i-1) % 3 == 0:
            label1.set_visible(True)
    ax1.legend()


    if show: plt.show()
    else: plt.savefig(figname.format(fignum=8))
    return None

def make_fig9(show=False):
    plt.figure(9)
    fig, ax1 = plt.subplots(1, 1, num=4, **options)
    w = np.logspace(-3, 3, num=1000)
    ax1.semilogx(w, np.real(e_by_e0(w)), '-k', label=r"$E'(\omega) = \text{ Re}\left( \frac{E}{E_0} \right)$")
    ax1.semilogx(w, np.imag(e_by_e0(w)), ':k', label=r"$E''(\omega) = \text{ Im}\left( \frac{E}{E_0} \right)$")
    ax1.set_ylabel(r'$E / E_0$', rotation=90, labelpad=5)
    ax1.set_xlabel(r'$\omega / \omega_\text{ads}$')
    ax1.set_xlim([1e-3, 1e3])
    ax1.set_ylim([0, 1])
    ax1.set_box_aspect(1)
    ax1.legend()


    if show: plt.show()
    else: plt.savefig(figname.format(fignum=9))
    return None


make_fig4()
make_fig5()
make_fig6() 
make_fig7()
make_fig8()
make_fig9()