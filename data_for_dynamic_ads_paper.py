import adsorption_solver
import multiprocessing
import os
import numpy as np

'''
Regenerates the numerical solutions used to make the figures in the paper on ionic
surfactant dynamic adsorption. Each run uses the exact settings stored in the original
data files (read from their .params attribute). The runs were made in three batches
with different settings, so the settings are listed per run below.

Output files are named eps_{eps}_Da_{Da}_Dn_{Dn}.nso to match the plotting script.
The number types matter for the file names: Da=1 (int) gives 'Da_1', Da=10.0 (float)
gives 'Da_10.0'.

Memory: the solver keeps every time step in memory until it saves, so each run needs
about as much RAM as its output file (~15 GB for each Da=100 run, ~10.5 GB for Da=1e5).
Be careful running those in parallel.
'''

out_folder = 'data'                     # where the .nso files are saved
run_in_parallel = False                 # True runs several at once (see memory note above)
n_parallel = 3                          # how many at once, if run_in_parallel is True

data_file_name = 'eps_{}_Da_{}_Dn_{}'

# meshes (shifted to start at x = 0)
x_500 = np.logspace(-8, 2, 500)
x_500 -= x_500[0]
x_2000 = np.logspace(-8, 3, 2000)
x_2000 -= x_2000[0]

# (eps, Da, Dn, mesh, t_f, dt_0, stepfrac, auto_terminate, figures)
runs = [
    # batch 1 (Mar 2025): eps = 1e-2
    (0.01,   1,        1,  x_500,  1e4, 1e-6,  0.9, True,  'Fig. 4a'),
    # batch 2 (Apr 2025): eps = 1e-4 Da sweep, run to equilibrium
    (0.0001, 0.1,      1,  x_500,  1e5, 1e-6,  0.5, True,  'Figs. 5b, 7'),
    (0.0001, 1,        1,  x_500,  1e5, 1e-12, 0.9, True,  'Figs. 4b, 5a, 7'),
    (0.0001, 10.0,     1,  x_500,  1e5, 1e-12, 0.9, True,  'Fig. 7'),
    (0.0001, 1000.0,   1,  x_500,  1e5, 1e-12, 0.9, True,  'Fig. 7'),
    (0.0001, 10000.0,  1,  x_500,  1e5, 1e-12, 0.9, True,  'Fig. 7'),
    (0.0001, 100000.0, 1,  x_500,  1e5, 1e-12, 0.9, True,  'Fig. 7'),
    # batch 3 (May 2025): Da = 100, longer and finer mesh, fixed end time (bulk profiles)
    (0.0001, 100.0,    1,  x_2000, 1e4, 1e-8,  0.9, False, 'Figs. 6a, 7, 8'),
    (0.0001, 100.0,    10, x_2000, 1e4, 1e-8,  0.9, False, 'Fig. 6b'),
]

notes_template = '''
Generated for the paper on ionic surfactant dynamic adsorption ({figures}). Langmuir
kinetics, non-adsorbing anion (kan=0), beta=1. Mesh: N={N} log-spaced nodes from 0 to
x={xmax:g}. Initial step dt_0={dt_0:g}, dynamic time-stepping with stepfrac={stepfrac}.
{ending} Exact parameters are in the object attribute (.params).
'''


def do_solve(eps_val, da_val, dn_val, x, t_f, dt_0, stepfrac, auto_terminate, figures):
    datafname = os.path.join(out_folder, data_file_name.format(str(eps_val), str(da_val), str(dn_val)))
    if auto_terminate:
        ending = f'Runs until Gamma stops changing (auto_terminate) or t={t_f:g}.'
    else:
        ending = f'Runs to t={t_f:g}.'
    notes = notes_template.format(figures=figures, N=len(x), xmax=x[-1], dt_0=dt_0,
                                  stepfrac=stepfrac, ending=ending)
    adsorption_solver.solve(x, t_f, dt_0=dt_0, eps=eps_val, Da=da_val, Dn=dn_val,
                            stepfrac=stepfrac, auto_terminate=auto_terminate,
                            filename=datafname, notes=notes, save=True)
    # nothing is returned: the run is saved to file, and returning the object from a
    # parallel worker would copy many GB back to the main process
    return None


if __name__ == '__main__':
    os.makedirs(out_folder, exist_ok=True)
    if run_in_parallel:
        with multiprocessing.Pool(n_parallel) as pool:
            pool.starmap(do_solve, runs)
    else:
        for run in runs:
            do_solve(*run)