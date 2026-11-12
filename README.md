Code for paper on dynamic interfacial mechanics of transiently adsorbing ionic surfactants.
[Uhlig & Khair (2026)](https://doi.org/10.1007/s10665-025-10492-1).

To reproduce all the data used in the paper, run [`data_for_dynamic_ads_paper.py`](data_for_dynamic_ads_paper.py).

[`adsorption_solver.py`](adsorption_solver.py) is the solver we used to solve the 1D Poisson–Nernst–Plank equations, where we 
used SciPy's `solve_bvp` to time step and treat the 1D PDE as a series of boundary value problems.

[`dynamic_adsorption_figs.py`](dynamic_adsorption_figs.py) creates all the figures in the paper in the style which they are shown. 
Note that the style includes text rendered in LaTeX (and hence requires an installation of LaTeX). 

[`numerical_solution_object.py`](numerical_solution_object.py) stores all of the data output from [`adsorption_solver.py`](adsorption_solver.py), as .nso files 
using pickle. These files can become quite large, so be aware of that. 
