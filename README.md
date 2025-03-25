### This is my work for my master thesis in Maastricht University.
Two different formulation for an MPC controller are given one in Python with [SCiPy] (https://scipy.org/) and one in C++ with [Ariadne] (https://www.ariadne-cps.org/).

- Motivation
This work investigates the validity of general solvers, like SCiPy provides, under floating point rounding errors.
Ariadne provides error bounds for the solution, we expect the SCiPy solution to be inside these error bounds.
