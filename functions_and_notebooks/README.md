This folder contains functions and notebooks:

"Failure_functions.py" and "Forecasting_functions.py" are Python files that contain the functions used for the MCMC inversion of the seismicity rate model parameters. The notebook "seismicity_rate_model_example.ipynb" provides an example of how to run the seismicity rate model. "Failure_functions.py" and "Forecasting_functions.py" are called by this notebook (read the Midland_Basin_HF_Seismicity/README.md file for additional information on the Python version and libraries required to run "seismicity_rate_model_example.ipynb").

"functions_cheng_undrained.py" and "functions_cheng_drained.py" are Python files that contain the functions used to calculate stresses and pore-pressure changes. The notebook "Stress_Pp_model_example.ipynb" provides an example of how to calculate stresses and pore-pressure changes for cluster C1 (Figure 8 in the manuscript). Read the Midland_Basin_HF_Seismicity/README.md file for additional information on the Python version and libraries required to run "Stress_Pp_model_example.ipynb". The notebook "Stress_Pp_model_example.ipynb" outputs a file called "cluster_0_example.pkl", which is then used as input for the seismicity rate model.

"plot_forecast_results.ipynb" shows how to reproduce the analyses presented in the "Results" and "Discussion" sections (Figures 8, 9, 10, 11, and 12).

"figure13.ipynb" shows how to reproduce the analysis presented in Figure 13.
