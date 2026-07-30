# Midland_Basin_HF_Seismicity
This repository contains the data and codes used in the paper "Earthquakes induced by hydraulic fracturing, a semi-analytical model with application to the Permian Basin". 

##############

The code provided in the "functions_and_notebooks" folder is written in Python. In "functions_and_notebooks" you will find two notebooks named:

1 - "Stress_Pp_model_example.ipynb"
2 - "seismicity_rate_model_example.ipynb"

"Stress_Pp_model_example.ipynb" provides an example of how to run the stress and pore-pressure model. To run this notebook, you will need to install Python v = 3.12.2 and the libraries listed in "requirement_stress_model.txt". "Stress_Pp_model_example.ipynb" calls functions (so make sure you run the notebook from the correct directory) defined in the two files "functions_cheng_drained.py" and "functions_cheng_undrained.py".

"seismicity_rate_model_example.ipynb" provides an example of how to run the seismicity rate model. To run this notebook, you will need to install Python v = 3.9.25 and the libraries listed in "requirement_seismicity_model.txt". "seismicity_rate_model_example.ipynb" calls functions defined in the two files "Failure_functions.py" and "Forecasting_functions.py".

The "functions_and_notebooks" folder contains additional files. Please read the README.md file in the folder for additional information.

###############

The "processed_data" folder contains different datasets: cluster information, forecast results, and other analyses. Please read the README.md file in the folder for additional information.
