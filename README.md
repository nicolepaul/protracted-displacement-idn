# Protracted displacement after disasters

This repository includes code to estimate direct causal effects of physical, socioeconomic, and psychological factors on displacement duration outcomes using household survey data following the 2018 Central Sulawesi earthquake, tsunami, and liquefaction disaster in Indonesia. The causal effects are estimated using debiased/double machine learning (DML). This is companion code for a manuscript undergoing peer review:

    Paul, Nicole, Eyitayo Opabola, Sukiman Nurdin, Dicky Pelupessy, Aulia Damayanti, Reval Rahmat Nurdin, Shafitri Rayhana, Adam, Sifa Salsabila Sahempa, and Carmine Galasso. “Reducing protracted displacement: Evidence for disaster risk reduction policy and practice.” In review. 

For information and materials related to the housheold survey itself, including the anonymized household-level responses and survey instrument, please refer to the following:

    Paul, Nicole, Eyitayo Opabola, Sukiman Nurdin, Dicky Pelupessy, Aulia Damayanti, Reval Rahmat Nurdin, Shafitri Rayhana, Adam, Sifa Salsabila Sahempa, and Carmine Galasso. “Disaster displacement in context: Household trajectories after the 2018 Central Sulawesi earthquake.” In review.

    Paul, Nicole, Shafitri Adam, Aulia Damayanti, Reval Rahmat Nurdin, Shafitri Rayhana, Adam, Sifa Salsabila Sahempa, Dicky Pelupessy, Sukiman Nurdin, Eyitayo Opabola, and Carmine Galasso. 2025. “Data: Household Relocation after the 2018 Central Sulawesi Earthquake and Tsunami.” RIN Dataverse, August 20. http://data.brin.go.id/dataset.xhtml?persistentId=hdl:20.500.12690/RIN/XXT3TT.


## Running the notebooks

The following notebooks are provided to replicate the analysis:

 * [**Effect_DML.ipynb**](/Effect_DML.ipynb): This is the primary notebook to run the DML analysis to estimate the average treatment effects 
 * [**Sensitivity.ipynb**](/Sensitivity.ipynb): This notebook reads the pre-calculated data in order to tabulate the E-values
 * [**Power.ipynb**](/Power.ipynb): This notebook performs a simulated power analysis to assess the required sample size given an effect size

Several common python libraries are used in the notebooks, in addition to custom scripts. It is recommended to run these notebooks using a [virtual environment](https://docs.python.org/3/library/venv.html). Once you have a virtual environment activated, you can install all dependencies with `pip install -r requirements.txt`.

## Other materials

For figures repesenting the causal identification for each considered treatment, see the **dag** subfolder.

For figures representing the estimated average treatment effect effects, please see the **img** subfolder.

For the data calculating during analysis, please see **data/results.sav**.