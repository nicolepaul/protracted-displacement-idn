# Protracted displacement after disasters

This repository includes code to estimate the effects of physical, socioeconomic, and psychological factors on displacement duration outcomes using household survey data following the 2018 Central Sulawesi earthquake, tsunami, and liquefaction event in Indonesia. The effects are estimated using debiased/double machine learning (DML). This is companion code for a manuscript undergoing peer review:

* Paul, Nicole, Eyitayo Opabola, Sukiman Nurdin, Dicky Pelupessy, Aulia Damayanti, Reval Rahmat Nurdin, Shafitri Rayhana, Adam, Sifa Salsabila Sahempa, and Carmine Galasso. “Reducing protracted displacement: Evidence for disaster risk reduction policy and practice.” In review. 

For information and materials related to the household survey itself, including the anonymized household-level responses and survey instrument, please refer to the following:

* Paul, Nicole, Eyitayo Opabola, Sukiman Nurdin, Dicky Pelupessy, Aulia Damayanti, Reval Rahmat Nurdin, Shafitri Rayhana, Adam, Sifa Salsabila Sahempa, and Carmine Galasso. “Disaster displacement in context: Household trajectories after the 2018 Central Sulawesi multi-hazard event.” In press.

* Paul, Nicole, Shafitri Adam, Aulia Damayanti, Reval Rahmat Nurdin, Shafitri Rayhana, Adam, Sifa Salsabila Sahempa, Dicky Pelupessy, Sukiman Nurdin, Eyitayo Opabola, and Carmine Galasso. 2025. “Data: Household Relocation after the 2018 Central Sulawesi Earthquake and Tsunami.” RIN Dataverse, August 20. http://data.brin.go.id/dataset.xhtml?persistentId=hdl:20.500.12690/RIN/XXT3TT.

## Causal identification and assumptions

To replicate the DAG in DAGgity, please refer to the **dag/replication.dag** file.

For figures repesenting the causal identification for each considered treatment, see the **dag** subfolder.

## Effect estimation, sensitivity analyses, and power analyses

The following notebooks are provided to replicate the analysis, separated by each category of considered factors:

 * [**1_Physical.ipynb**](/1_Physical.ipynb)
 * [**2_Socioeconomic.ipynb**](/2_Socioeconomic.ipynb)
 * [**3_Psychological.ipynb**](/3_Psychological.ipynb)

Several common python libraries are used in the notebooks, in addition to custom scripts. It is recommended to run these notebooks using a [virtual environment](https://docs.python.org/3/library/venv.html). Once you have a virtual environment activated, you can install all dependencies with `pip install -r requirements.txt`.

## Other materials

A notebook visualizing correlations and replicating the empirical cumulative distribution functions can be found at: [**0_Exploratory.ipynb**](/0_Exploratory.ipynb).

For figures representing the estimated average treatment effect effects, please see the **img** subfolder.