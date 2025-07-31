# Protracted displacement after disasters

This repository includes code to estimate direct causal effects of physical, socioeconomic, and psychological factors on displacement duration outcomes using household survey data following the 2018 Central Sulawesi earthquake, tsunami, and liquefaction disaster in Indonesia. The causal effects are estimated using double machine learning (DML).

For materials related to the housheold survey itself, including the anonymized household-level responses and survey instrument, please refer to the repository TODO. A smaller subset of the anonymized household-level responses are included herein for replication purposes.

## Running the notebooks

The [**Drivers_DML.ipynb**](/Drivers_DML.ipynb) notebook is the main way to run the code. 

Several common python libraries are used in the notebooks, in addition to custom scripts. It is recommended to run these notebooks using a [virtual environment](https://docs.python.org/3/library/venv.html). Once you have a virtual environment activated, you can install all dependencies with `pip install -r requirements.txt`.