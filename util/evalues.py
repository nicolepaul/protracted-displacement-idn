# References:
# VanderWeele, Tyler J., and Peng Ding. 2017. “Sensitivity Analysis in Observational Research: Introducing the E-Value.” Annals of Internal Medicine 167 (4): 268–74. https://doi.org/10.7326/M16-2607.
# VanderWeele, Tyler J., and Maya B. Mathur. 2020. “Commentary: Developing Best-Practice Guidelines for the Reporting of E-Values.” International Journal of Epidemiology 49 (5): 1495–97.

import numpy as np


# Convert risk difference to relative risk (or risk ratio)
def rd_to_rr(rd, p0):
    flip = False
    if rd < 0:
        rd = -rd
        p0 = 1 - p0 
        flip = True
    return 1 + (rd / p0), flip


# Calculate e-value (point)
def e_value_point(rr):
    flip = False
    if rr < 1:
        rr = 1.0 / rr
        flip = True
    return rr + np.sqrt(rr * (rr - 1)), flip


# Calculate e-value (confidence interval)
def e_value_ci(rr_lower, rr_upper):
    if rr_lower < -1 <= rr_upper:
        return 1.0
    if rr_lower > 1:
        rr = rr_lower
    else:
        rr = rr_upper
    return e_value_point(rr)
