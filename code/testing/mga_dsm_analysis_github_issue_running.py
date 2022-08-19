import numpy as np
import matplotlib.pyplot as plt
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy.kernel import constants
from tudatpy.kernel.math import root_finders

import mga_dsm_analysis_github_issue as func
import lowthrust_earthmars_self_v3 as func2

#func.mga_with_hodographic_shaping()
func2.mga_with_hodographic_shaping()

