import os
import yaml
import numpy as np
from cartopy import crs as ccrs
import numcodecs

# need to be globally consistent in the variable order  
ATMOS_VARS = ['CAPE@surface', 'DPT@1000mb', 'DPT@500mb', 'DPT@700mb', 'DPT@850mb', 'DPT@925mb', 
                 'HGT@1000mb', 'HGT@500mb', 'HGT@700mb', 'HGT@850mb', 'HGT@surface', 
                 'TMP@1000mb', 'TMP@500mb', 'TMP@700mb', 'TMP@850mb', 'TMP@925mb', 
                 'UGRD@1000mb', 'UGRD@250mb', 'UGRD@500mb', 'UGRD@700mb', 'UGRD@850mb', 'UGRD@925mb', 
                 'VGRD@1000mb', 'VGRD@250mb', 'VGRD@500mb', 'VGRD@700mb', 'VGRD@850mb', 'VGRD@925mb',
                 'TCDC@entire_atmosphere', 'HCDC@high_cloud_layer', 'MCDC@middle_cloud_layer', 'LCDC@low_cloud_layer', 
                 'PWAT@entire_atmosphere_single_layer', 'RHPW@entire_atmosphere', 'VIL@entire_atmosphere'] 
                     
TARGET_VARS =  ["TMP@2m_above_ground", "RH@2m_above_ground", "UGRD@10m_above_ground", "VGRD@10m_above_ground", 
                "GUST@surface", "DSWRF@surface", 'APCP_1hr_acc_fcst@surface']

assert not set(TARGET_VARS) & set(ATMOS_VARS), "TARGET_VARS and ATMOS_VARS must not overlap"

VAR_LEVELS = TARGET_VARS + ATMOS_VARS
    
TARGET_INDS = [VAR_LEVELS.index(var) for var in TARGET_VARS]
# the projection of the pjm area
projection = ccrs.LambertConformal(central_longitude=262.5, 
                                   central_latitude=38.5, 
                                   standard_parallels=(38.5, 38.5),
                                    globe=ccrs.Globe(semimajor_axis=6371229,
                                                     semiminor_axis=6371229))

pjm_chunk_ids = np.array([
                          ["4.9", "4.10", "4.11"], 
                          ["5.9", "5.10", "5.11"], 
                          ["6.9", "6.10", "6.11"], 
                         ])

# The NE area is defined by chunk rows 4-6 and chunk cols 9-11 (each chunk is 150x150 pixels, 3km res).
# Full HRRR grid origin (derived from lib/data_info.py PJM coords):
#   x_full[0] = 452479.8574780696 - 1050*3000 = -2697520.1425219304
#   y_full[0] = -237306.1525566636 - 450*3000  = -1587306.1525566636
# NE slice in full HRRR grid: y[600:1050], x[1350:1800]  (3 chunks x 150 = 450 pixels each side)
data_slice = {"y": slice(600, 1050), "x": slice(1350, 1799)}

xy_coords = {"x": 1352479.8574780696 + np.arange(449) * 3000,   # -2697520.14 + 1350*3000
             "y": 212693.8474433364  + np.arange(450) * 3000}   # -1587306.15 + 600*3000

#ne_extent = {"xy": [1352479.8574780696, 2699479.8574780696,      # x_min, x_max
#                    212693.8474433364,  1559693.8474433364]}

anl_encoding = {"chunks": (24, 450, 450),
                "compressor": numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1)}


