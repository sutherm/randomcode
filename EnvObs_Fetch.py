# Daily Water Level Reporting Tool
# Created by Mike Sutherland

"""
This script creates a plot of daily water levels at a user-defined NOAA tide
gauge that can be utilized during the initial QC of topo-bathy lidar
data acquisition
"""
# Import required python modules/site-packages
import os
import datetime
from datetime import timedelta
from glob import glob
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import seaborn as sns
import subprocess
import urllib

def definelastoolsbin():
    ##user_id = 
    lastoolsbin = os.path.join(os.path.expanduser('~'), "GTS_Tools", "LAStools", "bin")
    return lastoolsbin


def utc_time_conv(gpstime):
    # utc = 1980-01-06UTC + (gps - (leap_count(current) - leap_count(1980)))
    gps_zero = datetime.datetime(1980, 1, 6)
    utc_time = gps_zero + timedelta(seconds=(1e+09 + gpstime) - (37 - 19))
    return utc_time


def gps_query(lidar_path, out_path):
    getgps = subprocess.Popen([os.path.join(definelastoolsbin(), "lasinfo.exe"),
                               "-cpu64",
                               "-cores", str(15),
                               "-i", os.path.join(lidar_path, "*.laz"),
                               "-nh", "-nv", "-nr", "-nw", "-odir", out_path,
                               "-odix", "_GPS", "-otxt"])
    getgps.wait()
    if getgps.returncode == 0:
        print("GPS times queried")

    gps_list = glob(os.path.join(out_path, '*.txt'))
    gps_begin_list = []
    gps_end_list = []
    for i in gps_list:
        with open(i, 'r') as gps_time:
            parse_gps = gps_time.read().splitlines(True)
            gps_begin = [line for line in parse_gps if "gps_time" in line][0].split()[-2]
            gps_end = [line for line in parse_gps if "gps_time" in line][0].split()[-1]
            gps_begin_list.append(gps_begin)
            gps_end_list.append(gps_end)

    gps_min = np.amin(np.array(gps_begin_list, dtype=np.float32))
    gps_max = np.amax(np.array(gps_end_list, dtype=np.float32))

    utc_start = utc_time_conv(gps_min)
    utc_end = utc_time_conv(gps_max)

    return utc_start, utc_end


def gage_data_api_query(start, stop, station, dtype):
    if dtype != "wind":
        url = f"https://tidesandcurrents.noaa.gov/api/datagetter?begin_date={start}&end_date={stop}&station={station}&product={dtype}&datum=MLLW&time_zone=gmt&units=metric&format=csv"
    else:
        url = f"https://tidesandcurrents.noaa.gov/api/datagetter?begin_date={start}&end_date={stop}&station={station}&product={dtype}&datum=STND&time_zone=gmt&units=english&format=csv"
    df = pd.read_csv(url, index_col="Date Time", parse_dates=True)
    return df


def gage_meta_api_query(station):
    url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{station}/datums.json?units=metric"
    metadata_req = urllib.request.urlopen(url)
    metadata = json.load(metadata_req)
    datums = json_normalize(metadata["datums"])
    datums.set_index("name", inplace=True)
    return datums


# Seaborn display adjustment
sns.set(font_scale=0.75, style="whitegrid")

# Workspace definition; USER INPUTS
lidar_dir = r"C:\Users\msutherland\script_testing"
project_dir = r"C:\Users\msutherland\script_testing"
dest_dir = os.path.join(project_dir, "Data", "Acquisition", "Env_Obs")
if os.path.exists(dest_dir) is False:
    os.makedirs(dest_dir)
dewlogo = os.path.join(project_dir, "dewberry.png")

# NOAA Tide Gauge Identification
station_id = "8632200"

# Determine acquisition start/stop times
gps_acq = gps_query(lidar_dir, project_dir)
acq_start_date = gps_acq[0].strftime('%Y%m%d')
acq_end_date = gps_acq[1].strftime('%Y%m%d')

# Query NOAA CO-OPS API to return water level and wind data at tide gauge
obs_wl = gage_data_api_query(acq_start_date, acq_end_date, station_id, "water_level")
pred_wl = gage_data_api_query(acq_start_date, acq_end_date, station_id, "predictions")
winds = gage_data_api_query(acq_start_date, acq_end_date, station_id, "wind")

# Query NOAA CO-OPS API for tidal datum metadata
tdatums = gage_meta_api_query(station_id)

# Calculate MHW offset for plotting
mhw = float(testing.loc[["MHW"]].value)
mllw = float(testing.loc[["MLLW"]].value)
tidal_offset = mhw - mllw

# Plotting Environmental Observations
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle(f"Environmental Conditions Report, NOAA Tide Station: {station_id}\n Lidar Acquisition Date: {acq_start_date}\n",
             fontsize=12)

# Water Level Plotting
obs_wl.plot(y=" Water Level", color="red", ax=ax1)
pred_wl.plot(ax=ax1, color="blue")
ax1.axhline(tidal_offset, color="darkviolet", alpha=0.5, linewidth=2.25)
ax1.axvspan(gps_acq[0],
            gps_acq[1],
            label="Lidar Acq", color="g", alpha=0.25)
ax1.set(ylabel="Water Level (meters relative to MLLW)", title="Water Level")
ax1.legend(["Observed", "Predicted",
            "MHW", "Lidar Acq"], fontsize=8,
            bbox_to_anchor=(1, 0.75),
            loc='upper left', prop={'size': 8})

# Wind Speed Plotting
winds.plot(y=" Gust", color="red", ax=ax2)
winds.plot(y=" Speed", color="blue", ax=ax2)
ax2.axvspan(gps_acq[0],
          gps_acq[1],
          label="Lidar Acq", color="g", alpha=0.25)
ax2.set(ylabel="Wind Speed (knots)", title="Wind Speed")
ax2.legend(["Gusts", "Winds", "Lidar Acq"], fontsize=8,
            bbox_to_anchor=(1, 0.75),
            loc='upper left', prop={'size': 8})
plt.show()

# Adding Dewberry watermark to figure
im = mpl.image.imread(dewlogo)
fig.figimage(im, xo=5, yo=175, zorder=4, alpha=.05, origin='upper')

# Saving figure as pdf
env_obs_pdf = os.path.join(
        dest_dir,
        f"EnvObs_{station_id}_{gps_acq[0].strftime('%Y%m%d')}.pdf")
fig.savefig(env_obs_pdf)

print("Water Level Plot Created")