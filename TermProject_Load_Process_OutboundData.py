# Benjamin Pittman
# University of Washington-Bothell
# CS 581 Autumn 2017
# 
# Walk feature extraction for training data: demographics:syn5511429 , table: syn5511449
# portions of below code adapted from https://github.com/Sage-Bionetworks/PDBiomarkerChallenge

import synapseclient
# Synapse login - registration is required to access data and run the following code
# use syn = synapseclient.login() if you've already set up your config file
# syn = synapseclient.login(email='YOUR EMAIL ADDRESS', password='YOUR PASSWORD', rememberMe=True)
syn = synapseclient.login()

import pandas as pd
import json
import numpy as np
from scipy.stats import kurtosis, skew

# read in the healthCodes and professional-diagnoses of interest from demographics training table
demo_syntable = syn.tableQuery("SELECT * FROM syn5511429")
demo = demo_syntable.asDataFrame()
healthCodeList = ", ".join( repr(i) for i in demo["healthCode"]) 
hcList = [i for i in demo["healthCode"]]
PD_list = [i for i in demo["professional-diagnosis"]]

# Query 'walking training table' for walk data recordIDs and healthCodes. 
INPUT_WALKING_ACTIVITY_TABLE_SYNID = "syn5511449"
actv_walking_syntable = syn.tableQuery(('SELECT "recordId", "healthCode", "appVersion", "phoneInfo", "medTimepoint", "deviceMotion_walking_outbound.json.items" FROM {0} WHERE healthCode IN ({1}) AND "deviceMotion_walking_outbound.json.items" is not null').format(INPUT_WALKING_ACTIVITY_TABLE_SYNID, healthCodeList))
actv_walking = actv_walking_syntable.asDataFrame()
actv_walking['idx'] = actv_walking.index

# Pull professional diagnoses from demographics synapse tables and 
# join to our main dataframe table on healthCode (patient ID)
PD_Diagnosis_dict = {}
for i in range(len(hcList)):
    PD_Diagnosis_dict[hcList[i]] = PD_list[i]
hcList_to_join = []
for i in range(len(actv_walking['healthCode'])):
    hcList_to_join.append(PD_Diagnosis_dict[actv_walking.iloc[i]['healthCode']])
hcList_to_series = pd.Series(hcList_to_join)
actv_walking['professional-diagnosis'] = hcList_to_series.values


######################
# Download JSON Files
######################
# bulk download walk JSON files containing sensor data
walk_json_files = syn.downloadTableColumns(actv_walking_syntable, "deviceMotion_walking_outbound.json.items")
items = walk_json_files.items()

# create pandas dataframe of JSON filepaths and filehandleIDs
walk_json_files_temp = pd.DataFrame({"deviceMotion_walking_outbound.json.items": [i[0] for i in items], "outbound_walk_json_file": [i[1] for i in items]})

# convert ints to strings for merging
actv_walking["deviceMotion_walking_outbound.json.items"] = actv_walking["deviceMotion_walking_outbound.json.items"].astype(str)

# merge IDs/healthCodes with JSON data
actv_walk_temp = pd.merge(actv_walking, walk_json_files_temp, on="deviceMotion_walking_outbound.json.items")

####################
# Feature Extraction
####################
# constant for number of samples
sampleSize = 1000
# Helper function to derive distance from time
def CalculateRange(a, t):
    velocity = []
    distance = []
    velocity.append(0)
    distance.append(0)
    for i in (range(len(a) - 1)):
        velocity.append(abs(a[i])*(t[i + 1] - t[i]) + velocity[i])
        distance.append(velocity[i + 1]*(t[i + 1] - t[i]))
    return sum(distance)

# helper function for simple moving average calculation
def SimpleMovingAverage(myList):
    N = 5
    cumsum, moving_aves = [0], []

    for i, x in enumerate(myList, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N

            moving_aves.append(moving_ave)
    return moving_aves


# Feature arrays to hold features extracted from json values
# and attached to Dataframe
sampling_ok = []

g_range_x = []
g_range_y = []
g_range_z = []

g_mean_x = []
g_mean_y = []
g_mean_z = []

range_linear_acc_x = []
range_linear_acc_y = []
range_linear_acc_z = []

mean_linear_acc_x = []
mean_linear_acc_y = []
mean_linear_acc_z = []

range_rot_acc_rx = []
range_rot_acc_ry = []
range_rot_acc_rz = []

mean_rot_acc_rx = []
mean_rot_acc_ry = []
mean_rot_acc_rz = []

skew_rfft_linear_acc_x = []
skew_rfft_linear_acc_y = []
skew_rfft_linear_acc_z = []
skew_rfft_gravity_x = []
skew_rfft_gravity_y = []
skew_rfft_gravity_z = []
skew_rfft_rot_acc_rx = []
skew_rfft_rot_acc_ry = []
skew_rfft_rot_acc_rz = []
skew_rfft_attitude_x = []
skew_rfft_attitude_y = []
skew_rfft_attitude_z = []
skew_rfft_attitude_w = []

kurtosis_rfft_linear_acc_x = []
kurtosis_rfft_linear_acc_y = []
kurtosis_rfft_linear_acc_z = []
kurtosis_rfft_gravity_x = []
kurtosis_rfft_gravity_y = [] 
kurtosis_rfft_gravity_z = []
kurtosis_rfft_rot_acc_rx = []
kurtosis_rfft_rot_acc_ry = []
kurtosis_rfft_rot_acc_rz = []
kurtosis_rfft_attidude_x = []
kurtosis_rfft_attidude_y = []
kurtosis_rfft_attidude_z = []
kurtosis_rfft_attidude_w = []

sum_variances_g = []
sum_std_g = []

sum_std_linear_acc = []
sum_variances_linear_acc = []

sum_variances_rot_accel = []
sum_std_rot_acc = []

sum_variances_attitude = []
sum_std_attitude = []

x_zero_cross_rate = []
y_zero_cross_rate = []
z_zero_cross_rate = []


# loop through each row in dataframe to read in json file
# and create objects to attach to our Dataframe for processing
# into features later
for row in actv_walk_temp["outbound_walk_json_file"]: 
    with open(row) as json_data:
        # motion_object = MotionData.MotionData()
        data = json.load(json_data)
        t = []
        x = []
        y = []
        z = []
        rx = []
        ry = []
        rz = []
        ax = []
        ay = []
        az = []
        aw = []
        gx = []
        gy = []
        gz = []

        for item in data:
            t.append(item.get("timestamp"))
            
            x.append(item.get("userAcceleration").get("x"))
            y.append(item.get("userAcceleration").get("y"))
            z.append(item.get("userAcceleration").get("z"))
            
            ax.append(item.get("attitude").get("x"))
            ay.append(item.get("attitude").get("y"))
            az.append(item.get("attitude").get("z"))
            aw.append(item.get("attitude").get("w"))
            
            rx.append(item.get("rotationRate").get("x"))
            ry.append(item.get("rotationRate").get("y"))
            rz.append(item.get("rotationRate").get("z"))
            
            gx.append(item.get("gravity").get("x"))
            gy.append(item.get("gravity").get("y"))
            gz.append(item.get("gravity").get("z"))
            
        
        # Check for sufficient data size
        if len(t) < sampleSize:
            sampling_ok.append(False)
        else:
            sampling_ok.append(True)

        t_zero = t[0]
        t[:] = [n - t_zero for n in t]

        t = np.array(t[10:sampleSize])
        
        x = np.array(x[10:sampleSize])
        y = np.array(y[10:sampleSize])
        z = np.array(z[10:sampleSize])
    
        gx = np.array(gx[10:sampleSize])
        gy = np.array(gy[10:sampleSize])
        gz = np.array(gz[10:sampleSize])
        
        rx = np.array(rx[10:sampleSize])
        ry = np.array(ry[10:sampleSize])
        rz = np.array(rz[10:sampleSize])

        ax = np.array(ax[10:sampleSize])
        ay = np.array(ay[10:sampleSize])
        az = np.array(az[10:sampleSize])
        aw = np.array(aw[10:sampleSize])
        
        gx_range = CalculateRange(gx, t)
        gy_range = CalculateRange(gy, t)
        gz_range = CalculateRange(gz, t)
        
        '''
        # Compares gravitational displacement as a test for proper 
        # phone orientation
        if ((gy_range < gz_range) or (gy_range < gx_range)):
            sampling_ok[-1] = False
        '''

        # Fourier transforms of each data set
        ft_x = abs(np.fft.rfft(x))
        ft_y = abs(np.fft.rfft(y))
        ft_z = abs(np.fft.rfft(z))

        ft_gx = abs(np.fft.rfft(gx))
        ft_gy = abs(np.fft.rfft(gy))
        ft_gz = abs(np.fft.rfft(gz))

        ft_rx = abs(np.fft.rfft(rx))
        ft_ry = abs(np.fft.rfft(ry))
        ft_rz = abs(np.fft.rfft(rz))

        ft_ax = abs(np.fft.rfft(ax))
        ft_ay = abs(np.fft.rfft(ay))
        ft_az = abs(np.fft.rfft(az))
        ft_aw = abs(np.fft.rfft(aw))

        g_range_x.append(gx_range)
        g_range_y.append(gy_range)
        g_range_z.append(gz_range)

        g_mean_x.append(np.mean(abs(gx)))
        g_mean_y.append(np.mean(abs(gy)))
        g_mean_z.append(np.mean(abs(gz)))

        range_linear_acc_x.append(CalculateRange(x, t))
        range_linear_acc_y.append(CalculateRange(y, t))
        range_linear_acc_z.append(CalculateRange(z, t))

        mean_linear_acc_x.append(np.mean(abs(x)))
        mean_linear_acc_y.append(np.mean(abs(y)))
        mean_linear_acc_z.append(np.mean(abs(z)))

        range_rot_acc_rx.append(CalculateRange(rx, t))
        range_rot_acc_ry.append(CalculateRange(ry, t)) 
        range_rot_acc_rz.append(CalculateRange(rz, t))  

        mean_rot_acc_rx.append(np.mean(abs(rx)))
        mean_rot_acc_ry.append(np.mean(abs(ry)))
        mean_rot_acc_rz.append(np.mean(abs(rz)))

        skew_rfft_linear_acc_x.append(skew(ft_x, bias=False))
        skew_rfft_linear_acc_y.append(skew(ft_y, bias=False))
        skew_rfft_linear_acc_z.append(skew(ft_z, bias=False))

        skew_rfft_gravity_x.append(skew(ft_gx, bias=False))
        skew_rfft_gravity_y.append(skew(ft_gy, bias=False))
        skew_rfft_gravity_z.append(skew(ft_gz, bias=False))

        skew_rfft_rot_acc_rx.append(skew(ft_rx, bias=False))
        skew_rfft_rot_acc_ry.append(skew(ft_ry, bias=False))
        skew_rfft_rot_acc_rz.append(skew(ft_rz, bias=False))

        skew_rfft_attitude_x.append(skew(ft_ax, bias=False))
        skew_rfft_attitude_y.append(skew(ft_ay, bias=False))
        skew_rfft_attitude_z.append(skew(ft_az, bias=False))
        skew_rfft_attitude_w.append(skew(ft_aw, bias=False))

        kurtosis_rfft_linear_acc_x.append(kurtosis(ft_x, bias=False))
        kurtosis_rfft_linear_acc_y.append(kurtosis(ft_y, bias=False))
        kurtosis_rfft_linear_acc_z.append(kurtosis(ft_z, bias=False))

        kurtosis_rfft_gravity_x.append(kurtosis(ft_gx, bias=False))
        kurtosis_rfft_gravity_y.append(kurtosis(ft_gy, bias=False))
        kurtosis_rfft_gravity_z.append(kurtosis(ft_gz, bias=False))

        kurtosis_rfft_rot_acc_rx.append(kurtosis(ft_rx, bias=False))
        kurtosis_rfft_rot_acc_ry.append(kurtosis(ft_ry, bias=False))
        kurtosis_rfft_rot_acc_rz.append(kurtosis(ft_rz, bias=False))

        kurtosis_rfft_attidude_x.append(kurtosis(ft_ax, bias=False))
        kurtosis_rfft_attidude_y.append(kurtosis(ft_ay, bias=False))
        kurtosis_rfft_attidude_z.append(kurtosis(ft_az, bias=False))
        kurtosis_rfft_attidude_w.append(kurtosis(ft_aw, bias=False))

        sum_variances_g.append(np.var(gx) + np.var(gy) + np.var(gz))
        sum_std_g.append(np.std(gx) + np.std(gy) + np.std(gz))

        sum_variances_linear_acc.append(np.var(x) + np.var(y) + np.var(z))
        sum_std_linear_acc.append(np.std(x) + np.std(y) + np.std(z))

        sum_variances_rot_accel.append(np.var(rx) + np.var(ry) + np.var(rz))
        sum_std_rot_acc.append(np.std(rx) + np.std(ry) + np.std(rz))

        sum_variances_attitude.append(np.var(ax) + np.var(ay) + np.var(az) + np.var(aw))
        sum_std_attitude.append(np.std(ax) + np.std(ay) + np.std(az) + np.std(aw))

        x_crosses = np.nonzero(np.diff(x > 0))[0]
        y_crosses = np.nonzero(np.diff(y > 0))[0]
        z_crosses = np.nonzero(np.diff(z > 0))[0]

        x_zero_cross_rate.append(x_crosses.size / t[-1])
        y_zero_cross_rate.append(y_crosses.size / t[-1])
        z_zero_cross_rate.append(z_crosses.size / t[-1])

# add feature columns to Dataframe 
actv_walk_temp['Outbound_g_range_x'] = g_range_x
actv_walk_temp['Outbound_g_range_y'] = g_range_y
actv_walk_temp['Outbound_g_range_z'] = g_range_z
actv_walk_temp['Outbound_g_mean_x'] = g_mean_x
actv_walk_temp['Outbound_g_mean_y'] = g_mean_y
actv_walk_temp['Outbound_g_mean_z'] = g_mean_z
actv_walk_temp['Outbound_range_linear_acc_x'] = range_linear_acc_x
actv_walk_temp['Outbound_range_linear_acc_y'] = range_linear_acc_y
actv_walk_temp['Outbound_range_linear_acc_z'] = range_linear_acc_z
actv_walk_temp['Outbound_mean_linear_acc_x'] = mean_linear_acc_x
actv_walk_temp['Outbound_mean_linear_acc_y'] = mean_linear_acc_y
actv_walk_temp['Outbound_mean_linear_acc_z'] = mean_linear_acc_z
actv_walk_temp['Outbound_range_rot_acc_rx'] = range_rot_acc_rx
actv_walk_temp['Outbound_range_rot_acc_ry'] = range_rot_acc_ry
actv_walk_temp['Outbound_range_rot_acc_rz'] = range_rot_acc_rz
actv_walk_temp['Outbound_mean_rot_acc_rx'] = mean_rot_acc_rx
actv_walk_temp['Outbound_mean_rot_acc_ry'] = mean_rot_acc_ry
actv_walk_temp['Outbound_mean_rot_acc_rz'] = mean_rot_acc_rz

actv_walk_temp['Outbound_skew_rfft_linear_acc_x'] = skew_rfft_linear_acc_x
actv_walk_temp['Outbound_skew_rfft_linear_acc_y'] = skew_rfft_linear_acc_y
actv_walk_temp['Outbound_skew_rfft_linear_acc_z'] = skew_rfft_linear_acc_z
actv_walk_temp['Outbound_skew_rfft_gravity_x'] = skew_rfft_gravity_x
actv_walk_temp['Outbound_skew_rfft_gravity_y'] = skew_rfft_gravity_y
actv_walk_temp['Outbound_skew_rfft_gravity_z'] = skew_rfft_gravity_z
actv_walk_temp['Outbound_skew_rfft_rot_acc_rx'] = skew_rfft_rot_acc_rx
actv_walk_temp['Outbound_skew_rfft_rot_acc_ry'] = skew_rfft_rot_acc_ry
actv_walk_temp['Outbound_skew_rfft_rot_acc_rz'] = skew_rfft_rot_acc_rz
actv_walk_temp['Outbound_skew_rfft_attitude_x'] = skew_rfft_attitude_x
actv_walk_temp['Outbound_skew_rfft_attitude_y'] = skew_rfft_attitude_y
actv_walk_temp['Outbound_skew_rfft_attitude_z'] = skew_rfft_attitude_z

actv_walk_temp['Outbound_kurtosis_rfft_linear_acc_x'] = kurtosis_rfft_linear_acc_x
actv_walk_temp['Outbound_kurtosis_rfft_linear_acc_y'] = kurtosis_rfft_linear_acc_y
actv_walk_temp['Outbound_kurtosis_rfft_linear_acc_z'] = kurtosis_rfft_linear_acc_z
actv_walk_temp['Outbound_kurtosis_rfft_gravity_x'] = kurtosis_rfft_gravity_x
actv_walk_temp['Outbound_kurtosis_rfft_gravity_y'] = kurtosis_rfft_gravity_y
actv_walk_temp['Outbound_kurtosis_rfft_gravity_z'] = kurtosis_rfft_gravity_z
actv_walk_temp['Outbound_kurtosis_rfft_rot_acc_rx'] = kurtosis_rfft_rot_acc_rx
actv_walk_temp['Outbound_kurtosis_rfft_rot_acc_ry'] = kurtosis_rfft_rot_acc_ry
actv_walk_temp['Outbound_kurtosis_rfft_rot_acc_rz'] = kurtosis_rfft_rot_acc_rz
actv_walk_temp['Outbound_kurtosis_rfft_attidude_x'] = kurtosis_rfft_attidude_x
actv_walk_temp['Outbound_kurtosis_rfft_attidude_y'] = kurtosis_rfft_attidude_y
actv_walk_temp['Outbound_kurtosis_rfft_attidude_z'] = kurtosis_rfft_attidude_z
actv_walk_temp['Outbound_kurtosis_rfft_attidude_w'] = kurtosis_rfft_attidude_w

actv_walk_temp['Outbound_sum_variances_g'] = sum_variances_g
actv_walk_temp['Outbound_sum_std_g'] = sum_std_g
actv_walk_temp['Outbound_sum_std_linear_acc'] = sum_std_linear_acc
actv_walk_temp['Outbound_sum_variances_linear_acc'] = sum_variances_linear_acc
actv_walk_temp['Outbound_sum_variances_rot_accel'] = sum_variances_rot_accel
actv_walk_temp['Outbound_sum_std_rot_acc'] = sum_std_rot_acc
actv_walk_temp['Outbound_sum_variances_attitude'] = sum_variances_attitude
actv_walk_temp['Outbound_sum_std_attitude'] = sum_std_attitude

actv_walk_temp['Outbound_x_zero_cross_rate'] = x_zero_cross_rate
actv_walk_temp['Outbound_y_zero_cross_rate'] = y_zero_cross_rate
actv_walk_temp['Outbound_z_zero_cross_rate'] = z_zero_cross_rate

# Remove unnecessary columns
actv_walk_temp.drop(["deviceMotion_walking_outbound.json.items", "outbound_walk_json_file"], axis=1, inplace=True)

# convert sample size test array to series and use to eliminate rows
not_enough_samples = pd.Series(sampling_ok)
reduced_data_set = actv_walk_temp[not_enough_samples]

reduced_data_set.to_csv('mPowerFeatrues_outbound_1205_reduced.csv')
