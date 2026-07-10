import pvlib
import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_squared_error
import math
# On timestamps:
# The camera scans while rotating, so an image labelled 15:30 was actually captured somewhere in 15:30-15:40.
# EPFL GHI mean is stamped by the left edge (15:30 covers 15:30-15:40); the json times are UTC (Unix).
# MeteoSwiss is stamped by the right edge (15:30 covers 15:20-15:30) and is in UTC.
# Read data
def get_GHI_data(path,key,suffix = '_gre000z0_1_data.txt',new_name = 'GHI_mean'):
    df = pd.read_csv(path+key+suffix, delimiter = ';', header = [0], na_values='-').drop(columns=['stn'])
    # Convert time columns to Zurich datetime
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M').dt.tz_localize('UTC').dt.tz_convert('Europe/Zurich').dt.tz_localize(None)
    df.rename(columns={'gre000z0': new_name}, inplace=True)

    #labelled by right edge time,should change to left edge time
    df['time'] = df['time'] - pd.Timedelta(minutes=10)
    return df

def get_meteo_data(file, new_name):
    df = pd.read_csv(file, delimiter = ';', header = [0], na_values='-').drop(columns=['stn'])
    # Convert time columns to Zurich datetime
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M').dt.tz_localize('UTC').dt.tz_convert('Europe/Zurich').dt.tz_localize(None)
    df.columns = ['time', new_name]

    #labelled by left edge time,should change to right edge time
    df['time'] = df['time'] - pd.Timedelta(minutes=10)

    return df

def get_GHI_data_w_reference(path, key,Location_coor):
    df = get_GHI_data(path,key)
    # if os.path.exists(path+key+'_gor000za_1_data.txt'):
    #     df_std = get_GHI_data(path,key,suffix='_gor000za_1_data.txt',new_name='GHI_std')

    # Get clear sky GHI (using 2 models)
    ghi_ineichen = ghi_clear_sky(latitude=Location_coor[key][0],longitude=Location_coor[key][1],start=df['time'].min(),end=df['time'].max(),model = 'ineichen')
    ghi_ineichen.rename(columns={'value_cs':'ghi_ineichen'},inplace=True)

    ghi_haurwitz = ghi_clear_sky(latitude=Location_coor[key][0],longitude=Location_coor[key][1],start=df['time'].min(),end=df['time'].max(),model = 'haurwitz')
    ghi_haurwitz.rename(columns={'value_cs':'ghi_haurwitz'},inplace=True)

    df = pd.merge(df, ghi_ineichen, on='time', how='left')
    
    df_GHI = pd.merge(df, ghi_haurwitz, on='time', how='left')

    return df_GHI

def get_EPFL_GHI_data(key,Location_coor,file_name,max_min = False):
    # file name if end with json:
    if file_name.endswith('.json'):
        # EPFL data downloaded as Unix time, in UTC
        with open(file_name, 'r') as f: #'MeteoSwiss/GHI_EPFL_Jan2023_Sep2023.json'
            data_epfl = json.load(f)['results'][0]['series'][0]['values'] 
        df_epfl_ori = pd.DataFrame(data_epfl, columns=['time', 'EPFL_value'])#.dropna()
        # data initially in GTM, change to Zurich time
        df_epfl_ori['time'] = pd.to_datetime(df_epfl_ori['time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Zurich').dt.tz_localize(None)
    elif file_name.endswith('.csv'): # csv file: timestamps are in local time
        data_epfl = pd.read_csv(file_name, header = [0])
        df_epfl_ori = pd.DataFrame(data_epfl, columns=['time', 'EPFL_value'])#.dropna()
        # data initially in GTM, change to Zurich time
        df_epfl_ori['time'] = pd.to_datetime(df_epfl_ori['time'], format="%d.%m.%Y %H:%M")
        
    
    df_epfl_ori.set_index('time', inplace=True)

    if max_min:
        df_epfl = df_epfl_ori.resample('10Min').agg(['mean', 'std', 'max', 'min'])
        df_epfl.columns = ['GHI_mean', 'GHI_std', 'GHI_max', 'GHI_min']
    else:
        df_epfl = df_epfl_ori.resample('10Min').agg(['mean', 'std'])#labelled by left edge time
        df_epfl.columns = ['GHI_mean', 'GHI_std']

    #df_epfl.index = df_epfl.index + pd.Timedelta(minutes=10)
   
    df_epfl.reset_index(inplace=True)

    # Get clear sky GHI (using 2 models)
    ghi_ineichen = ghi_clear_sky(latitude=Location_coor[key][0],longitude=Location_coor[key][1],start=df_epfl['time'].min(),end=df_epfl['time'].max(),model = 'ineichen')
    ghi_ineichen.rename(columns={'value_cs':'ghi_ineichen'},inplace=True)

    ghi_haurwitz = ghi_clear_sky(latitude=Location_coor[key][0],longitude=Location_coor[key][1],start=df_epfl['time'].min(),end=df_epfl['time'].max(),model = 'haurwitz')
    ghi_haurwitz.rename(columns={'value_cs':'ghi_haurwitz'},inplace=True)

    df_epfl = pd.merge(df_epfl, ghi_ineichen, on='time', how='left')
    df_GHI = pd.merge(df_epfl, ghi_haurwitz, on='time', how='left')
    
    # clean epfl data
    df_GHI['time'] = pd.to_datetime(df_GHI['time'])
    exclude_dates = [
    ('2022-10-04', '2022-10-07'),
    ('2022-10-10', '2022-10-11'),
    ('2022-10-12', '2022-10-13'),
    ('2022-10-17', '2022-10-29'),
    ('2022-11-23', '2022-11-24')]

    mask = ~df_GHI['time'].between(exclude_dates[0][0], exclude_dates[0][1])
    for start, end in exclude_dates[1:]:
        mask &= ~df_GHI['time'].between(start, end)

    df_GHI = df_GHI[mask]
    return df_GHI


#### model can be: ‘ineichen’, ‘haurwitz’, ‘simplified_solis’.
def ghi_clear_sky(latitude,longitude,start,end, model='haurwitz', altitude=0):
    # tz info is required or pvlib assumes UTC: localise start/end to Zurich then convert (they carry no geo info yet)
    start = pd.to_datetime(start).tz_localize('Europe/Zurich').tz_convert('UTC')
    end = pd.to_datetime(end).tz_localize('Europe/Zurich').tz_convert('UTC')
    times = pd.date_range(start, end, freq='10Min').tz_convert('Europe/Zurich')
    #times = pd.date_range(start, end, freq='10Min').tz_localize('UTC').tz_convert('Europe/Zurich')#.tz_convert('Europe/Zurich')
    loc = pvlib.location.Location(latitude=latitude, longitude=longitude, altitude=altitude)

    # Get solar position: ps DataFrame has the following columns:
    # apparent_elevation, elevation, apparent_azimuth, azimuth, apparent_zenith, zenith.
    # apparent terms means it counts for the effects of atmospheric refraction
    ps = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    ps = ps[['apparent_zenith', 'zenith','apparent_elevation']]

    #get GHI values
    cs= loc.get_clearsky(times, model=model,solar_position=ps)

    ghi_tmp = cs['ghi']
    ghi_tmp = ghi_tmp.reset_index()
    ghi_tmp.columns = ['time', 'value_cs']
    # now, time is converted back to zurich time automatically
    ghi_tmp['time'] = ghi_tmp['time'].dt.tz_localize(None)
    ghi_tmp.drop_duplicates(subset='time',inplace=True)
    return ghi_tmp


def add_clearsky_columns(df, latitude, longitude, altitude=0, model='haurwitz', ghi_col='ghi'):
    """Add the clear-sky / GHI-index columns the forecasting pipeline needs.

    Given a time series that only has raw GHI (`time`, `ghi`), compute from the
    site coordinates (via pvlib, see `ghi_clear_sky`):

      - ``ghi_clear_sky``          : clear-sky GHI at each timestamp
      - ``day``                    : calendar date
      - ``GHI_daily_max_clearsky`` : per-day maximum of ``ghi_clear_sky``
      - ``GHI_percent_wrt_max``    : 100 * ghi / GHI_daily_max_clearsky  (the GHI index)

    This is the transform used to build the shipped ``ghi_{SITE}_pure_scaled.csv``
    files, exposed so a raw-GHI csv can be turned into a trainable one.
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    cs = ghi_clear_sky(latitude=latitude, longitude=longitude,
                       start=df['time'].min(), end=df['time'].max(),
                       model=model, altitude=altitude).rename(columns={'value_cs': 'ghi_clear_sky'})
    df = pd.merge(df, cs, on='time', how='left')
    df['day'] = df['time'].dt.date.astype(str)
    df['GHI_daily_max_clearsky'] = df.groupby('day')['ghi_clear_sky'].transform('max')
    df['GHI_percent_wrt_max'] = 100.0 * df[ghi_col] / df['GHI_daily_max_clearsky']
    return df


def get_weather_profile(df,range, reference = 'ghi_haurwitz'):
    df['abs_diff'] = abs(df['GHI_mean']-df[reference])
    # calculate mean and max
    df_daily_ghi = df.groupby(pd.Grouper(key='time',freq='D')).agg({'abs_diff':'mean','ghi_haurwitz':'max'}).rename(columns={'abs_diff':'abs_diff_mean'})
    df_daily_ghi['ghi_daily_mean_percent'] = df_daily_ghi['abs_diff_mean']/df_daily_ghi['ghi_haurwitz']*100
    df_daily_count = df.groupby(pd.Grouper(key='time',freq='D'))['abs_diff'].count().rename('count')
    df_daily = pd.merge(df_daily_ghi,df_daily_count,on='time')
    df_daily = df_daily.loc[df_daily['count']>30] # only keep days with more than 5h data

    days = df_daily.loc[(df_daily['ghi_daily_mean_percent']<range[1]) & (df_daily['ghi_daily_mean_percent']>=range[0])].index
    # Convert Timestamp objects in 'days' to Python's date objects
    days_as_dates = [ts.date() for ts in days]
    return df.loc[df['time'].dt.date.isin(days_as_dates)]


def RRSE(*args):
    if len(args) == 1 and isinstance(args[0], pd.DataFrame):
        df = args[0]
        y_true, y_pred = df['ghi_haurwitz'].values, df['GHI_mean'].values
    elif len(args) == 2:
        y_true, y_pred = args
    else:
        raise ValueError("Invalid input: Pass either a DataFrame or two arrays/lists.")
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.sqrt(numerator / denominator)

def RMSE(df,a='GHI_mean_x',b='GHI_mean_y'):
    if df.empty:
        return None
    return math.sqrt(mean_squared_error(df[a], df[b]))

#Daily average of abs difference between Ghi and ghi_clear_sky, In percentage of the max daily ghi_clear_sky
def daily_ghi_dev_mean_percent(df, start = None,end = None,min_count = None):
    if (start != None) and (end != None):
        slice_index = (df['time']> pd.to_datetime(start))&(df['time']< pd.to_datetime(end))
        df = df.loc[slice_index]
    df1 = df.copy()
    df1.loc[:,'abs_diff'] = abs(df1['GHI_mean']-df1['ghi_haurwitz'])
    # calculate mean and max
    df_daily_ghi = df1.groupby(df1['time'].dt.date).agg({'abs_diff':'mean','ghi_haurwitz':'max'})
    df_daily_ghi.rename(columns={'abs_diff':'abs_diff_mean','ghi_haurwitz':'ghi_haurwitz_max'},inplace=True)
    df_daily_ghi['ghi_daily_mean_percent'] = df_daily_ghi['abs_diff_mean']/df_daily_ghi['ghi_haurwitz_max']*100
    
    if min_count:
        df_daily_count = df1.groupby(pd.Grouper(key='time',freq='D'))['abs_diff'].count().rename('count')
        df_daily = pd.merge(df_daily_ghi,df_daily_count,on='time')
        df_daily = df_daily.loc[df_daily['count']>min_count] # remove the days with less than 100 data points(only the last day)
    else:
        df_daily = df_daily_ghi
    return df_daily

# Daily volotility of GHI, scaled by the max daily ghi_clear_sky
def daily_ghi_volatility(df,start = None,end = None):
    if (start != None) and (end != None):
        slice_index = (df['time']> pd.to_datetime(start))&(df['time']< pd.to_datetime(end))
        df = df.loc[slice_index]
    df1 = df.copy()
    df1 = df1.dropna(subset=['GHI_mean','ghi_haurwitz'])
    df1 = df1.loc[df1['ghi_haurwitz']>0]
    df1['diff'] = df1['GHI_mean']/df1['ghi_haurwitz']

    df2 = df1.groupby(df1['time'].dt.date).agg({'diff':'std','ghi_haurwitz':'max'})
    df2['Volatility'] = df2['diff']/df2['ghi_haurwitz']*100
    return df2


