
import numpy as np
from datetime import datetime
import pandas as pd
import gc
from numpy.random import choice
from os.path import exists
from sklearn.model_selection import train_test_split


# Define the covariates (predictor variables) for the model
# These include temporal variables (iweek, month, year),
# spatial variables (lat, lon, elevation),
# traffic emissions (various AADT variables),
# land use features (impervious surfaces, NDVI),
# meteorological variables (temperature, wind speed),
# air pollutant concentrations and other environmental factors
tcovs = ['iweek', 'lat', 'lon', 'ele30m', 'x2', 'y2', 'xy',
         'int_d_300m_c1', 'int_d_300m_c2', 'int_d_300m_c3', 'int_d_300m_c4',
         'int_d_5km_c1', 'int_d_5km_c2', 'int_d_5km_c3', 'int_d_5km_c4',
         'NONFWY_Light_AADT_50res_5000neigh_emissions', 'NONFWY_Light_AADT_50res_300neigh_emissions',
         'NONFWY_Heavy_AADT_50res_5000neigh_emissions', 'NONFWY_Heavy_AADT_50res_300neigh_emissions',
         'FWY_Light_AADT_50res_5000neigh_emissions', 'FWY_Light_AADT_50res_300neigh_emissions',
         'FWY_Heavy_AADT_50res_5000neigh_emissions', 'FWY_Heavy_AADT_50res_300neigh_emissions',
         'airport_distance', 'airport_highdistance', 'impervious', 'impervious_15km', 'rmax', 'rmin',
         'sph', 'srad', 'tmmn', 'tmmx', 'vs', 'BCMASS_BOT', 'CO_BOT', 'DD_O3_BOT', 'GMITO3',
         'NIMASS25_BOT', 'NO_BOT', 'NO2_BOT', 'O3_BOT', 'PL_BOT', 'PS', 'SSMASS25_BOT', 'NDVI1km',
         'NDVI15km', 'ozone24mean', 'Population_300m', 'ele30m', 'poirest', 'poifast', 'no2_pv1',
         'no2_pv2', 'nox_pv1', 'nox_pv2', 'U2M', 'V2M', 'U10M', 'V10M', 'U50M', 'V50M', 'PBLH', 'stagnation',
         'mixing', 'day_y', 'month', 'year', 'day_y_sine', 'day_y_cos', 'r50nlcd_22_1km_log',
         'r50nlcd_23_1km_log', 'r50nlcd_24_1km_log', 'r50nlcd_21_1km_log', 'r50nlcd_31_1km_log',
         'r50nlcd_22_15km_log', 'r50nlcd_23_15km_log', 'r50nlcd_24_15km_log',
         'r50nlcd_21_15km_log', 'r50nlcd_31_15km_log']

def loadData(tfl=None):
    if tfl is None:
        tfl = "/disk10t/phynoexmodel/input/alldata4train_sbasis_upndvi.csv"
    data = pd.read_csv(tfl)
    data['no2_m_log'] = np.log(data['no2_m'])
    data['nox_m_log'] = np.log(data['nox_m'])
    data['start_date1'] = data['start_date'].apply(datetime.strptime, args=('%Y-%m-%d',))
    data['pandamic_id'] = 1
    data.loc[data[data['start_date1'] < datetime.strptime('2020-03-01', '%Y-%m-%d')].index, 'pandamic_id'] = 0
    data['newid2'] = data['newid2'].astype('str')
    data = data.dropna()
    print('data size:', data.shape)
    return data


def selectedSiteSrc(data, srcStr, srcs, idStr, prop=0.2):
    subdata = data[data[srcStr].isin(srcs)]
    sgrp = subdata[idStr].value_counts()
    sitesCnt = pd.DataFrame({idStr: sgrp.index, 'samplesize': sgrp.values}, index=sgrp.index)
    np.random.seed()
    cnts = sitesCnt['samplesize'].astype(np.float64)
    sitesCnt['prop'] = cnts / np.sum(cnts)
    testsiteIndex = choice(np.array([i for i in range(len(sitesCnt))]), int(prop * len(sitesCnt)), replace=False,
                           p=sitesCnt['prop'].values)
    selsites = sitesCnt.iloc[testsiteIndex][idStr].copy()
    del sitesCnt
    gc.collect()
    return selsites.values


def selectSites(data, idStr='id', stratified='source', prop=0.2, tpath=None):
    if tpath is None:
        pFile = '/disk10t/phynoexmodel/model_tf_out/selsites.csv'
    else:
        pFile = tpath + '/selsites.csv'
    if not exists(pFile):
        selsites_aqs = selectedSiteSrc(data, 'source', ['AQS'], 'newid2', prop=prop)
        selsites_uci = selectedSiteSrc(data, 'source', ['UCI'], 'newid2', prop=prop)
        selsites_ucla = selectedSiteSrc(data, 'source', ['UCLA'], 'newid2', prop=prop)
        selsites_usc = selectedSiteSrc(data, 'source', ['USC'], 'newid2', prop=prop)
        selsites = np.concatenate((selsites_aqs, selsites_usc, selsites_uci, selsites_ucla))
        print('AQS:', len(selsites_aqs), 'USC:', len(selsites_usc), 'UCI:', len(selsites_uci),
              'UCLA:', len(selsites_ucla))
        pd.DataFrame({idStr: selsites}).to_csv(pFile, index=False)
    else:
        selsites = pd.read_csv(pFile)
        selsites = selsites[idStr].values
    trainsitesIndex = np.where(~data[idStr].isin(selsites))[0]
    indTestsitesIndex = np.where(data[idStr].isin(selsites))[0]
    tfl = tpath + '/selsites_samples.csv'
    data.iloc[indTestsitesIndex].to_csv(tfl, index=False)
    print("Selected sites: ", len(selsites))
    print("Selected sites index: ", indTestsitesIndex)
    print("Selected sites index size: ", len(indTestsitesIndex), indTestsitesIndex.shape)
    return trainsitesIndex, indTestsitesIndex


def train_test_splitR(data, targetindex, stratified, test_size=0.2):
    dfindex = data.iloc[targetindex].index
    sgrp = data.iloc[targetindex][stratified].value_counts()
    data.loc[dfindex, 'stratifed_cnt'] = sgrp.loc[data.loc[dfindex, stratified]].values
    pos1_index = np.where(data['stratifed_cnt'] < 5)[0]
    posT_index = np.where(data['stratifed_cnt'] >= 5)[0]
    np.random.seed()
    trainsiteIndex, testsiteIndex = train_test_split(posT_index,
                                                     stratify=data.iloc[posT_index][stratified],
                                                     test_size=test_size)
    trainsiteIndex = np.concatenate((trainsiteIndex, pos1_index))
    print('sampling slpit: ', len(trainsiteIndex), len(testsiteIndex))
    return (trainsiteIndex, testsiteIndex)
