import numpy as np
import time
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def build_data():
    input_path = 'C:/Users/yinan/Desktop/gnn-residual-correlation-master/datasets/election/'

    election = pd.read_csv(input_path + 'election.csv')
    education = pd.read_csv(input_path + 'education.csv')
    income = pd.read_csv(input_path + 'income.csv')
    population = pd.read_csv(input_path + 'population.csv')
    unemployment = pd.read_csv(input_path + 'unemployment.csv')

    uic_2013 = pd.read_excel('C:/Users/yinan/Desktop/gnn-residual-correlation-master/UrbanInfluenceCodes2013.xls')

    data = pd.DataFrame(election, columns=['fips_code', 'county'])

    data["dem_2012"] = election[
        "dem_2012"]  # , "total_2012"]]#.apply(lambda x: x["dem_2012"] / x["total_2012"], axis=1)
    data["gop_2012"] = election[
        "gop_2012"]  # , "total_2012"]]#.apply(lambda x: x["gop_2012"] / x["total_2012"], axis=1)
    data["dem_2016"] = election[
        "dem_2016"]  # , "total_2016"]]#.apply(lambda x: x["dem_2016"] / x["total_2016"], axis=1)
    data["gop_2016"] = election[
        "gop_2016"]  # , "total_2016"]]#.apply(lambda x: x["gop_2016"] / x["total_2016"], axis=1)

    data = pd.merge(data, education, how='left', on='fips_code')
    data['BachelorRate2012'] = data['BachelorRate2012'].map(lambda x: x / 100)
    data['BachelorRate2016'] = data['BachelorRate2016'].map(lambda x: x / 100)

    income_tem = pd.DataFrame(income, columns=['fips_code', 'MedianIncome2012', 'MedianIncome2016'])
    data = pd.merge(data, income_tem, how='left', on='fips_code')

    population_tem = pd.DataFrame(population, columns=['fips_code', 'POP_ESTIMATE_2012', 'POP_ESTIMATE_2016',
                                                       'Economic_typology_2015'])
    data = pd.merge(data, population_tem, how='left', on='fips_code')
    data["N_POP_CHG_2016"] = population[['N_POP_CHG_2016', 'POP_ESTIMATE_2016']].apply(
        lambda x: x["N_POP_CHG_2016"] / x['POP_ESTIMATE_2016'], axis=1)
    data['INTERNATIONAL_MIG_2016'] = population[['INTERNATIONAL_MIG_2016', 'POP_ESTIMATE_2016']].apply(
        lambda x: x['INTERNATIONAL_MIG_2016'] / x['POP_ESTIMATE_2016'], axis=1)
    data['DOMESTIC_MIG_2016'] = population[['DOMESTIC_MIG_2016', 'POP_ESTIMATE_2016']].apply(
        lambda x: x['DOMESTIC_MIG_2016'] / x['POP_ESTIMATE_2016'], axis=1)
    data['NET_MIG_2016'] = population[['NET_MIG_2016', 'POP_ESTIMATE_2016']].apply(
        lambda x: x['NET_MIG_2016'] / x['POP_ESTIMATE_2016'], axis=1)
    data['GQ_ESTIMATES_2016'] = population[['GQ_ESTIMATES_2016', 'POP_ESTIMATE_2016']].apply(
        lambda x: x['GQ_ESTIMATES_2016'] / x['POP_ESTIMATE_2016'], axis=1)

    unemployment_tem = pd.DataFrame(unemployment,
                                    columns=['fips_code', 'Civilian_labor_force_2012', 'Civilian_labor_force_2016',
                                             'Unemployment_rate_2012', 'Unemployment_rate_2016', 'Employed_2016',
                                             'Urban_influence_code_2013'])
    data = pd.merge(data, unemployment_tem, how='left', on='fips_code')
    data['Unemployment_rate_2012'] = data['Unemployment_rate_2012'].map(lambda x: x / 100)
    data['Unemployment_rate_2016'] = data['Unemployment_rate_2016'].map(lambda x: x / 100)
    data['Employed_2016'] = data[['Employed_2016', 'Civilian_labor_force_2016']].apply(lambda x: x['Employed_2016'] / x['Civilian_labor_force_2016'], axis=1)

    data.to_csv(input_path + 'project_data.csv', encoding='utf-8', index=False)


def map_data():
    input_path = 'C:/Users/yinan/Desktop/gnn-residual-correlation-master/datasets/election/'

    map = pd.read_csv(input_path + 'data.csv')


    election = pd.read_csv(input_path + 'election.csv')

    data = pd.DataFrame(election, columns=['fips_code', 'county'])
    data["result"] = election[["dem_2016", "gop_2016"]] .apply(lambda x: int(x["dem_2016"] > x["gop_2016"]), axis=1)
    data.sort_values("fips_code", inplace=True)
    data.rename(columns={'fips_code': 'county_fips'}, inplace=True)

    data = pd.merge(map, data, how='left', on='county_fips')
    data.to_csv(input_path + 'map.csv', encoding='utf-8', index=False)


def pca():

    input_path = 'C:/Users/yinan/Desktop/gnn-residual-correlation-master/datasets/election/'
    data = pd.read_csv(input_path + 'project_data.csv')


    x = data.iloc[:,4:]
    y = data[["gop_2016" , "dem_2016"]].apply(lambda x: int(x["gop_2016"] < x["dem_2016"]), axis=1)

    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    # x = np.arange(pca.singular_values_.shape[0]) + 1
    # sum_value = np.sum(pca.singular_values_)
    # y = sorted(pca.singular_values_, reverse=True) / sum_value
    # plt.plot(x, y)
    # plt.show()

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])
    label = pd.DataFrame(data=y
                         , columns=['target'])
    finalDf = pd.concat([principalDf, label], axis=1)
    print(finalDf)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)

    targets = [0, 1]
    colors = ['r', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)

    ax.legend(targets)
    ax.grid()
    plt.show()


def rf():
    input_path = 'C:/Users/yinan/Desktop/gnn-residual-correlation-master/datasets/election/'
    data = pd.read_csv(input_path + 'project_data.csv')
    k = 4
    x = data.iloc[:, k:]
    y = data[["gop_2016", "dem_2016"]].apply(lambda x: int(x["gop_2016"] < x["dem_2016"]), axis=1)

    x = StandardScaler().fit_transform(x)
    rf = RandomForestRegressor()
    rf.fit(x, y)

    print(rf.score(x,y))

    x = data.iloc[:, k:]
    l = []
    for i, c in zip(rf.feature_importances_, list(x.columns)):
        l.append((c,i))
    l.sort(key=lambda x: x[1],reverse=True)

    for i in l:
        print("{}: {:.4f}".format(i[0],i[1]))


if __name__ == '__main__':
    build_data()

























