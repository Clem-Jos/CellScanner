import pandas as pd
import numpy as np
import math
# from scipy import stats
import random as rand
import fcsparser
import os
from datetime import datetime
import sys
import statistics as stat
import matplotlib.pyplot as plt
import warnings
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from matplotlib.colors import Normalize
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QMessageBox

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedShuffleSplit  # cross validation
from sklearn.model_selection import GridSearchCV  # randomforest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # Needed
from sklearn.semi_supervised import LabelPropagation
import matplotlib.gridspec as gridspec
import runScript as r
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels


def visualize(filename, axes=['FL1-A', 'FL3-A'], cd=''):
    txt = filename.copy()
    for i in range(len(filename)):
        filename[i] = cd + filename[i]
    arrays = r.importFile(filename, gating=None)
    x = axes[0]
    y = axes[1]
    for i in range(len(arrays)):
        A = arrays[i].loc[:, [x, y]].copy()
        A.loc[:, x] = np.log10(A.loc[:, x])
        A.loc[:, y] = np.log10(A.loc[:, y])
        plt.figure(figsize=(9, 9))
        plt.scatter(A[x], A[y], s=0.2)
        plt.title(txt[i])
        plt.show()


def addSpeciesTag(narray, species):
    narray.insert(len(narray.columns), 'SPECIES', len(narray) * [species], True)
    return narray


def gatingFunction(narray, save=None, filename='', cwd='', fc='Accuri'):  # graph change FL2 to FL3
    labels = []
    A = narray.loc[:, ['FL1-A', 'FL3-A']].copy()
    A.loc[:, 'FL1-A'] = np.log10(A.loc[:, 'FL1-A'])
    A.loc[:, 'FL3-A'] = np.log10(A.loc[:, 'FL3-A'])

    if 'Accuri' in fc or 'accuri' in fc:
        for row in list(narray.index):

            if math.isnan(np.log10(narray.loc[row, 'FL3-A'])) or math.isnan(np.log10(narray.loc[row, 'FL1-A'])) \
                    or narray.loc[row, 'FL3-A'] > (0.0241 * narray.loc[row, 'FL1-A'] ** 1.0996):
                narray = narray.drop([row])
                labels.append(0)

            elif narray.loc[row, 'FSC-A'] > 100000 and narray.loc[row, 'SSC-A'] > 10000:
                narray = narray.drop([row])
                labels.append(0)

            else:
                labels.append(1)

    elif 'Cytoflex' in fc or 'cytoflex' in fc:

        for row in list(narray.index):
            if math.isnan(np.log10(narray.loc[row, 'FL3-A'])) or math.isnan(np.log10(narray.loc[row, 'FL1-A'])) \
                    or np.log10(narray.loc[row, 'FL3-A']) > (1.5 * np.log10(narray.loc[row, 'FL1-A']) - 2.8) \
                    or np.log10(narray.loc[row, 'FL2-A']) > (2.5 * np.log10(narray.loc[row, 'FL1-A']) - 9):
                narray = narray.drop([row])
                labels.append(0)
            # elif narray.loc[row, 'FSC-A'] > 100000 and narray.loc[row, 'SSC-A'] > 10000:
            #   narray = narray.drop([row])
            #  labels.append(0)
            else:
                labels.append(1)
    if save is not None and save != 'None':
        x = 'FL1-A'
        y = 'FL3-A'
        colours = {0: 'r', 1: 'g'}
        cvec = [colours[label] for label in labels]
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)
        ax.scatter(A[x], A[y], c=cvec)
        ax.set_title('Line gating ' + filename)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if save == 'show':
            plt.show()
        elif save == 'save':
            fig.savefig(cwd + 'gating_' + filename.split('/')[-1][:-4] + '.png')
            plt.close()
        else:
            plt.close()

    return narray


def randomSelection(narray, nbC=1000, random_state=None):
    values = []
    length = len(narray)
    if length < nbC:
        nbC = length
        warnings.warn('The number of selected in smaller than expected ' + str(length) +
                      ' the training and the prediction can be impacted.')
    for i in range(nbC):
        val = rand.randint(0, len(narray.index) - 1)
        while val in values:
            val = rand.randint(0, nbC - 1)
        values.append(val)
        values.sort()
    newArray = narray.iloc[values]
    return newArray


def splitInformation(narray):
    data = narray.iloc[:, 0:-1]
    target = narray.loc[:, 'SPECIES']
    return data, target


def logisticRegression(data, target, ratio, logReg='l2', asolver='lbfgs', random_state=None):
    np.set_printoptions(threshold=np.inf)
    train_img, test_img, train_lbl, test_lbl = train_test_split(data, target, test_size=ratio,
                                                                random_state=random_state)
    scaler = StandardScaler()
    scaler.fit(train_img)
    train_img = scaler.transform(train_img)
    test_data = test_img
    test_img = scaler.transform(test_img)
    logisticRegr = LogisticRegression(penalty=logReg, solver=asolver, max_iter=1000)
    logisticRegr.fit(train_img, train_lbl)
    predict = logisticRegr.predict(test_img[0:])
    logisticRegr.score(test_img, test_lbl)
    predict_lbl = predict.tolist()
    known_lbl = test_lbl.values.tolist()
    return predict_lbl, known_lbl, scaler, logisticRegr, test_data


def neuralNetwork(data, target, ratio, activation='relu', solver='lbfgs', max_iter=1000, random_state=None):
    np.set_printoptions(threshold=np.inf)
    train_img, test_img, train_lbl, test_lbl = train_test_split(data, target, test_size=ratio,
                                                                random_state=random_state)
    test_data = test_img
    scaler = StandardScaler()
    scaler.fit(train_img)
    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img)
    grid = MLPClassifier(solver=solver, hidden_layer_sizes=200, activation=activation, max_iter=max_iter,
                         early_stopping=False,
                         learning_rate='constant')
    grid.fit(train_img, train_lbl)
    predict = grid.predict(test_img[0:])
    predict_lbl = predict.tolist()
    known_lbl = test_lbl.values.tolist()
    return predict_lbl, known_lbl, scaler, grid, test_data


def randomForest(data, target, ratio, n_estimators=200, criterion='gini', random_state=None):
    np.set_printoptions(threshold=np.inf)
    train_img, test_img, train_lbl, test_lbl = train_test_split(data, target, test_size=ratio,
                                                                random_state=random_state)
    test_data = test_img
    scaler = StandardScaler()
    scaler.fit(train_img)
    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img)
    grid = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=random_state)
    grid.fit(train_img, train_lbl)
    # print('BASE ESTIMATOR \n',grid.base_estimator_)
    # print('ESTIMATOR \n',grid.estimators_)
    sp = list(set(target))
    print('FEATURE IMPORTANCE\n', grid.feature_importances_)
    features = [str(i) for i in list(grid.feature_importances_)]
    print(sp)
    print(features)
    imp = sp + features
    print(imp)
    sep = ';'
    f = open('Results/importance.csv', 'a')
    f.write(sep.join(imp) + '\n')
    f.close()
    predict = grid.predict(test_img[0:])
    predict_lbl = predict.tolist()
    known_lbl = test_lbl.values.tolist()
    return predict_lbl, known_lbl, scaler, grid, test_data


def randomGuessing(data, species, ptype='train', target=None, ratio=1 / 7, random_state=None):
    if target is None:
        target = []
    if ptype == 'train':
        train_img, test_img, train_lbl, test_lbl = train_test_split(data, target, test_size=ratio,
                                                                    random_state=random_state)
        lgth = len(test_lbl)
        test_data = test_img
    else:
        lgth = len(data)
        test_data = data
        test_lbl = []
    predict = []
    spNb = len(species)
    for i in range(lgth):
        predict.append(species[rand.randint(0, spNb - 1)])
    return predict, list(test_lbl), test_data


def scalerTest(data, scaler, logisticReg):
    data = scaler.transform(data)
    predict = logisticReg.predict(data[0:])
    predict_lbl = predict.tolist()

    return predict_lbl


def statAnalysis(predicted, known, species):  # TO OPTIMIZE I can directly extract the matrix
    units = ['P', 'N', 'TN', 'TP', 'FN', 'FP', 'TPR', 'TNR', 'PPV', 'ACC', 'F1']
    res = pd.DataFrame(columns=units, index=species + ['unknown'] + ['MEAN'])
    for s in species + ['unknown']:
        p = 0
        n = 0
        tn = 0
        tp = 0
        fn = 0
        fp = 0

        for pr, k in zip(predicted, known):
            if k == s:
                p = p + 1
                if pr == s:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                n = n + 1
                if pr == s:
                    fp = fp + 1
                else:
                    tn = tn + 1
        if p == 0:
            tpr = 1
        else:
            tpr = tp / p
        if n == 0:
            tnr = 1
        else:
            tnr = tn / n
        if tp + fp == 0:
            ppv = 1
        else:
            ppv = tp / (tp + fp)
        if p + n == 0:
            if tp + tn == 0:
                acc = 1
            else:
                acc = 0
        else:
            acc = (tp + tn) / (p + n)
        if (2 * tp + fp + fn) == 0:
            f1 = 0
        else:
            f1 = (2 * tp) / (2 * tp + fp + fn)
        res.loc[s, 'P'] = p
        res.loc[s, 'N'] = n
        res.loc[s, 'TN'] = tn
        res.loc[s, 'TP'] = tp
        res.loc[s, 'FN'] = fn
        res.loc[s, 'FP'] = fp
        res.loc[s, 'TPR'] = tpr
        res.loc[s, 'TNR'] = tnr
        res.loc[s, 'PPV'] = ppv
        res.loc[s, 'ACC'] = acc
        res.loc[s, 'F1'] = f1
    res.loc['MEAN'] = res.mean(axis=0)
    res.loc['MEAN', 'ACC'] = accuracy_score(known, predicted)
    res.loc['MEAN', 'F1'] = f1_score(known, predicted, average='weighted')
    return res


def exportPrediction(predictedLbl, samples, cwd, typeF, run, typeP='AVERAGE', repeat=0):
    l = '-'
    file = cwd

    if typeF == 'align':
        file = file + str(run) + l.join(samples) + '.csv'
    else:
        file = file + 'predictions.csv'

    f = open(file, 'a')
    sps = set(predictedLbl)
    print('hahahaha')
    species = sorted(sps)
    print('bbbbb')
    print(species)
    f.write(l.join(samples) + '\n')
    if typeP == 'AVERAGE':
        f.write('AVERAGE PREDICTION FOR EACH RECORD WITH %i REPEAT\n' % repeat)
        for spc in species:
            f.write('average;' + str(spc) + ';' + str(predictedLbl.count(spc)) + '\n')
        f.write('average;TOTAL:;' + str(len(predictedLbl)) + '\n')
    else:
        f.write('SINGLE PREDICTION\n')
        for spc in species:
            f.write('single;' + str(spc) + ';' + str(list(predictedLbl).count(spc)) + '\n')
        f.write('single;TOTAL:;' + str(len(predictedLbl)) + '\n')
        f.write("\n")
    f.close()


def exportStatistics(statistics, samples, cwd, typeF):
    l = '-'
    file = cwd
    if typeF == 'align':
        file = file + 'Learning_Stat.csv'
    else:
        file = file + 'Pred_Stat.csv'
    f = open(file, 'a')
    f.write(l.join(samples))
    f.close()
    statistics.to_csv(file, sep=';', mode='a')
    return None


def constructStatDF(nbB):
    header = ['NB_SPECIES', 'SPECIES', 'SAMPLES', 'TRAINING']
    for i in range(nbB):
        header.append('ACC_' + str(i + 1))
        header.append('STDEV_ACC_' + str(i + 1))
        header.append('F1_' + str(i + 1))
        header.append('SDTEV_F1_' + str(i + 1))
    header.append('GLOBAL_ACC')
    header.append('GLOBAL_STDEV_ACC')
    header.append('GLOBAL_F1')
    header.append('GLOBAL_SDTEV_F1')
    df = pd.DataFrame(columns=header)
    return df, header


def constructRatioDF(nbB, species):
    header = ['NB_SPECIES', 'SPECIES', 'SAMPLES', 'AVERAGE/SINGLE']
    for i in range(nbB + 1):
        header.append('Predicted_' + species[i])
        header.append('St_Dev_Predicted_' + species[i])
    header.append('TotNumber')
    df = pd.DataFrame(columns=header)
    return df, header


def meanPrediction(data, samples, SP):
    column = data.iloc[:, 0].unique()
    column = column.tolist()
    column = [x for x in column if x == x]
    values1 = []
    values2 = []
    link = '_'
    # case average
    if 'average' in data.index:
        average = data.loc['average']
        values1.append(len(column) - 1)
        species = list(set(list(average.index)))
        species.sort()
        values1.append(link.join(SP))
        values1.append(samples)
        values1.append('average')

        for spc in species:
            if spc != 'TOTAL:':
                values1.append(float(average.loc[spc].mean()))
                values1.append(float(average.loc[spc].std()))
        values1.append(float(average.loc['TOTAL:'].mean()))

    if 'single' in data.index:
        single = data.loc['single']
        species = list(set(list(single.index)))
        species.sort()
        values2.append(len(column) - 1)
        values2.append(link.join(SP))
        values2.append(samples)
        values2.append('single')
        for spc in SP:
            if spc == 'unknown':
                values2.append(0)
                values2.append(0)
            elif spc != 'TOTAL:':
                values2.append(float(single.loc[spc].mean()))
                values2.append(float(single.loc[spc].std()))
        values2.append(float(single.loc['TOTAL:'].mean()))
    return values1, values2


def calculateStatDataFrame(data, samples, align):
    data.rename(columns={'Unnamed: 0': 'species'}, inplace=True)
    column = data.iloc[:, 0].unique()
    column = column.tolist()
    column = [x for x in column if x == x and x != 'unknown']

    values = []
    link = '_'
    values.append(len(column) - 1)
    values.append(link.join(column[:-1]))
    values.append(samples)
    values.append(align)
    for s in column:
        df = data[data.species.isin([s])]
        a = df.loc[:, 'ACC'].values.tolist()
        for i in range(len(a)):
            a[i] = float(a[i])
        acc = stat.mean(a)
        values.append(acc)
        if len(a) > 1:
            stdacc = stat.stdev(a)
        else:
            stdacc = 0
        values.append(stdacc)
        f = df.loc[:, 'F1'].values.tolist()
        for i in range(len(f)):
            f[i] = float(f[i])
        f1 = stat.mean(f)
        values.append(f1)
        if len(a) > 1:
            stdF1 = stat.stdev(f)
        else:
            stdF1 = 0
        values.append(stdF1)
    return values


def unique(list1):
    x = np.array(list1)
    b = np.unique(x)
    c = b.tolist()
    return c


def graph3d(data, predict, target, species, param=['FL1-A', 'FL3-A', 'FSC-A'], statistics=None, show='show', cwd='',
            repeat=1, name='', predtype='analysis',
            clust=False):
    posParam = [data.columns.get_loc(param[0]), data.columns.get_loc(param[1]), data.columns.get_loc(param[2])]
    if statistics is not None:
        f1 = statistics.loc['MEAN', 'F1']
        acc = statistics.loc['MEAN', 'ACC']
    else:
        f1 = 0
        acc = 0
    fig = plt.figure(figsize=(13, 10))
    G = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(G[0, 0], projection='3d')
    ax2 = fig.add_subplot(G[0, 1], projection='3d')
    for sp in species:
        posk = []
        posp = []
        xk = []
        yk = []
        zk = []
        xp = []
        yp = []
        zp = []

        for i in range(len(target)):
            if clust:
                if target[i] == sp:
                    posk.append(i)
                    xk.append(data.iloc[i, posParam[0]])
                    yk.append(data.iloc[i, posParam[1]])
                    zk.append(data.iloc[i, posParam[2]])
                if predict[i] == sp:
                    posp.append(i)
                    xp.append(data.iloc[i, posParam[0]])
                    yp.append(data.iloc[i, posParam[1]])
                    zp.append(data.iloc[i, posParam[2]])

            else:
                if target[i] == sp:
                    posk.append(i)
                    xk.append(np.log10(data.iloc[i, posParam[0]]))
                    yk.append(np.log10(data.iloc[i, posParam[1]]))
                    zk.append(np.log10(data.iloc[i, posParam[2]]))
                if predict[i] == sp:
                    posp.append(i)
                    xp.append(np.log10(data.iloc[i, posParam[0]]))
                    yp.append(np.log10(data.iloc[i, posParam[1]]))
                    zp.append(np.log10(data.iloc[i, posParam[2]]))
        ax1.scatter(xk, yk, zk, label=sp, s=0.5)
        ax2.scatter(xp, yp, zp, label=sp, s=0.5)
    ax1.set_title('Expected Result')
    ax2.set_title('Predicted result')
    ax1.set_xlabel(param[0])
    ax1.set_ylabel(param[1])
    ax1.set_zlabel(param[2])
    ax1.legend()
    ax2.set_xlabel(param[0])
    ax2.set_ylabel(param[1])
    ax2.set_zlabel(param[2])
    if predtype == 'analysis' or predtype == 'training':
        fig.suptitle('Species prediction for \n' + name + '\nACC: %.2f\nF1: %.2f' % (acc, f1), size=15)
    else:
        fig.suptitle('Species prediction for \n' + name, size=15)
    ax2.legend()
    if show == 'show':
        plt.show()
    elif show == 'save':
        if predtype == 'analysis' or predtype == 'prediction':
            plt.savefig(cwd + 'graph_tool_analysis_' + name + '.png')
        elif predtype == 'prediction':
            plt.savefig(cwd + 'prediction_graph_' + str(repeat) + '.png')
        elif predtype == 'training':
            plt.savefig(cwd + 'training_graph_' + name + '.png')
        elif predtype == 'clustering':
            plt.savefig(cwd + 'Clustering_graph' + name + '.png')
        plt.close()
    return data


def graph3dRef(data, predict, target, species, param, statistics, show, cwd, refdata, reflabel, repeat=1, name='',
               predtype='analysis', clust=False):
    posParam = [data.columns.get_loc(param[0]), data.columns.get_loc(param[1]), data.columns.get_loc(param[2])]
    # print(posParam)
    f1 = statistics.loc['MEAN', 'F1']
    acc = statistics.loc['MEAN', 'ACC']
    fig = plt.figure(figsize=(13, 10))
    G = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(G[1, 0], projection='3d')
    ax2 = fig.add_subplot(G[1, 1], projection='3d')
    ax3 = fig.add_subplot(G[0, 1], projection='3d')
    for sp in species:
        posk = []
        posp = []
        xk = []
        yk = []
        zk = []
        xp = []
        yp = []
        zp = []
        xr = []
        yr = []
        zr = []
        for i in range(len(reflabel)):
            if reflabel[i] == sp:
                xr.append(refdata.iloc[i, posParam[0]])
                yr.append(refdata.iloc[i, posParam[1]])
                zr.append(refdata.iloc[i, posParam[2]])

        for i in range(len(target)):
            if clust:
                if target[i] == sp:
                    posk.append(i)
                    xk.append(data.iloc[i, posParam[0]])
                    yk.append(data.iloc[i, posParam[1]])
                    zk.append(data.iloc[i, posParam[2]])
                if predict[i] == sp:
                    posp.append(i)
                    xp.append(data.iloc[i, posParam[0]])
                    yp.append(data.iloc[i, posParam[1]])
                    zp.append(data.iloc[i, posParam[2]])

            else:
                if target[i] == sp:
                    posk.append(i)
                    xk.append(np.log10(data.iloc[i, posParam[0]]))
                    yk.append(np.log10(data.iloc[i, posParam[1]]))
                    zk.append(np.log10(data.iloc[i, posParam[2]]))
                if predict[i] == sp:
                    posp.append(i)
                    xp.append(np.log10(data.iloc[i, posParam[0]]))
                    yp.append(np.log10(data.iloc[i, posParam[1]]))
                    zp.append(np.log10(data.iloc[i, posParam[2]]))

        ax1.scatter(xk, yk, zk, label=sp)
        ax2.scatter(xp, yp, zp, label=sp)
        ax3.scatter(xr, yr, zr, label=sp)
    ax1.set_title('Clusters')
    ax2.set_title('Identified')
    ax3.set_title('References')
    ax1.set_xlabel(param[0])
    ax1.set_ylabel(param[1])
    ax1.set_zlabel(param[2])
    ax2.legend()
    ax2.set_xlabel(param[0])
    ax2.set_ylabel(param[1])
    ax2.set_zlabel(param[2])
    ax1.legend()
    ax3.set_xlabel(param[0])
    ax3.set_ylabel(param[1])
    ax3.set_zlabel(param[2])
    ax3.legend()
    if predtype == 'analysis' or predtype == 'training':
        fig.suptitle('Species prediction for \n' + name + '\nACC: %.2f\nF1: %.2f' % (acc, f1), size=15)
    else:
        fig.suptitle('Species prediction for \n' + name, size=15)
    # print(show)
    if show == 'show':
        plt.show()
    elif show == 'save':

        if predtype == 'analysis' or predtype == 'prediction':
            plt.savefig(cwd + 'graph_tool_analysis_' + name + '.png')
        elif predtype == 'prediction':
            plt.savefig(cwd + 'prediction_graph_' + str(repeat) + '.png')
        elif predtype == 'training':
            plt.savefig(cwd + 'training_graph_' + name + '.png')
        elif predtype == 'clustering':
            plt.savefig(cwd + 'Clustering_graph' + name + '.png')
        plt.close()
    return data


def mergeSameSpecies(arrays, species):
    newSp = list(set(species))
    newSp.sort()
    newArrays = []
    for i in newSp:
        j = species.count(i)
        indexes = [k for k, x in enumerate(species) if x == i]
        l = 1
        df = arrays[indexes[0]]
        while l != j:
            df = df.merge(arrays[indexes[l]], how='outer')
            l = l + 1
        newArrays.append(df)
    return newArrays, newSp


def assessmentValue(statValues, species, cwd, sample, typeF):
    ACC = []
    F1 = []
    SENS = []
    PREC = []
    for df in statValues:
        ACC.append(df.loc['MEAN', 'ACC'])
        F1.append(df.loc['MEAN', 'F1'])
        sens = []
        prec = []

        for sp in species:
            if df.loc[sp, 'TP'] != 0:
                sens.append(df.loc[sp, 'TP'] / (df.loc[sp, 'TP'] + df.loc[sp, 'FN']))
                prec.append(df.loc[sp, 'TP'] / (df.loc[sp, 'TP'] + df.loc[sp, 'FP']))
            else:
                sens.append('None')
                prec.append('None')
        SENS.append(sens)
        PREC.append(prec)

    SENS = np.array(SENS).T.tolist()
    PREC = np.array(PREC).T.tolist()
    acc = stat.mean(ACC)
    f1 = stat.mean(F1)
    if len(ACC) > 1:
        stdAcc = stat.stdev(ACC)
        stdf1 = stat.stdev(F1)
    else:
        stdAcc = 'None'
        stdf1 = 'None'
    sens = []
    stdSens = []
    prec = []
    stdPrec = []
    for i in range(len(species)):
        if len(SENS[i]) > 1:
            if 'None' in SENS[i]:  # if not enough value for a species
                sens.append('None')
                prec.append('None')
                stdSens.append('None')
                stdPrec.append('None')
            else:
                sens.append(stat.mean([float(x) for x in SENS[i]]))
                prec.append(stat.mean([float(x) for x in PREC[i]]))
                stdSens.append(stat.stdev([float(x) for x in SENS[i]]))
                stdPrec.append(stat.stdev([float(x) for x in PREC[i]]))
        else:
            sens.append(SENS[i][0])
            prec.append(PREC[i][0])
            stdSens.append('None')
            stdPrec.append('None')

    if typeF == 'align':
        file = cwd + 'Reference.csv'
        f = open(file, 'a')
        f.write('Reference information\n')
    else:
        file = cwd + 'assessment.csv'
        f = open(file, 'a')
        f.write('Assessment for the merged files : \n')
    for i in range(len(sample)):
        f.write(sample[i] + '\n')
    f.write('Accuracy;' + str(acc) + ';Standard deviation;' + str(stdAcc) + '\n')
    f.write('F1;' + str(f1) + ';Standard deviation;' + str(stdf1) + '\n')
    for i in range(len(species)):
        f.write(str(species[i]) + ';Sensitivity;' + str(sens[i]) + ';Standard deviation;' + str(
            stdSens[i]) + ';Precision;' + str(prec[i]) + ';Standard deviation;' + str(stdPrec[i]) + '\n\n')


def fileOption(cwd, files, species, files2, species2, nbC, nbC2, gating='line', predAn='prediction',
               predtype='neur', ratio=1 / 7.0, repeat=1, average=True,
               doubt=0, channels=[], dicChannels={},fc='Accuri',type=''):
    f = open(cwd + 'option.txt', 'a')
    now = datetime.now()
    date = now.strftime("%d/%m/%Y %H:%M")
    if predAn == 'prediction':
        f.write(type+'Prediction ' + date + '\n')
    else:
        f.write(type+'Tool analysis ' + date + '\n')
    f.write('Class records:\n')
    for sp in set(species):
        f.write('\t- ' + sp + '\n')
    f.write('\n References:\n')
    for ref in files:
        f.write('\t->' + ref + '\n')
    if predAn == 'prediction':
        f.write('Files for prediction :\n')
        for pred in files2:
            f.write('\t -' + pred + '\n')
    else:
        f.write('Merged file for tool analysis:\n')
        for file, sp in zip(files2, species2):
            f.write('\t- ' + file + ' ->' + sp + '\n')
    f.write('\Flow cytometer:  '+fc)
    f.write('\nSelected channels:\n')
    f.write('\t' + ', '.join(channels) + '\n')
    f.write('Reference channels:\n')
    for k, v in dicChannels.items():
        f.write('\t-' + k + ' <- ' + str(v) + '\n')
    if predtype == 'neur':
        predictionT = 'Neural network'
    elif predtype == 'log':
        predictionT = 'Logistic regression'
    elif predtype == 'rand':
        predictionT = 'Random forest'
    else:
        predictionT = 'Random guess'
    f.write('\nTraining:\nNumber of cell per records: ' + str(nbC) + '\nRatio Test/Training: ' + str(ratio) + '\n')
    if predtype == 'analysis':
        f.write(
            '\nRunning:\nNumber of repeat: ' + str(repeat) + '\nNumber of event per species: ' + str(nbC2) +
            '\nType of gating: ' + str(gating) + '\nPrediction type:' + predictionT + '\n')
    else:
        f.write('\nRunning:\nNumber of repeat: ' + str(repeat) + '\nType of gating: ' + str(gating) +
                '\nPrediction type:' + predictionT + '\n')

    f.write('Calculate an average prediction with the repeated results: ' + str(average) +
            '\nMinimum normalized prediction on repeats: ' + str(doubt))

    f.write('\n\nCellScanner version: 1.00\nWriten by Clemence JOSEPH\n27/11/2020')
    f.close()


def createCombination(init_condition, nbComb):
    """
    Function creating a ist of species combination function used to run species pairwise
    Can be use alone
    :param init_condition: List compound to use in combination
    :param nbComb:
    :return:
    """
    combination = []
    for i in range(len(init_condition)):
        combination.append([init_condition[i]])
    while len(combination[-1]) < nbComb:
        comb2 = []
        for a in combination:
            for b in init_condition:
                comb2.append(a + [b])
        combination = deleteBis(comb2)
    return combination


def deleteBis(valbis):
    """
    Function linked to create combination, delete the dubbed data.
    :param valbis:
    :return:
    """
    toDel = []
    for i in range(len(valbis)):
        valbis[i].sort()
        for j in range(len(valbis[i])):
            if valbis[i][j] in valbis[i][j + 1:]:
                toDel.append(i)
    if toDel:
        toDel.sort(reverse=True)
    for i in toDel:
        del (valbis[i])
    toDel = []
    for i in range(len(valbis)):
        if valbis[i] in valbis[i + 1:]:
            toDel.append(i)
    if toDel:
        toDel.sort(reverse=True)
    for i in toDel:
        del (valbis[i])
    return valbis


def position(valuesL):
    ordered = []
    for l in valuesL:
        m = l.copy()
        m.sort(reverse=True)
        lgt = len(l)
        p = [0] * lgt
        lgt = lgt - 1
        for a in m:
            p[l.index(a)] = lgt
            lgt = lgt - 1
        ordered.append(p)
    return ordered


def isBetter(clusterSP, dicSP, spp, dsp, var=0.05):
    if clusterSP == [] and dicSP == {}:
        return True
    elif len(dsp) < len(dicSP):
        return False
    elif len(dsp) > len(dicSP):
        return True
    else:
        kf = list(dicSP.keys())
        kn = list(dsp.keys())
        kf.sort()
        kn.sort()
        if kf == kn:  # si les espces identifiées sont les mêmes
            for i in range(len(kf)):
                print(dsp[kf[i]], ':', dicSP[kf[i]] + dicSP[kf[i]] * var)
                if dsp[kf[i]] > (dicSP[kf[i]] + dicSP[kf[i]] * var):  # Si la distance est plus elevée on annule
                    print(20)
                    return False
            return True
        print('found different cluster even same number, find an alternative in this case considered best solution')
        return True


def distance(pav, rav, column=['FSC-A', 'FL1-A', 'FL3-A']):
    dist = []
    for i in range(len(pav)):
        T = []
        if column is None or column == []:
            for j in range(len(rav)):
                T = T + [np.nansum([abs(ele) ** 2 for ele in list(pav.iloc[i] - rav.iloc[j])])]
        else:
            posParam = []
            for a in column:
                posParam.append(pav.columns.get_loc(a))
            for j in range(len(rav)):
                T = T + [np.nansum([abs(ele) ** 2 for ele in list(pav.iloc[i, posParam] - rav.iloc[j, posParam])])]
        dist.append(T)

    return dist


def cmFile(cm, species, cwd, what):

    f = open(cwd + 'cmData.csv', 'a')
    species=[str(i) for i in species]

    f.write(what + ':;\n')
    l = len(species)
    table = []
    table.append(['', ''] + ['P'] * l)
    table.append(['', ''] + species)
    for i in range(l):
        ssTable = ['E'] + [str(species[i])]
        for j in range(l):
            ssTable = ssTable + [str(cm[i][j])]
        table.append(ssTable)
    for line in table:
        f.write(';'.join(line) + '\n')
    f.write('\n Normalized:\n')
    cm2 = cm.copy()
    f.close()
    f=open(cwd + 'cmData.csv', 'a')
    for i in range(l):
        tot = sum(cm[i])
        for j in range(l):
            if tot > 0:
                cm2[i][j] = (cm[i][j] * 100) / tot
            else:
                cm2[i][j] = 0
    table2 = []
    table2.append(['', ''] + ['P'] * l)
    table2.append(['', ''] + species)
    for i in range(l):
        ssTable = ['E'] + [str(species[i])]
        for j in range(l):
            ssTable = ssTable + [str(cm2[i][j])]
        table2.append(ssTable)
    for line in table2:
        f.write(';'.join(line) + '\n')
    f.close()


def getSorted(aSp, arrays, species):
    """
    This function select in an arrays the arrays from the indicated species, via a list of species.
    :param aSp:
    :param arrays:
    :param species:
    :return:
    """

    narrays = []
    for i in range(len(species)):
        if species[i] == aSp:
            narrays.append(arrays[i])
    if len(narrays) > 1:
        narray = pd.concat(narrays, ignore_index=True, sort=False)
    else:
        narray = narrays[0]
    return narray


def gate(label, array):
    """
    This function remove the datapoint from an array when they are labelled 'blank" or 'unknown'. The labels are
    in a list.
    :param label: list of label for each data point in the array
    :param array: array containing flowcytometric data point
    :return: narray is the array rid of blank and unknown data point.
    """
    narray = array.copy()
    for i in range(len(array) - 1, -1, -1):
        if label[i] == 'blank' or label[i] == 'unknown':
            narray = narray.drop([i])
    return narray


def machineGating(refArrays, species, predArrays, species2, predType='neur', ratio=1 / 2, random_state=None,
                  show='save',
                  cwd='', rept=10, name='ref'):
    """
    THis function gate the flow cytometric data, by using several machine learning predictions. From Blank values, and
    cells values (separated in monocultures), the programm will learn to reconize 2 groups with the reference data
    (RefArrays). A prediction of each monoculture dataarray, will indicate, which lie is from line or from blanc.
    In fact, if the cells monoculture contains blank values, the prediction model will hesitate between blanc and
    species. By performing several run, an comparing the prediction, we can name unknown the value between the two class.
    the values labelled unknown and the values labelled blank will be removed from the array.

    :param refArrays: List of arrays containing data to use to train the predictive model.
    :param species: List of species matching the arrays in refArrays
    :param predArrays: List of arrays containing data to gate.
    :param species2: List of species matching the arrays in predArrays
    :param predType: indicate the algorithm used for the prediction. 'neur','rand','log'
    :param ratio: ratio of learning set vs test set for the learning program; usually 1/7
    :param random_state:
    :param show: if 'save' figures are saved in the indicated directory, if 'show' figures will appear on the screen. if
        'None', no figures is produced
    :param cwd: path where the figures are saved
    :param rept: number of prediction per species
    :param name: name used to save the figures
    :return: NewArray is a list of array obtained from predArrays, where each array contain a monoculture data (which
    contain fusion of th same species, if several array from the same species where given) without blanc values. newSp
    is the list of species matching the arrays.
    """
    blk = ['BLANK', 'Blank', 'blank']
    if 'Blank' in species or 'blank' in species or 'BLANK' in species:
        blank = []
        other = []
        sp = []
        NewArray = []
        newSp = []
        sp2 = []
        pOther = []
        for i in range(len(refArrays)):
            if species[i] in blk:
                blank.append(refArrays[i])
            else:
                other.append(refArrays[i])
                sp.append(species[i])
        for i in range(len(species2)):
            if species2[i] not in blk:
                sp2.append(species2[i])
                pOther.append(predArrays[i])
        for aSp in list(set(sp2)):
            predictions = []
            data2, target2, species3 = treat([getSorted(aSp, pOther, sp2)], [aSp], None, mode='analysis')
            aSpArray = getSorted(aSp, other, sp)
            for i in range(rept):
                data, target, species = treat(blank.copy() + [aSpArray], ['blank'] * len(blank) + [aSp], 5000,
                                              mode='train')
                scaler, classifier, predict_lbl, known_lbl, predict_data = learning(predType, data, target, ratio,
                                                                                    random_state, species)
                predict_lbl2 = predict(predType, scaler, classifier, data2, species)
                predictions.append(predict_lbl2)
            prediction = bestPred(predictions, 0.7)
            nName = 'gating_' + name + '_' + aSp
            graph3d(data2, prediction, target2, species + ['unknown'], name=nName, show=show, cwd=cwd)
            NewArray.append(gate(prediction, data2))
            newSp.append(aSp)

        return NewArray, newSp
    else:
        return predArrays, species2


def importFile(files, gating='line', save='None', fc='Accuri', cwd='', channels=[], dicChannels={},sep =','):
    """This function import fcs or csv files from there name and convert theme into arrays according to 'fc' either
    Cytoflex or Accuri. The function also select the wanted column. if all column has to be kept, hide the line #'data=data[[]]'#
    :param files: list of filenames (=path is different of current directory)
    :param gating: type of gating: line, none, machine? TODO
    :param save: parameter associated to the gating graph. If 'show', will appear to the screen, if 'save',
        will be save in the dedicated directory, if 'None' no graph is produced
    :param fc: type of flow cytometry machine used, 'Cytoflex' or 'Accuri'
    :param cwd: Result directory path (where gating graphs have to be saved)
    :return: data arrays (one per file) gated according to the parameters
    """
    arrays = []
    for i in range(len(files)):
        if files[i][-3:] == 'csv':
            data = pd.read_csv(files[i], header=0, sep=sep)

        elif files[i][-3:] == 'fcs':
            meta, data = fcsparser.parse(files[i], reformat_meta=True)
        else:
            data = []
        data = data.rename(columns=dicChannels)
        for a in channels:
            if a not in data.columns:
                popup('Wrong flow cytometer',
                      'The flow cytometer is not compatible with the selected data.\n\nPlease check that the selected files are from the selected flow cytometer.')
                return []
        data = data[channels]  # hide this line if all columns have to be used in the program
        arrays.append(data)
    for i in range(len(arrays)):
        if gating is not None:
            if gating == 'line':
                arrays[i] = gatingFunction(arrays[i], save=save, filename=files[i],cwd=cwd, fc=fc)
    return arrays


def treat(someArrays, species, nbC, mode='pred', cluster=False, random_state=None):
    """
    Import function into data frames, gate the data and select limited number of data for reference files.
    The function also convert and select the target channels.
    :param random_state:
    :param cluster: Boolean, if the programm use clustering instead of classification
    :param someArrays: Flow cytometric data arrays which have to be treated
    :param species: list of species linked to reference files, or prediction files ('unknown' if not indicated)
    :param nbC: number of cell per species from reference files for Learning and training set of the model
    :param mode: indicate if extracted files are for reference or for prediction.
    :return: return the imported dataset, the expected label and the list of species implied on the file ('unknown'
    if not known)
    """

    arrays = []
    fusion = []
    for anArray in someArrays:

        arrays.append(anArray.copy())
    if mode == 'train' or mode == 'analysis':
        arrays, species = mergeSameSpecies(arrays, species)
    for i in range(len(arrays)):
        if nbC is not None:
            arrays[i] = randomSelection(arrays[i], nbC, random_state=random_state)
        if cluster:
            arrays[i] = arrays[i].dropna()

            #arrays[i] = arrays[i].drop(columns=['FL4-A', 'FL4-H'])
            arrays[i] = arrays[i].dropna()
            for j in range(0, len(arrays[i])):
                for k in range(len(arrays[i].columns)):
                    arrays[i].iloc[j, k] = np.log10(arrays[i].iloc[j, k])
            arrays[i] = arrays[i].replace([np.inf, -np.inf], np.nan)
            arrays[i] = arrays[i].dropna()
        arrays[i] = addSpeciesTag(arrays[i], species[i])  # TODO check if array len =1 works in this case

        if len(arrays) > 1:
            fusion = pd.concat(arrays, ignore_index=True, sort=False)
        else:
            fusion = arrays[0]
    data, target = splitInformation(fusion)
    return data, target, species


def learning(predType, data, target, ratio, random_state, species=None):
    """
    This function create the machine learning model from references datasets. The ratio indicate the ratio of dataset
    for Learning/Training set.
    :param predType:
    :param data:
    :param target:
    :param ratio:
    :param random_state:
    :param species:
    :return:    It return predicted and expected data of the training test, the scaler and classifier to proceed new
                prediction and the predicted data usable for graph.
    """
    if species is None:
        species = []
    if predType == 'log':
        predict_lbl, known_lbl, scaler, classifier, predict_data = logisticRegression(data, target, ratio,
                                                                                      random_state=random_state)
    elif predType == 'neur':
        predict_lbl, known_lbl, scaler, classifier, predict_data = neuralNetwork(data, target, ratio,
                                                                                 random_state=random_state)
    elif predType == 'rand':
        predict_lbl, known_lbl, scaler, classifier, predict_data = randomForest(data, target, ratio,
                                                                                random_state=random_state)
    else:  # predType == 'rdguess':
        predict_lbl, known_lbl, predict_data = randomGuessing(data, species, 'train', target, ratio, random_state)
        scaler = ''
        classifier = ''
    return scaler, classifier, predict_lbl, known_lbl, predict_data


def predict(predtype, scaler, classifier, data, species):
    """
    This function run a prediction from the scaler and classifier prduced by learning function.
    :param predtype:    Type of machine learning calculation ('rand': random forest,'neur':neural network,
                        'log': logistic regression,'rdguess': random guessing)
    :param scaler:      Scaler object produced by learning function
    :param classifier:  Classifier object produced by learning function
    :param data:        a dataset for prediction with same column as dataset used as references.
    :param species:     List of species used in references files.
    :return:            Prediction label list for the input dataset.
    """
    if predtype == 'rdguess':
        predict_lbl, g, h = randomGuessing(data, species, 'predict')
    else:
        if len(data) == 0:
            predict_lbl = []
        else:
            predict_lbl = scalerTest(data, scaler, classifier)
    return predict_lbl


def averageConfM(ConfMList):
    """
    This function create an average confusion matrix from the different runs.
    :param ConfMList: List of confusion matrix (from different runs)
    :return: An average array of confusion matrix
    """
    nb = len(ConfMList[0])
    nr = len(ConfMList)
    new = []
    print(ConfMList)
    for i in range(nb):
        l = []
        for j in range(nb):
            n = 0
            for k in range(len(ConfMList)):
                n = n + ConfMList[k][i][j]
            n = n / nr
            l.append(n)
        new.append(l)
    return np.array(new)


def bestPred(predict_lbls, doubt=0):
    """
    This function create an average prediction for the different runs. For each cell the program compare the predicted
    species. If the most predicted species is less predicted than the minimum ratio given by doubt, unknown will be set
    as the predicted species.

    :param predict_lbls: list of list of species predicted from data file. All list compound are get from predict
                         function
    :param doubt: ratio minimum of the best most predicted species to be tagged with it. If not cell is tagged unknown.
    :return: return a list wit predicted species names
    """
    predict_lbl = []
    repeat = len(predict_lbls)
    repeatm = repeat * doubt
    if repeat > 1:
        for i in range(len(predict_lbls[0])):
            l = []
            for j in range(repeat):
                l.append(predict_lbls[j][i])
            v = max(set(l), key=l.count)
            if l.count(v) > repeatm:
                predict_lbl.append(v)
            else:
                predict_lbl.append('unknown')
        return predict_lbl
    else:
        return predict_lbls[0]


def plotConfusionMatrix(cm, classes, save, cwd, name='', normalize=False,
                        title=None, cmap=plt.cm.Blues, predAn='prediction'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized ' + name
        else:
            title = name

    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]), ylim=(-0.5, 1 * len(classes) - 0.5),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    cm.astype(float)
    thresh = np.nanmax(cm) / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save == 'show':
        plt.show()
    elif save == 'save':
        if predAn == 'training':
            plt.savefig(cwd + 'References_cm.png')
        elif predAn == 'analysis':
            plt.savefig(cwd + 'Tool_analysis_species_cm.png')
        plt.close()
    return ax


class Popup(QWidget):

    def __init__(self, name='', sentence=''):
        QWidget.__init__(self)
        msg = QMessageBox.warning(self, name, sentence)


def popup(title, sentence):
    start = Popup(title, sentence)

    return
