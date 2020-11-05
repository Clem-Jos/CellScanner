"""PIPELINE 1 : normal prediction """
# IMPORT FILES

import machineLearningPackage as f
import numpy as np

import os
from sklearn.metrics import confusion_matrix

from datetime import datetime




def predictionMultiple(files, refArrays, species, files2, Data, target2, nbC, repeat=0,
                       param=None, predAn='prediction', predType='rand', ratio=1 / 7.0, random_state=None,
                       save='show', cwd='', average=True):
    """
    predictionMultiple create one model from reference files. Then predict one or several file depending on the
    prediction files and the type of prediction (prediction or analysis)
    :param random_state:
    :param Data:
    :param target2:
    :param refArrays:
    :param average:
    :param files:       List of monoculture reference file names to create the model.
    :param species:     List of monoculture species corresponding at 'files'.
    :param files2:      List of monoculture file(s) or community file(s) for prediction.
    :param nbC:         Number of cell selected per species for model creation (1000)
    :param repeat:       Number of the current run (print references graph for first run only)
    :param param:       List of 3 column names defining 3D graph axes
    :param predAn:      'prediction' is set for unknown prediction, one prediction is done for each file in files2
                        or 'analysis' is set for tool or community assessment. all files from files2 are merged into one
                         prediction
    :param predType:    Type of machine learning calculation ('rand': random forest,'neur':neural network,
                        'log': logistic regression,'rdguess': random guessing
    :param ratio:       ratio Learning/ training on dataset
    :param save:        'show' show 3D graph and confusion matrix on screen or 'save' save graphs on dedicated
                        directory.
    :param cwd:         saving file location
    :return:            statistics about model, statistics about prediction,list of predicted label for files, list of
                        expected results,list of confusion matrix,list of species,and list of datafiles.
    """

    if param is None:
        param = ['FL1-A', 'FL3-A', 'FSC-A']
    data, target, species = f.treat(refArrays, species, nbC, mode='train')
    scaler, classifier, predict_lbl, known_lbl, predict_data = f.learning(predType, data, target, ratio, random_state,
                                                                        species)

    conf = confusion_matrix(known_lbl, predict_lbl, species)
    statistics = f.statAnalysis(predict_lbl, known_lbl, species)
    fls = []
    for file in files:
        fls.append(file.split('/')[-1][:-4])
    f.exportStatistics(statistics, ['-'.join(fls)], cwd, 'align')
    if repeat == 0:
        f.graph3d(predict_data, predict_lbl, known_lbl, species, param, statistics, save, cwd, repeat,
                  name='Reference repeat 1', predtype='training')

    statistics2 = []
    predict_lbl2 = []
    conf2 = []
    if predAn == 'prediction':
        for i in range(len(Data)):
            pred_lbl2 = f.predict(predType, scaler, classifier, Data[i], species)
            predict_lbl2.append(pred_lbl2)
            statistics2.append(f.statAnalysis(pred_lbl2, target2[i], species))
            if not average and save is not None and repeat == 0:
                f.graph3d(Data[i], pred_lbl2, target2[i], species + ['unknown'], param, statistics2[i], save, cwd,
                          repeat,
                          name=files2[i].split('/')[-1][:-4] + ' repeat 1', predtype=predAn)
    else:
        predict_lbl2 = f.predict(predType, scaler, classifier, Data[0], species)  # probleme d'importation dans Analysis
        statistics2.append(f.statAnalysis(predict_lbl2, target2[0], species))
        conf2 = confusion_matrix(target2[0], predict_lbl2, species)
        predict_lbl2 = [predict_lbl2]
        if not average and save is not None and repeat == 0:
            f.graph3d(Data[0], predict_lbl2[0], target2[0], species, param, statistics2[0], save, cwd, repeat,
                      name='Tool analysis prediction repeat 0', predtype=predAn)
    return statistics, statistics2, predict_lbl2, target2, conf, conf2, species, Data


def predictions(files, species, files2, species2, nbC=1000, nbC2=None, gating='line', showgat=False,
                predAn='prediction',
                predtype='neur', ratio=1 / 7.0, repeat=1, average=True,
                doubt=0, random_state=None, save='save', fc='Accuri',
                param=None,channels = [], dicChannels = {}, ):
    """
    This Function is used by the program. It runs several time the function 'prediction_multiples' to create statistical
    results. The average parameter and the doubt parameter add more flexibility to the result from the different
    repeat. It also allows to print graph and create statistic and result files.
    :param nbC2:
    :param param:
    :param files:
    :param species:
    :param files2:
    :param species2:
    :param nbC:         cf prediction_multiples
    :param gating:
    :param showgat:
    :param predAn:      cf prediction_multiples
    :param predtype:    cf prediction_multiples
    :param ratio:       cf prediction_multiples
    :param repeat:       Number of run wanted
    :param average:     If True, function create an average prediction from n run prediction.
    :param doubt:       proportion minimum of same prediction
    :param random_state:
    :param save:
    :param fc:
    :return: results directory with figures is save='save', parameters file, prediction and statistic files.
    """
    # todo default channel name and dicChannels
    if param is None:
        param = ['FL3-A', 'FL1-A', 'FSC-A']
    cwd = 'Results/'
    now = datetime.now()
    dirName = now.strftime("%Y%m%d-%H_%M_%S/")
    os.mkdir(cwd + dirName)
    cwd = cwd + dirName
    f.fileOption(cwd, files, species, files2, species2, nbC, nbC2, gating, predAn, predtype, ratio, repeat, average,
                 doubt,channels=channels,dicChannels=dicChannels,fc=fc)
    if showgat and save is not None:
        showgat = save
    else:
        showgat = None
    Data = []
    target2 = []
    refArrays = f.importFile(files, gating=gating, save=showgat, fc=fc, cwd=cwd,channels=channels,dicChannels=dicChannels)
    if refArrays == [] :
        return 'None'
    predArrays = f.importFile(files2, gating=gating, save=showgat, fc=fc, cwd=cwd,channels=channels,dicChannels=dicChannels)
    if predArrays == []:
        return 'None'

    blankArrays = []
    blk =['BLANK', 'Blank', 'blank']
    for i in range(len(species)):
        if species[i] in blk:
            blankArrays.append(refArrays[i])
    if gating == 'machine':
        refArrays,species = f.machineGating(refArrays, species, refArrays, species,cwd=cwd,show=showgat,name='ref')
        if predAn == 'analysis':
            predArrays, species2 = f.machineGating(refArrays+blankArrays, species+['blank']*len(blankArrays), predArrays, species2,cwd = cwd,show = showgat, name='pred')
            # todo check if 10 and 5000 is not too much!!
        else :
            refArrays = refArrays + blankArrays
            species = species+['blank']*len(blankArrays)

    if predAn == 'prediction':
        for anArray in predArrays:
            data2, atarget2, species2 = f.treat([anArray], species2, None, mode=predAn)
            Data.append(data2)
            target2.append(atarget2)
    else:
        data2, target2, species2 = f.treat(predArrays, species2, nbC2, mode=predAn)
        Data.append(data2)
        target2 = [target2]

    # treat data
    statisticsLearn = []
    statisticsPred = []
    predict_lbls = []
    confusionM = []
    confusionM2 = []
    acc1 = []
    F1 = []
    oldSpecies = species.copy()
    for i in range(repeat):
        stat, stat2, predict_lbl, target, conf, conf2, species, \
            data2 = predictionMultiple(files, refArrays, oldSpecies, files2, Data, target2, nbC, repeat=i,
                                       param=param, predAn=predAn, predType=predtype, ratio=ratio,
                                       random_state=random_state, save=save, cwd=cwd, average=average)
        statisticsLearn.append(stat)
        statisticsPred.append(stat2)
        predict_lbls.append(predict_lbl)
        confusionM.append(conf)
        confusionM2.append(conf2)
        acc1.append(stat.loc['MEAN', 'ACC'])
        F1.append(stat.loc['MEAN', 'F1'])

    f.assessmentValue(statisticsLearn, species, cwd, [], 'align')
    confM = f.averageConfM(confusionM)
    confM2 = f.averageConfM(confusionM2)
    f.cmFile(confM, species, cwd, 'Reference CM')
    if save is not None:
        f.plotConfusionMatrix(confM, species, save, cwd, normalize=True, name=' average CM for reference',
                            predAn='training')

    target = np.array(target)
    predict_lbls = np.array(predict_lbls)
    conf2 = []
    acc2 = []
    f12 = []
    for i in range(len(statisticsPred[0])):
        PredStat = []
        for j in range(repeat):
            PredStat.append(statisticsPred[j][i])
        if average:
            predict_lbl = f.bestPred(list(predict_lbls[:, i]), doubt=doubt)  # average of 10
            if len(predict_lbl) > 0:
                conf2.append(confusion_matrix(target[i], predict_lbl, species + ['unknown']))
                statistics2 = f.statAnalysis(list(predict_lbl), list(target[i]), species)

                acc2.append(statistics2.loc['MEAN', 'ACC'])
                f12.append(statistics2.loc['MEAN', 'F1'])

                if predAn == 'prediction':
                    if save is not None:
                        f.graph3d(data2[i], predict_lbl, target[i], species + ['unknown'], param, statistics2, save,
                                  cwd, i + 1, name=files2[i].split('/')[-1][:-4], predtype=predAn)

                    f.exportPrediction(list(predict_lbl), [files2[i].split('/')[-1][:-4]], cwd, 'predict',
                                       i + 1, 'AVERAGE', repeat)
                else:
                    f.exportStatistics(statistics2, [n.split('/')[-1][:-4] for n in files2], cwd,
                                       'predict')
                    f.assessmentValue([statistics2], species, cwd, [files2[i].split('/')[-1][:-4]], 'predict')
                    f.exportPrediction(list(predict_lbl), [files2[i].split('/')[-1][:-4]], cwd, 'predict',
                                       i + 1, 'AVERAGE', repeat)
                    print(cwd)
                    print('create the file')
                    f.cmFile(conf2[i], species, cwd, 'Prediction CM')
                    if save is not None:
                        f.graph3d(data2[i], predict_lbl, target[i], species + ['unknown'], param, statistics2, save,
                                  cwd, i + 1, name='with' + str(len(species)) + ' species', predtype=predAn)
                        f.plotConfusionMatrix(conf2[i], species + ['unknown'], save, cwd, normalize=True,
                                            name='CM with' + str(len(species)) + ' species', predAn='analysis')

            else:
                conf2 = []
        else:
            for j in range(repeat):
                if predAn == 'prediction':
                    f.exportPrediction(predict_lbls[j][i], [files2[i].split('/')[-1][:-4]], cwd,
                                       'predict', i, 'SINGLE')
                else:
                    f.exportStatistics(PredStat[j], [files2[i].split('/')[-1][:-4]], cwd, 'predict')
                    f.exportPrediction(predict_lbls[j][i], [files2[i].split('/')[-1][:-4]], cwd,
                                       'predict', i, 'SINGLE')
            if predAn == 'analysis':
                f.assessmentValue(PredStat, species, cwd, [files2[i].split('/')[-1][:-4]], 'predict')
                f.cmFile(confM2, species, cwd, 'Prediction CM')
                if save is not None:
                    f.plotConfusionMatrix(confM2, species + ['unknown'], save, cwd, normalize=True,
                                        name='average confusion matrix for Tool analysis', predAn='analysis')

    return os.getcwd().replace('\\', '/') + '/' + cwd
