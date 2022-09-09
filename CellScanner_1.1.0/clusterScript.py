""" This script is executing a clustering algorithm with a maximum of n clusters
from n to 1 we calculate the distance of each cluster to the reference cluster.
we keep the labelled clusters and the distance values. From 1 run to the other, if the labelled distances are the same
or bette, we decrease the number of cluster. If the number of labelled cluster decrease or if the distances are worst
we keep the last result as the best cluster option"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import machineLearningPackage as f
from sklearn.metrics import confusion_matrix

# ####IMPORT AND TREAT DATA :
def assignCluster(dist, sp):
    """

    :param dist:
    :param sp:
    :return:
    """
    clusterSP = [n for n in range(len(dist))]
    for b in range(len(dist)):
        order = f.position(dist)
        order_sp = f.position(np.array(dist).T.tolist())
        for i in range(len(dist)):
            for j in range(len(sp)):
                if order[i][j] == 0 and order_sp[j][i] == 0 and dist[i][j] != 10:
                    clusterSP[i] = sp[j]
                    dist[i] = [10] * len(sp)
                    for k in range(len(dist)):
                        dist[k][j] = 10

    return clusterSP


def annotation(data, prediction, clust_nb, nmax, mpRef, param, species):
    """

    :param data:
    :param prediction:
    :param clust_nb:
    :param nmax:
    :param mpRef:
    :param param:
    :param species:
    :return:
    """
    p = f.pd.concat([data, f.pd.DataFrame({'PREDICTION': prediction})], axis=1, join='inner')
    mpPred = p.groupby('PREDICTION').mean()
    dist = f.distance(mpPred, mpRef, column=param)
    spp = [0] * clust_nb
    dsp = {}
    order = f.position(dist)
    order_sp = f.position(np.array(dist).T.tolist())
    for i in range(clust_nb):
        for j in range(nmax):
            if order[i][j] == 0 and order_sp[j][i] == 0:
                spp[i] = species[j]
                dsp[species[j]] = dist[i][j]
    return spp, dsp, dist


def clustering(refFiles, species, predFiles, sppred, predAn, param, nbC=1000, nbC2=None, save='show', var=0.05,
               gating='line', showgat=None, method=1, fc='Accuri',channels=[], dicChannels={}):
    """

    :param refFiles:
    :param species:
    :param predFiles:
    :param sppred:
    :param predAn:
    :param param:
    :param nbC:
    :param nbC2:
    :param save:
    :param var:
    :param gating:
    :param showgat:
    :param method:
    :param fc:
    :return:
    """
    fileNames = [a.split('/')[-1][:-4]for a in predFiles]
    # create dicrectory & file option
    cwd = 'Results/'
    now = f.datetime.now()
    dirName = now.strftime("%Y%m%d-%H_%M_%S/")
    f.os.mkdir(cwd + dirName)
    cwd = cwd + dirName
    if showgat and save is not None:
        showgat = save
    else:
        showgat = None
    f.fileOption(cwd, refFiles, species, predFiles, sppred, nbC, nbC2, gating, predAn, '', var, 0, True, 0,channels=channels,dicChannels=dicChannels,type=('Clustering: '))
    # ## import and treat data
    refArrays = f.importFile(refFiles, gating=gating, save=showgat, fc=fc, cwd=cwd, channels=channels,dicChannels=dicChannels)
    if refArrays == []:
        return 'None'
    predArrays = f.importFile(predFiles, gating=gating, save=showgat, fc=fc, cwd=cwd, channels=channels,dicChannels=dicChannels)
    if predArrays == []:
        return 'None'
    Data = []
    targetPred = []
    blankArrays = []
    blk = ['BLANK', 'Blank', 'blank']
    for i in range(len(species)):
        if species[i] in blk:
            blankArrays.append(refArrays[i])
    if gating == 'machine':
        refArrays, species = f.machineGating(refArrays, species, refArrays, species, cwd=cwd, show=showgat, name='ref',
                                             predType='neur', param=param)
        if predAn == 'analysis':
            predArrays, species2 = f.machineGating(refArrays + blankArrays, species + ['blank'] * len(blankArrays),
                                                   predArrays, sppred, cwd=cwd, show=showgat, name='pred',
                                                   param=param, predType='neur')
            # todo check if 10 and 5000 is not too much!! here we add the former blank from the new gating output
        else:
            refArrays = refArrays + blankArrays
            species = species + ['blank'] * len(blankArrays)

    if predAn == 'prediction':
        for anArray in predArrays:
            data2, atarget2, species2 = f.treat([anArray], sppred, None, mode='clustering', cluster=True)
            Data.append(data2)
            targetPred.append(atarget2)
    else:
        data2, target2, species2 = f.treat(predArrays, sppred, nbC2, mode='clustering', cluster=True)
        Data.append(data2)
        targetPred.append(target2)
    dataRef, targetRef, species = f.treat(refArrays, species, nbC, mode='analysis', cluster=True)
    print(dataRef)

    # ##################CALCULATION############################################
    nmax = len(species)
    n = nmax
    species.sort()
    # point moyen ref :
    r = f.pd.concat([dataRef, targetRef], axis=1, join='inner')
    mpRef = r.groupby('SPECIES').mean()
    if method == 1:
        for k in range(len(Data)):
            S = []
            P = []
            for nb in range(2, n + 1):
                clusters = AgglomerativeClustering(n_clusters=nb).fit(Data[k])
                predictL = clusters.labels_
                P.append(predictL)
                S.append((silhouette_score(Data[k], predictL)))

            posMax = S.index(max(S))
            clust_nb = posMax + 2
            clusterSP, dicSP, distance = annotation(Data[k], P[posMax], clust_nb, nmax, mpRef, param, species)

            label = P[posMax]
            if clust_nb == 2:
                clusters = AgglomerativeClustering(n_clusters=1).fit(Data[k])
                predictL = clusters.labels_
                spp, dsp, dist = annotation(Data[k], predictL, 1, nmax, mpRef, param, species)

                if f.isBetter(clusterSP, dicSP, spp, dsp, var):
                    clust_nb = 1
                    distance = dist
                    label = predictL

            clusterSP = assignCluster(distance, species)
            predLabel = []
            i = 0
            for i in range(len(label)):
                if clusterSP[label[i]] != 0:
                    predLabel.append(clusterSP[label[i]])
                else:
                    predLabel.append(label[i])
            allsp = species + [i for i, x in enumerate(clusterSP) if x == 0]
            if predAn == 'analysis':
                conf = confusion_matrix(targetPred[k], predLabel, labels=allsp)
                #pA = f.pd.DataFrame({'exp': targetPred[k], 'pred': predLabel})
                #pA.to_csv(cwd + 'results.csv', sep=';', mode='a')
                f.cmFile(conf, allsp, cwd, 'Prediction CM')
                f.plotConfusionMatrix(conf, allsp, save, cwd, normalize=True,
                                      name='CM with' + str(len(species)) + ' species', predAn='analysis')
            # newSp = list(set(species)) TODO check why I created it

            statistics = f.statAnalysis(predLabel, targetPred[k], allsp)

            f.exportStatistics(statistics, [''], cwd, 'predict')
            f.assessmentValue([statistics], allsp, cwd, [''], 'predict')
            if predAn == 'analysis' and save is not None:
                print(5.3)
                f.graph3d(Data[k], predLabel, targetPred[k], allsp + ['unknown'], param, statistics, save, cwd, 0,
                            name='with' + ' '.join(species), predtype=predAn, clust=True)
            elif predAn == 'prediction' and save is not None:
                f.graph3dRef(Data[k], predLabel, label, allsp + list(range(0, clust_nb)), param, statistics, save,
                               cwd,
                               refdata=dataRef, reflabel=targetRef, repeat=0,
                               name='with' + ' '.join(species), predtype='clustering',
                               clust=True)  # TODO changer les graph par les cluster numbers

    elif method == 2:
        # point moyen ref :
        for k in range(len(Data)):
            clusterSP = []
            dicSP = {}
            clust_nb = 0
            distance = []
            predLabel = []
            n = nmax
            label =[]
            while n > 0:# limit a deux clusters
                # ##PREDICTION##
                clusters = AgglomerativeClustering(n_clusters=n).fit(Data[k])
                predictL = clusters.labels_
                if n > 1:
                    print(n, ':', silhouette_score(Data[k], predictL))

                spp, dsp, dist = annotation(Data[k], predictL, n, nmax, mpRef, param, species)

                if f.isBetter(clusterSP, dicSP, spp, dsp, var):
                    dicSP = dsp
                    clust_nb = n
                    distance = dist
                    n = n - 1
                    label = predictL

                else:
                    n = 0
            clusterSP = assignCluster(distance, species)
            i = 0
            for i in range(len(label)):
                if clusterSP[label[i]] != 0:
                    predLabel.append(clusterSP[label[i]])
                else:
                    predLabel.append(label[i])
            # #### ASSESSMENT PART, GRAPH AND ACCURACY ####
            allsp = species + list(range(0, clust_nb))
            if predAn == 'analysis':
                conf = confusion_matrix(targetPred[k], predLabel, labels=allsp)
                pA = f.pd.DataFrame({'exp': targetPred[k], 'pred': predLabel})
                pA.to_csv(cwd + 'results.csv', sep=';', mode='a')
                # newSp = list(set(species)) TODO why same question
                f.plotConfusionMatrix(conf, allsp, save, cwd, normalize=True, name='CM with' + str(len(species)) + ' species', predAn='analysis')
                f.cmFile(conf, allsp, cwd, 'Prediction CM')
                statistics = f.statAnalysis(predLabel, targetPred[k], allsp)
                f.exportStatistics(statistics, [''], cwd, 'predict')
                f.assessmentValue([statistics], allsp, cwd, [''], 'predict')
            else:
                statistics = f.statAnalysis(predLabel, targetPred[k], allsp)
                f.exportPrediction(list(predLabel), [predFiles[k].split('/')[-1][:-4]], cwd, 'predict',
                                   k + 1, '', )
            if predAn == 'analysis' and save is not None:
                f.graph3d(Data[k], predLabel, list(targetPred[k]), allsp + ['unknown'], param, statistics, save, cwd, 0,
                            name='with' + ' '.join(species), predtype=predAn, clust=True)
            elif predAn == 'prediction' and save is not None:
                f.graph3dRef(Data[k], predLabel, label, allsp, param, statistics, save, cwd, refdata=dataRef,
                               reflabel=targetRef, repeat=0, name=fileNames[k], predtype='clustering',
                               clust=True)
                # TODO changer les graph par les cluster numbers

    return f.os.getcwd().replace('\\', '/') + '/' + cwd
