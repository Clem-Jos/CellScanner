""" function to launch the programm from the app:"""

from runScript import *
import db_script as db
import clusterScript as c
from fcsparser import parse
from pandas import read_csv
from PyQt5.QtWidgets import QMessageBox, QWidget


def run(self):
    """
    Function from the interface launching the machine learning program
    :param self:
    :return:
    """
    channels, replaced, dicChannels = db.getChannels()
    # prediction ou assessment
    typeA = self.state
    # Species
    species = []
    sp = self.selectedSpecies

    # reference file from the database
    files = []
    for s in sp:
        ref = db.getReferences(s, self.parent.aClass)
        for i in range(len(ref)):
            files = files + ['references/' + ref[i]]
            species = species + [s]
    # Analysis files:
    files2 = self.files
    # Parameter from the database : gating number repeat...
    repeat = int(db.getParamValue('reapt'))
    param = [db.getParamValue('graph1'), db.getParamValue('graph2'), db.getParamValue('graph3')]
    doubt = int(db.getParamValue('doubt')) / 100
    fc = db.getFcValue()
    nbC = int(db.getParamValue('nbC'))
    ratio = float(db.getParamValue('ratio'))
    var = float(db.getParamValue('clustDist'))

    average = db.getParamValue('average')
    if average in ['True', '1', 1, True]:
        average = True
    else:
        average = False

    figure = db.getParamValue('figure')
    if figure == 'None':
        figure = None
    elif figure == 'Save':
        figure = 'save'
    else:
        figure = 'show'

    gating = db.getParamValue('gating')
    if gating == 'None' or gating is None or gating == 0:
        gating = None
    elif gating == 'Line':
        gating = 'line'
    else:
        gating = 'machine'

    method = db.getParamValue('method')
    if method == 'Neural network':
        method = 'neur'
    elif method == 'Random forest':
        method = 'rand'
    elif method == 'Logistic regression':
        method = 'log'
    elif method == 'Random guess':
        method = 'rdguess'

    showGating = db.getParamValue('showGat')
    if showGating in [True, 1, 'True', '1']:
        showgat = True
    else:
        showgat = False

    nbC2 = int(db.getParamValue('nbC2'))
    if nbC2 == 0:
        nbC2 = None

    if typeA == 'prediction':
        files2 = files2[0]
        species2 = ['unknown'] * len(files2)
        directory = predictions(files, species, files2, species2, nbC, None, gating=gating, showgat=showgat,
                                predAn='prediction', predtype=method, ratio=ratio, repeat=repeat, average=average,
                                doubt=doubt, random_state=0, save=figure, fc=fc, param=param,channels=replaced,dicChannels=dicChannels)
    elif typeA == 'assessment':  # case of assessment
        species2 = []
        nf = []
        for i in range(len(files2)):
            nf = nf + files2[i]
            species2 = species2 + ([sp[i]] * len(files2[i]))
        files2 = nf.copy()
        directory = predictions(files, species, files2, species2, nbC, nbC2, gating=gating, showgat=showgat,
                                predAn='analysis', predtype=method, ratio=ratio, repeat=repeat, average=average,
                                doubt=doubt, random_state=0, save=figure, fc=fc, param=param,channels=replaced,dicChannels=dicChannels)
    elif typeA == 'clustering':
        files2 = files2[0]
        species2 = ['unknown'] * len(files2)
        directory = c.clustering(files, species, files2, species2, predAn='prediction', param=param, nbC=nbC, nbC2=None,
                                 save=figure, var=var,
                                 gating=gating, showgat=showgat, fc=fc, method=2,channels=replaced,dicChannels=dicChannels)
    elif typeA == 'clustA':
        species2 = []
        nf = []
        for i in range(len(files2)):
            nf = nf + files2[i]
            species2 = species2 + ([sp[i]] * len(files2[i]))
        files2 = nf.copy()
        directory = c.clustering(files, species, files2, species2, predAn='analysis', param=param, nbC=nbC, nbC2=nbC2,
                                 save=figure, var=var, method=2,
                                 gating=gating, showgat=showgat, fc=fc,channels=replaced,dicChannels=dicChannels)
    else:
        directory = 'None'

    return directory


def getChannelsFromFile(filename):
    if filename[-3:] == 'csv':
       data = read_csv(filename, header=0, sep=',', nrows=2)
    elif filename[-3:] == 'fcs':
        meta, data = parse(filename, reformat_meta=True)
    return (list(data.columns))



def isempty(L):
    v = True
    sum = 0
    if len(L) > 0:
        for l in L:
            sum = sum + len(l)
        if sum > 0:
            v = False
    return v
