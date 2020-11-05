import runScript as s

sp=['Alistipes indistinctus','Bacteroides thetaiotaomicron','Bacteroides uniformis',
 'Bacteroides vulgatus','Blautia hydrogenotrophica','Collinsella aerorofaciens',
 'Escherichia coli','Odoriacter splanchnicus','Prevotella sp.','Roseburia intestinalis']

ai1 = 'C:/Users/u0128864/Desktop/Programm/phenotypeProgramm/references/Alistipes_indistinctus-061219-GAM-1.fcs'
bt1 = 'C:/Users/u0128864/Desktop/Programm/phenotypeProgramm/references/Bacteroides_thetaiotaomicron-061219-GAM-1.fcs'
bu1 = 'C:/Users/u0128864/Desktop/Programm/phenotypeProgramm/references/Bacteroides_uniformis-061219-GAM-1.fcs'
bv1 = 'C:/Users/u0128864/Desktop/Programm/phenotypeProgramm/references/Bacteroides_vulgatus-061219-GAM-1.fcs'
bh1 = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/Blautia_hydrogenotrophica-061219-GAM-1.fcs'
ca1 = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/Collinsella_aerofaciens-061219-GAM-1.fcs'
ec1 = 'D:/DATA/Flow_Cytometry/EXP_160120_Ratio/Escherichia_coli-160120-GAM-1.fcs'
os1 = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/Odoribacter_splanchnicus-061219-GAM-1.fcs'
p1  = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/prevotella_sp-061219-GAM-1.fcs'
ri1 = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/Roseburia_intestinalis-061219-GAM-1.fcs'


ai2 = 'C:/Users/u0128864/Desktop/Programm/phenotypeProgramm/references/Alistipes_indistinctus-061219-GAM-2.fcs'
bt2 = 'C:/Users/u0128864/Desktop/Programm/phenotypeProgramm/references/Bacteroides_thetaiotaomicron-061219-GAM-2.fcs'
bu2 = 'C:/Users/u0128864/Desktop/Programm/phenotypeProgramm/references/Bacteroides_uniformis-061219-GAM-2.fcs'
bv2 = 'C:/Users/u0128864/Desktop/Programm/phenotypeProgramm/references/Bacteroides_vulgatus-061219-GAM-2.fcs'
bh2 = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/Blautia_hydrogenotrophica-061219-GAM-2.fcs'
ca2 = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/Collinsella_aerofaciens-061219-GAM-2.fcs'
ec2 = 'D:/DATA/Flow_Cytometry/EXP_160120_Ratio/Escherichia_coli-160120-GAM-2.fcs'
os2 = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/Odoribacter_splanchnicus-061219-GAM-2.fcs'
p2  = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/prevotella_sp-061219-GAM-2.fcs'
ri2 = 'D:/DATA/Flow_Cytometry/EXP_061219_Monocultures/FCS_treated/Roseburia_intestinalis-061219-GAM-2.fcs'

f1=[ai1,bt1,bu1,bv1,bh1,ca1,ec1,os1,p1,ri1]
f2=[ai2,bt2,bu2,bv2,bh2,ca2,ec2,os2,p2,ri2]
"""
Please find an example of script to run mannualy the programm.

Each Brackets represent  one 'programm' experiment. You should have the same number of brackets in each list.
files: list of reference files for each experiment
Species: list of species matching reference files 
files2: list of files for  prediction or analysis. If you want to predict files content one prediction will be process for each file in the brakets.
If you want to do an assessment, all files in the brakets will be merged into one big prediction.
Species2: list of Species matching prediction or analysis files. If you don't know the species please put unknown for each. 
Mind the spelling it have to be the same than on 'Species' object.
    For the Parameters: 
    - 1000 : number of reference cells per species
    - gating = 'line' or None or 'Machine'
    - predAn = 'prediction' or 'analysis'
    - predtype = 'neur', 'rand', 'log' or 'rdguess'
    - ratio = 0.14 in the app (ratio test/learning)
    - reapt : number of run per prediction
    - average : built an average predition from the different runs
    - doubt : minium ratio of prediction to not be labeled unknown
    - save= 'save' or 'show' to print or save figures


"""


files = [[ai1,bt1],[bu1,bv1]]
Species=[['ai','bt'],['bu','bv']]
files2 = [[ai2],[bv2]]
Species2 = [['ai'],['bv']]


for file,file2,species,species2 in zip(files,files2,Species,Species2):
    s.predictions(file, species, file2, species2, nbC= 1000,nbC2=None, gating='line', predAn='analysis', predtype='neur', ratio=1 / 7.0,
                  repeat=10, showgat=False, average=True, doubt=0.7, save='save')