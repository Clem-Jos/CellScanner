import runScript as s
import os.path
from os import path
##RUN for 13 species
sp=['Bifidobacterium adolescentis','Blautia hydrogenotrophica','Bacteroides thetaiotaomicron',
    'Bacteroides uniformis','Bilophila wadsworthia','Collinsella aerofaciens','Escherichia coli','Faecalibacterium prausnitzii',
    'Prevotella copri','Parabacteroides merdae','Ruminococcus bromii','Roseburia intestinalis']
ba1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Bifidobacterium_adolescentis-050520-mGAM-1.fcs'
bh1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Blautia_hydrogenotrophica-211119-GAM-1.fcs'
bt1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Bacteroides_thetaiotaomicron-050520-mGAM-1.fcs'
bu1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Bacteroides_uniformis-050520-mGAM-1.fcs'
bw1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Bilophila_wadsworthia-050520-mGAM-1.fcs'

ca1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Collinsella_aerofaciens-050520-mGAM-1.fcs'
ec1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Escherichia_coli-050520-mGAM-1.fcs'
fp1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Faecalibacterium_prausnitzii-171219-GAM-1.fcs'
pc1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Prevotella_copri-050520-mGAM-1.fcs'
pm1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Parabacteroides_merdae-141019-GAM-1.fcs'
rb1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Ruminococcus_bromii-050520-mGAM-1.fcs'
ri1 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Roseburia_intestinalis-050520-mGAM-1.fcs'

ba2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Bifidobacterium_adolescentis-050520-mGAM-2.fcs'
bh2 =  'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Blautia_hydrogenotrophica-211119-GAM-2.fcs'
bt2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Bacteroides_thetaiotaomicron-050520-mGAM-2.fcs'
bu2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Bacteroides_uniformis-050520-mGAM-2.fcs'
bw2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Bilophila_wadsworthia-050520-mGAM-2.fcs'
ca2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Collinsella_aerofaciens-050520-mGAM-2.fcs'
ec2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Escherichia_coli-050520-mGAM-2.fcs'
fp2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Faecalibacterium_prausnitzii-171219-GAM-2.fcs'
pc2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Prevotella_copri-050520-mGAM-2.fcs'
pm2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Parabacteroides_merdae-141019-GAM-2.fcs'
rb2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Ruminococcus_bromii-050520-mGAM-2.fcs'
ri2 = 'C:/Users/u0128864/Desktop/Programm/CellScanner-1.0.0/references/Roseburia_intestinalis-050520-mGAM-2.fcs'

f1=[ba1,bh1,bt1,bu1,bw1,ca1,ec1,fp1,pc1,pm1,rb1,ri1]
f2=[ba2,bh2,bt2,bu2,bw2,ca2,ec2,fp2,pc2,pm2,rb2,ri2]

for f in bw1:
    print("File exists:" + str(path.exists(f)))
for f in bw2:
    print("File exists:" + str(path.exists(f)))
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
# FIND ALL COMBINATION
combination =s.f.createCombination(sp,2)
files=[]
Species=[]
files2=[]
Species2=[]
for aComb in combination:
    Species.append([aComb[0],aComb[1]])
    Species2.append([aComb[0], aComb[1]])
    files.append([f1[sp.index(aComb[0])],f1[sp.index(aComb[1])]])
    files2.append([f2[sp.index(aComb[0])], f2[sp.index(aComb[1])]])
print(len(Species))
print(len(files))
print(len(Species2))
print(len(files2))

#CHECK THE FORMULA ? MISS CHANNELS AND DICO
for file,file2,species,species2 in zip(files,files2,Species,Species2):
    s.predictions(file, species, file2, species2, nbC= 1000,nbC2=1000, gating='line', predAn='analysis', predtype='neur',
                  ratio=1 / 7.0, repeat=10, showgat=False, average=True, doubt=0.7, save='show',fc='Accuri',
                  param=['FSC-A','FL1-A','FL3-A'],
                  channels=['FSC-A','FSC-H','SSC-A','SSC-H','FL1-A','FL1-H','FL2-A','FL2-H','FL3-A','FL3-H','FL4-A','FL4-H','Width'])

