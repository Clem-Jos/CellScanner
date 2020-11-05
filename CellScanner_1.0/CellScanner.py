import sys
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QAction, QComboBox, QSpinBox, \
    QGridLayout, \
    QFileDialog, qApp, QMessageBox, QAbstractSpinBox, QCheckBox, QHBoxLayout, QScrollArea, QLineEdit, QCompleter, \
    QListView, QApplication, QDoubleSpinBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QStandardItemModel, QIcon, QFont,QColor,QPixmap
import db_script as db
from functools import partial

import runCalculation as r


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowIcon(QIcon('logo.png'))
        self.setWindowTitle("CellScanner")
        self.resize(250, 450)
        self.move(50, 200)
        db.updateReferenceInfo()
        self.label = QLabel("CellScanner")
        self.label2 = QLabel()
        self.img = QPixmap('logo.png')
        self.img=self.img.scaled(150,150,Qt.KeepAspectRatio, Qt.FastTransformation)
        self.label2.setPixmap(self.img)
        self.label2.resize(self.img.width(),self.img.height())
        self.label3 = QLabel("Let's predict what is \n in your medium!")

        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Century Gothic", 19, QFont.Bold))
        self.label2.setAlignment(Qt.AlignCenter)
        self.label3.setAlignment(Qt.AlignCenter)
        self.label3.setFont(QFont("Century Gothic", 15))
        self.buttonP = QPushButton('New Prediction')
        self.buttonA = QPushButton('Tool Analysis')
        self.buttonU = QPushButton('Update Data')
        self.buttonC = QPushButton('Clustering')
        self.buttonCa = QPushButton('Clustering Analysis')
        self.buttonP.clicked.connect(self.clickP)
        self.buttonA.clicked.connect(self.clickA)
        self.buttonU.clicked.connect(self.clickU)
        self.buttonC.clicked.connect(self.clickC)
        self.buttonCa.clicked.connect(self.clickCa)
        layout = QGridLayout()
        layout.addWidget(self.label,0,0,1,2)
        layout.addWidget(self.label2, 1, 0, 1, 2)
        layout.addWidget(self.label3, 2, 0, 1, 2)
        layout.addWidget(self.buttonP,3,0,1,2)
        layout.addWidget(self.buttonA,4,0,1,2)
        layout.addWidget(self.buttonC,5,0,1,1)
        layout.addWidget(self.buttonCa,5,1,1,1)
        layout.addWidget(self.buttonU,6,0,1,2)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.toolbarFunction()

    def toolbarFunction(self):
        exitAct = QAction(QIcon('exit.png'), 'Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(self.close)
        exitAct2 = QAction(QIcon('param.png'), 'Parameter', self)
        exitAct2.setShortcut('Ctrl+p')
        exitAct2.triggered.connect(self.clickS)
        fcAct = QAction(QIcon('fc.png'), 'Flow cytometer', self)
        fcAct.triggered.connect(self.clickFc)
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(exitAct)
        self.toolbar2 = self.addToolBar('Setting')
        self.toolbar2.addAction(exitAct2)
        self.toolbar3 = self.addToolBar('Flow cytometer')
        self.toolbar3.addAction(fcAct)
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def clickP(self):
        self.cams = WindowPredictAssess('prediction')
        self.cams.show()

    def clickA(self):
        self.cams = WindowPredictAssess('assessment')
        self.cams.show()

    def clickU(self):
        self.cams = WindowUpdate()
        self.cams.show()

    def clickC(self):
        self.cams= WindowPredictAssess('clustering')
        self.cams.show()

    def clickS(self):
        self.cams = SettingWindow()
        self.cams.show()

    def clickCa(self):
        self.cams= WindowPredictAssess('clustA')
        self.cams.show()

    def clickFc(self):
        self.cams = ConfigFlowcytometer()
        self.cams.show()

########################### PREDICTION WINDOWS

class WindowPredictAssess(QWidget):

    def __init__(self, state):
        QWidget.__init__(self)
        v = db.getFcValue()
        if v == []:
            msg = QMessageBox.warning(self, 'Error', 'Please select a flow cytometer to launch the program')
            self.show()
            self.close()
            return
        self.setWindowIcon(QIcon('logo.png'))
        self.state = state
        if self.state == 'clustering':
            self.setWindowTitle('Clustering step 1/4')
            self.title = QLabel('Cluster prediction of community\n    \n') #TODO
        elif self.state == 'assessment':
            self.setWindowTitle('Predict step 1/4')
            self.title = QLabel('Species prediction\nfor in-silico community\n    \n')
        else:
            self.setWindowTitle('Predict step 1/4')
            self.title = QLabel('Species prediction\nin community\n    \n')
        self.resize(400, 100)
        self.move(500, 200)

        self.title.setFont(QFont('Arial', 13, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        classList = db.getClassNames(True)
        self.classes = QComboBox()
        for a in classList:
            self.classes.addItem(a)
        self.ok1 = QPushButton('OK')
        self.ok1.clicked.connect(self.classUpdate)
        self.grid = QGridLayout()
        self.grid.addWidget(self.title, 0, 0, 1, 2)
        self.labelClass = QLabel('Class   ')
        self.grid.addWidget(self.labelClass, 2, 0)
        self.grid.addWidget(self.classes, 2, 1)
        self.grid.addWidget(self.ok1, 3, 1)
        self.setLayout(self.grid)

    def classUpdate(self):

        self.aClass = str(self.classes.currentText())
        self.classes.setDisabled(True)
        self.ok1.deleteLater()
        if self.state=='clustering' or self.state == 'clustA':
            self.setWindowTitle('Clustering step 2/4')
        else:
            self.setWindowTitle('Predict step 2/4')
        self.label1 = QLabel('Number of ' + self.aClass + ' records  ')
        # self.label1.setFont(QFont('Arial', 17, QFont.Bold))
        self.entry = QSpinBox(self)
        self.entry.setRange(2, len(db.getRecordNames(self.aClass, True)))
        self.ok = QPushButton('OK')
        self.ok.clicked.connect(self.update)
        self.grid.addWidget(self.label1, 3, 0)
        self.grid.addWidget(self.entry, 3, 1)
        self.grid.addWidget(self.ok, 4, 1)
        self.setLayout(self.grid)
        self.show()

    def update(self):
        value = self.entry.value()
        self.entry.setDisabled(True)
        self.ok.deleteLater()
        if self.state=='clustering'or self.state == 'clustA':
            self.setWindowTitle('Clustering step 3/4')
        else:
            self.setWindowTitle('Predict step 3/4')
        self.record = db.getRecordNames(self.aClass, True)
        self.teachersselect = CheckableComboBox(self)
        self.teachersselect.setMaxCheckable(value)
        for i in range(len(self.record)):
            self.addTeacher(self.record[i], i)
        self.teachersselect.lblSelectItem = QLabel(self)
        self.subtitle = QLabel('Selected records ')
        self.grid.addWidget(self.teachersselect, 4, 1)
        self.grid.addWidget(self.teachersselect.lblSelectItem, 5, 0)
        self.grid.addWidget(self.subtitle, 4, 0)
        self.ok2 = QPushButton('OK')
        self.ok2.setEnabled(False)
        self.grid.addWidget(self.ok2, 6, 1)
        self.ok2.clicked.connect(self.validate)

    def validate(self):
        speciesPos = self.teachersselect.getChecked()
        self.cams = WindowLPF(self, speciesPos)
        self.close()

    def addTeacher(self, species, i):
        self.teachersselect.addItem(species)
        item = self.teachersselect.model().item(i, 0)
        item.setCheckState(Qt.Unchecked)


class WindowLPF(QWidget):
    def __init__(self, parent=None, speciesPosition=[], *args, **kwargs):
        QWidget.__init__(self)
        self.setWindowIcon(QIcon('logo.png'))
        self.parent = parent
        self.state = self.parent.state
        self.species = self.parent.record
        self.speciesPos = speciesPosition
        self.resize(400, 100)
        self.move(500, 200)
        self.selectedSpecies = []
        self.grid = QGridLayout()
        if self.state == 'prediction':
            self.files = [[]]
            self.setWindowTitle('Predict step 4/4')
            self.title = QLabel('\nFile(s) selection for the prediction(s)\n    \n')
            self.title.setFont(QFont('Arial', 13, QFont.Bold))
            self.title.setAlignment(Qt.AlignCenter)
            self.inputFile = QPushButton('Select file(s)')
            self.selectedFiles = QLabel()
            self.inputFile.clicked.connect(partial(self.openFileNamesDialog, 0))
            self.ok = QPushButton('OK')
            self.ok.clicked.connect(self.report)
            for i in range(len(self.speciesPos)):
                self.selectedSpecies.append(self.species[self.speciesPos[i]])
            self.grid.addWidget(self.selectedFiles, 2, 0)
            self.grid.addWidget(self.ok, 3, 1)
            self.grid.addWidget(self.inputFile, 1, 0)

        elif self.state == 'assessment':
            self.files = len(self.speciesPos) * [[]]
            self.setWindowTitle('Predict step 4/4')
            self.title = QLabel('\nFile(s) selection for assessment\n    \n')
            self.title.setFont(QFont('Arial', 13, QFont.Bold))
            self.title.setAlignment(Qt.AlignCenter)
            self.buttons = []
            self.labels = []
            self.selectedFiles = []
            for i in range(len(self.speciesPos)):
                self.i = i
                self.selectedSpecies.append(self.species[self.speciesPos[i]])
                self.labels.append(QLabel(self.species[self.speciesPos[i]]))
                self.buttons.append(QPushButton('Select file'))
                self.selectedFiles.append(QLabel())
                self.buttons[i].clicked.connect(partial(self.openFileNamesDialog, i))
                self.grid.addWidget(self.labels[i], i + 1, 0)
                self.grid.addWidget(self.selectedFiles[i], i + 1, 1)
                self.grid.addWidget(self.buttons[i], i + 1, 2)
            self.ok = QPushButton('OK')
            self.ok.clicked.connect(self.report)
            self.grid.addWidget(self.ok, i + 2, 2)
        elif self.state == 'clustering':
            self.files = [[]]
            self.setWindowTitle('Clusering step 4/4')
            self.title = QLabel('\nFile(s) selection for the clustering(s)\n    \n')
            self.title.setFont(QFont('Arial', 13, QFont.Bold))
            self.title.setAlignment(Qt.AlignCenter)
            self.inputFile = QPushButton('Select file(s)')
            self.selectedFiles = QLabel()
            self.inputFile.clicked.connect(partial(self.openFileNamesDialog, 0))
            self.ok = QPushButton('OK')
            self.ok.clicked.connect(self.report)
            for i in range(len(self.speciesPos)):
                self.selectedSpecies.append(self.species[self.speciesPos[i]])
            self.grid.addWidget(self.selectedFiles, 2, 0)
            self.grid.addWidget(self.ok, 3, 1)
            self.grid.addWidget(self.inputFile, 1, 0)
        elif self.state == 'clustA':
            self.files = len(self.speciesPos) * [[]]
            self.setWindowTitle('Clustering step 4/4')
            self.title = QLabel('\nFile(s) selection for assessment\n    \n')
            self.title.setFont(QFont('Arial', 13, QFont.Bold))
            self.title.setAlignment(Qt.AlignCenter)
            self.buttons = []
            self.labels = []
            self.selectedFiles = []
            for i in range(len(self.speciesPos)):
                self.i = i
                self.selectedSpecies.append(self.species[self.speciesPos[i]])
                self.labels.append(QLabel(self.species[self.speciesPos[i]]))
                self.buttons.append(QPushButton('Select file'))
                self.selectedFiles.append(QLabel())
                self.buttons[i].clicked.connect(partial(self.openFileNamesDialog, i))
                self.grid.addWidget(self.labels[i], i + 1, 0)
                self.grid.addWidget(self.selectedFiles[i], i + 1, 1)
                self.grid.addWidget(self.buttons[i], i + 1, 2)
            self.ok = QPushButton('OK')
            self.ok.clicked.connect(self.report)
            self.grid.addWidget(self.ok, i + 2, 2)
        self.grid.addWidget(self.title, 0, 0, 1, 2)
        self.setLayout(self.grid)
        self.show()

    def setSpeciesPos(self, aList):
        self.speciesPos = aList

    def openFileNamesDialog(self, i):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select data file(s)", "",
                                                "All Files (*csv *fcs);;CSV Files (*.csv);;FCS Files (*.fcs)",
                                                options=options)
        if files:
            self.files[i] = files
        j = '\n'
        if self.state == 'prediction' or self.state == 'clustering':
            self.selectedFiles.setText(j.join(self.files[i]))
        elif self.state == 'assessment' or self.state == 'clustA':
            self.selectedFiles[i].setText(j.join(self.files[i]))

    def report(self):
        a = r.isempty(self.files)
        if r.isempty(self.files):
            msg = QMessageBox.warning(self, 'No file selected', "Please select at least one file!", )
        else:
            self.calculation = WindowCalculation(self)
            self.calculation.show()
            qApp.processEvents()
            self.close()
            self.run()


    def run(self):
        directory = r.run(self)
        if directory =='None':
            self.calculation.close()
        else:
            self.calculation.setDirectory(directory)


class WindowCalculation(QWidget):
    def __init__(self, parent, *args, **kwargs):
        QWidget.__init__(self)

        self.grid = QGridLayout()
        self.setWindowIcon(QIcon('logo.png'))
        self.parent = parent
        self.directory = 'None'
        self.setWindowTitle('Calculation')
        self.resize(300, 100)
        self.move(500, 200)
        self.progressBar = QLabel('Prediction in progress...')
        self.progressBar.setFont(QFont('Arial', 13, QFont.Bold))
        self.grid.addWidget(self.progressBar, 1, 0)
        self.setLayout(self.grid)
        self.show()

    def setDirectory(self, directory):
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().setParent(None)

        self.directory = directory
        self.title = QLabel('Results saved here ')
        link = QLabel()
        l = '<a href=' + self.directory[:-1] + '> ' + self.directory[:-1] + '</a>'
        link.setText(l)
        link.setOpenExternalLinks(True)
        self.grid.addWidget(self.title)
        self.grid.addWidget(link)
        self.setLayout(self.grid)
        self.show()

########################################  DATABASE UPDATE


class WindowUpdate(QWidget):

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self)

        classList = db.getClassNames()
        self.setWindowIcon(QIcon('logo.png'))
        self.setWindowTitle('Update reference data')
        self.resize(300, 300)
        self.move(500, 200)
        self.label = QLabel('Update reference data:')
        self.label.setFont(QFont('Arial', 13, QFont.Bold))
        self.label2 = QLabel('In this page you can create, delete and update a class,\n '
                             'click update to add new records and  reference files to a class.')
        self.labelClass = QLabel('\nSelect a Class to update:')
        self.aClassQ = QComboBox()

        for a in classList:
            self.aClassQ.addItem(a)
        self.buttonAdd = QPushButton('Create a class')
        self.buttonAdd.clicked.connect(self.addClass)
        self.buttonDel = QPushButton('Delete a class')
        self.buttonDel.clicked.connect(self.delClass)
        self.ok = QPushButton('Update')
        self.ok.clicked.connect(self.updateClass)
        self.closeB = QPushButton('Exit')
        self.closeB.clicked.connect(self.closeW)
        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0, 1, 2)
        layout.addWidget(self.label2, 1, 0, 1, 2)
        layout.addWidget(self.labelClass, 2, 0)
        layout.addWidget(self.aClassQ, 3, 0)
        layout.addWidget(self.ok, 4, 0)
        layout.addWidget(self.buttonAdd, 3, 1)
        layout.addWidget(self.buttonDel, 4, 1)
        layout.addWidget(self.closeB, 5, 1)
        self.setLayout(layout)
        self.show()

    def addClass(self):
        window = CreateEntry(parent=self)

    def delClass(self):
        window = DelReference(self, 'null')
        self.close()

    def closeW(self):
        self.close()

    def updateClass(self):
        self.aClass = str(self.aClassQ.currentText())
        self.newWindow = ClassUpdate(self.aClass)
        self.newWindow.show()
        self.close()


class ClassUpdate(QWidget):
    def __init__(self, aClass):
        QWidget.__init__(self)
        self.layout = QHBoxLayout(self)
        self.setWindowIcon(QIcon('logo.png'))
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.grid = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout.addWidget(self.scrollArea)

        self.state = None
        self.aClass = aClass
        self.setWindowTitle('Update reference data')

        self.resize(900, 800)
        self.move(400, 100)
        self.sp = db.getRecordNames(self.aClass)

        self.files = []
        self.ref = []

        for aRec in self.sp:
            f, references = db.getReferenceNames(aRec, self.aClass)
            self.files.append(f)
            self.ref.append(references)
        self.label = QLabel('Select, add of delete a reference')
        self.label.setFont(QFont('Arial', 13, QFont.Bold))

        self.labels = []
        self.comboBox = []
        self.refButtons = []
        self.delButtons = []

        self.grid.addWidget(self.label, 0, 0, 1, 2)
        i = 0
        for i in range(len(self.sp)):
            self.labels.append(QLabel(self.sp[i]))
            self.comboBox.append(CheckableComboBox(self))
            self.comboBox[i].lblSelectItem = QLabel(self)
            self.comboBox[i].lblSelectItem.hide()
            self.refButtons.append(QPushButton('Add a Ref'))
            self.delButtons.append(QPushButton('Delete a Ref'))
            self.refButtons[i].clicked.connect(partial(self.addRef, self.sp[i]))
            self.delButtons[i].clicked.connect(partial(self.delRef, self.sp[i]))
            for j in range(len(self.files[i])):
                self.comboBox[i].addItem(self.files[i][j])
                item = self.comboBox[i].model().item(j, 0)
                if self.ref[i][j] == 'T':
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)
            self.grid.addWidget(self.labels[i], i + 1, 0)
            self.grid.addWidget(self.comboBox[i], i + 1, 1)
            self.grid.addWidget(self.refButtons[i], i + 1, 2)
            self.grid.addWidget(self.delButtons[i], i + 1, 3)
        self.new = QPushButton('Add a New record')
        self.new.clicked.connect(partial(self.newSp, self.aClass))
        self.delete = QPushButton('Del a record')
        self.delete.clicked.connect(partial(self.delRec, self.aClass))
        self.ok = QPushButton('OK')
        self.ok.clicked.connect(self.validate)
        self.grid.addWidget(self.delete, i + 2, 2)
        self.grid.addWidget(self.new, i + 2, 3)
        self.grid.addWidget(self.ok, i + 3, 3)

    def addRef(self, aSpecies):
        # create a new window + select ref file as csv of fcs
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileNames(self, "Select reference file for " + aSpecies, "",
                                               "All Files (*csv *fcs);;CSV Files (*.csv);;FCS Files (*.fcs)",
                                               options=options)
        for f in file:
            exist = db.addReference(aSpecies, f, self.aClass)
            if exist:
                msg = QMessageBox.warning(self, 'Reference already known', '')
        self.validate()
        self.close()

    def delRef(self, aSpecies):
        window = DelReference(self, self.aClass, aSpecies)
        self.close()

    def delRec(self, aClass):
        window = DelReference(self, self.aClass)
        self.close()

    def newSp(self, aClass):
        window = CreateEntry(self.aClass, self)

    def validate(self):
        res = []
        for i in range(len(self.sp)):
            res.append(self.comboBox[i].checkedItems())
        for j in range(len(self.sp)):
            L = []
            for item in res[j]:
                L.append(item.text())
            db.changeReferences(self.sp[j], L)
        self.window = ClassUpdate(self.aClass)
        self.window.show()
        self.close()


class DelReference(QWidget):
    def __init__(self, parent, aClass, aRecord=None, *args, **kwargs):
        QWidget.__init__(self)
        self.setWindowIcon(QIcon('logo.png'))
        self.state = None
        self.aClass = aClass
        self.parent = parent
        self.aRecord = aRecord
        if self.aClass == 'null':
            self.setWindowTitle('Delete a class')
            ref = db.getClassNames()
            self.label = QLabel('Select the class you want to delete')
        elif self.aRecord is None:
            self.setWindowTitle('Delete a record')
            ref = db.getRecordNames(self.aClass)
            self.label = QLabel('Select the record you want to delete from the class' + self.aClass)
        else:
            self.setWindowTitle('Delete a reference')
            ref, isRef = db.getReferenceNames(self.aRecord, self.aClass)
            self.label = QLabel('Select the reference you want to delete from the record ')
        self.resize(500, 100)
        self.move(500, 200)
        self.teachersSelect = CheckableComboBox(self)
        self.teachersSelect.setMaxCheckable(len(ref) - 1)
        for i in range(len(ref)):
            self.addTeacher(ref[i], i)
        self.teachersSelect.lblSelectItem = QLabel(self)
        self.delButtons = QPushButton('Delete')
        self.delButtons.clicked.connect(self.delRef)
        self.layout = QGridLayout()
        self.layout.addWidget(self.label, 0, 0, 1, 2)
        self.layout.addWidget(self.teachersSelect, 1, 1)
        self.layout.addWidget(self.teachersSelect.lblSelectItem, 1, 0)
        self.layout.addWidget(self.delButtons, 2, 1)
        self.setLayout(self.layout)
        self.show()

    def delRef(self):
        self.todel = self.teachersSelect.checkedItems()
        j = ' '
        l = []
        for item in self.todel:
            l.append(item.text())
        if self.todel:
            buttonReply = QMessageBox.question(self, 'Delete reference data',
                                               'Do you want to delete ' + j.join(l) + '?',
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                if self.aClass == 'null':
                    for one in l:
                        db.delClass(one)
                elif self.aRecord is None:
                    for one in l:
                        db.delRecord(self.aClass, one)
                else:
                    for one in l:
                        db.delReference(self.aRecord, one, self.aClass)
                self.parent.close()
                self.close()
                if self.aClass == 'null':
                    self.window = WindowUpdate()
                    self.window.show()
                else:
                    self.window = ClassUpdate(self.aClass)
                    self.window.show()
            else:
                self.close()
        else:
            self.close()

    def addTeacher(self, species, i):
        self.teachersSelect.addItem(species)
        item = self.teachersSelect.model().item(i, 0)
        item.setCheckState(Qt.Unchecked)


class CreateEntry(QWidget):
    def __init__(self, aClass=None, parent=None):
        QWidget.__init__(self)
        self.setWindowIcon(QIcon('logo.png'))
        self.parent = parent
        self.aClass = aClass
        self.setWindowTitle('Create a new entry')
        self.resize(500, 100)
        self.move(500, 200)
        if self.aClass is None:
            self.label = QLabel('Create a new class')
        else:
            self.label = QLabel('Create a new ' + self.aClass)
        self.label2 = QLabel('Name: ')
        self.textbox = QLineEdit(self)
        if self.aClass == 'Species':
            sp = db.getSpecies()
            completer = QCompleter(sp)
            self.textbox.setCompleter(completer)
        self.ok = QPushButton('Create')
        self.ok.clicked.connect(partial(self.createAction))
        self.layout = QGridLayout()
        self.layout.addWidget(self.label, 0, 0, 1, 2)
        self.layout.addWidget(self.label2, 1, 0)
        self.layout.addWidget(self.textbox, 1, 1)
        self.layout.addWidget(self.ok, 2, 1)
        self.setLayout(self.layout)
        self.show()

    def createAction(self):
        name = self.textbox.text()
        message = QMessageBox.question(self, 'Create Entry', "Do you want to create a new entry \n" + name + '?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            if self.aClass is None:
                exist = db.addClass(name)
                if exist:
                    msg = QMessageBox.warning(self, 'Impossible to create the class.',
                                              "Class " + name + " alredy exsist ")
                self.parent.close()
                self.window = WindowUpdate()
                self.window.show()
            else:
                exist = db.addRecord(self.aClass, name)
                if exist:
                    msg = QMessageBox.warning(self, 'Impossible to create the record.',
                                              "Record " + name + " alredy exsist ")
                self.parent.close()
                self.window = ClassUpdate(self.aClass)
                self.window.show()
            self.close()
        else:
            self.close()

########################################  SETTING WINDOW


class SettingWindow(QWidget):

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self)
        v = db.getFcValue()
        if v == []:
            msg = QMessageBox.warning(self, 'Error', 'Please select a flow cytometer to launch the program')
            self.show()
            self.close()
            return
        self.setWindowIcon(QIcon('logo.png'))
        self.layout = QHBoxLayout(self)
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.grid = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout.addWidget(self.scrollArea)

        title = QLabel('Advanced parameters')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Century Gothic", 15, QFont.Bold))
        self.fig = ['Save', 'Show', 'None']
        self.gat = ['None', 'Line', 'Machine']
        self.met = ['Neural network', 'Random forest', 'Logistic regression', 'Random guess']
        #self.graphOption = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A', 'FSC-H', 'SSC-H', 'FL1-H', 'FL2-H',
                         #   'FL3-H', 'FL4-H', 'Width']
        channels, self.graphOption, dic = db.getChannels()

        repeatL = QLabel('Number of runs ')
        self.repeat = QSpinBox(self)
        self.repeat.setRange(1, 50)


        averageL = QLabel('Create average prediction from runs')
        self.average = QCheckBox()
        doubtL = QLabel('Percentage minimum of appearance \nto validate a prediction')
        self.doubt = QSpinBox(self)
        self.doubt.setRange(0, 100)
        nbCl = QLabel('Training:\n Number of cells per record')
        self.nbC = QSpinBox(self)
        self.nbC.setRange(100, 10000)
        nbC2l = QLabel('Tool analysis:\n Number of cells per record')
        self.nbC2 = QSpinBox(self)
        self.nbC2.setRange(0, 10000)
        ratioL = QLabel('Ratio learning/test')
        self.ratio = QDoubleSpinBox()
        self.ratio.setRange(0, 2)
        self.ratio.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        figureL = QLabel('Figures view mode')
        self.figure = QComboBox()
        self.figure.addItems(self.fig)
        gatingL = QLabel('Gating type ')
        self.gating = QComboBox()
        self.gating.addItems(self.gat)
        self.ok = QPushButton('Save Settings')
        self.ok.clicked.connect(self.saveParam)
        self.default = QPushButton('Back on default')
        self.default.clicked.connect(partial(self.maj, True))
        self.methodL = QLabel("Machine learning method")
        self.method = QComboBox()
        self.method.addItems(self.met)
        self.showGat = QCheckBox()
        self.showGatL = QLabel('Show gating effect')
        self.clustDist = QDoubleSpinBox(self)
        self.clustDist.setRange(0, 5)
        self.clustDist.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.clustDistL = QLabel('Cluster distance %')

        self.axesL = QLabel('Select 3D graph axes')
        self.axe1 = QComboBox()
        self.axe1.addItems(self.graphOption)
        self.axe2 = QComboBox()
        self.axe2.addItems(self.graphOption)
        self.axe3 = QComboBox()
        self.axe3.addItems(self.graphOption)

        self.grid.addWidget(title, 0, 0, 1, 4)
        self.grid.addWidget(repeatL, 1, 0)
        self.grid.addWidget(self.repeat, 1, 1)
        self.grid.addWidget(averageL, 2, 0)
        self.grid.addWidget(self.average, 2, 1)
        self.grid.addWidget(doubtL, 3, 0)
        self.grid.addWidget(self.doubt, 3, 1)
        self.grid.addWidget(figureL, 4, 0)
        self.grid.addWidget(self.figure, 4, 1)
        self.grid.addWidget(gatingL, 1, 2)
        self.grid.addWidget(self.gating, 1, 3)
        self.grid.addWidget(nbCl, 3, 2)
        self.grid.addWidget(self.nbC, 3, 3)
        self.grid.addWidget(nbC2l, 4, 2)
        self.grid.addWidget(self.nbC2, 4, 3)
        self.grid.addWidget(self.methodL, 5, 2)
        self.grid.addWidget(self.method, 5, 3)
        self.grid.addWidget(ratioL, 6, 2)
        self.grid.addWidget(self.ratio, 6, 3)
        self.grid.addWidget(self.showGatL, 2, 2)
        self.grid.addWidget(self.showGat, 2, 3)
        self.grid.addWidget(self.axesL, 5, 0)
        self.grid.addWidget(self.axe1, 5, 1)
        self.grid.addWidget(self.axe2, 6, 1)
        self.grid.addWidget(self.axe3, 7, 1)
        self.grid.addWidget(self.clustDistL, 8,0 )
        self.grid.addWidget(self.clustDist, 8, 1)

        self.grid.addWidget(self.ok, 9, 0, 2, 2)
        self.grid.addWidget(self.default, 9, 2, 2, 2)
        self.setWindowTitle("Settings")
        self.resize(900, 500)
        self.move(50, 200)
        self.maj(False)

    def maj(self, default):

        if default:
            db.activeDefaultParam()
            db.removeUnusedParam()
        self.repeat.setValue(int(db.getParamValue('reapt')))
        checked = db.getParamValue('average')
        if checked == 'True' or checked == 1 or checked == True:
            v = 1
        else:
            v = 0
        self.average.setChecked(v)
        checked2 = db.getParamValue('showGat')
        if checked2 == 'True' or checked2 == 1 or checked2 == True:
            v2 = 1
        else:
            v2 = 0
        self.showGat.setChecked(v2)

        self.doubt.setValue(int(db.getParamValue('doubt')))
        self.figure.setCurrentIndex(self.fig.index(db.getParamValue('figure')))
        self.gating.setCurrentIndex(self.gat.index(db.getParamValue('gating')))
        self.method.setCurrentIndex(self.met.index(db.getParamValue('method')))

        self.nbC.setValue(int(db.getParamValue('nbC')))
        self.nbC2.setValue(int(db.getParamValue('nbC2')))
        self.ratio.setValue(float(db.getParamValue('ratio')))
        self.clustDist.setValue(float(db.getParamValue('clustDist')))
        g1 = db.getParamValue('graph1')
        g2 = db.getParamValue('graph2')
        g3 = db.getParamValue('graph3')
        sv = False
        if g1 in self.graphOption:
            self.axe1.setCurrentIndex(self.graphOption.index(g1))
        else:
            sv = True
        if g2 in self.graphOption:
            self.axe2.setCurrentIndex(self.graphOption.index(g2))
        else:
            sv = True
        if g3 in self.graphOption:
            self.axe3.setCurrentIndex(self.graphOption.index(g3))
        else:
            sv = True
        if sv:
            self.saveParam()

    def saveParam(self):
        db.desactivateParam()
        db.removeUnusedParam()
        db.saveParam("reapt", self.repeat.value())
        db.saveParam("average", self.average.isChecked())
        db.saveParam("showGat", self.showGat.isChecked())
        db.saveParam("clustDist", self.clustDist.value())
        db.saveParam("doubt", self.doubt.value())
        db.saveParam("figure", self.fig[self.figure.currentIndex()])
        db.saveParam("gating", self.gat[self.gating.currentIndex()])
        db.saveParam("method", self.met[self.method.currentIndex()])
        db.saveParam("graph1", self.graphOption[self.axe1.currentIndex()])
        db.saveParam("graph2", self.graphOption[self.axe2.currentIndex()])
        db.saveParam("graph3", self.graphOption[self.axe3.currentIndex()])
        db.saveParam("nbC", self.nbC.value())
        db.saveParam("nbC2", self.nbC2.value())
        db.saveParam("ratio", self.ratio.value())

        self.maj(False)


#################### PARAM FLOW CYTOMETER


class ConfigFlowcytometer(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self)

        self.fcList = db.getFlowcytometerList()
        self.setWindowIcon(QIcon('logo.png'))
        self.setWindowTitle('Flow cytometer configuration')
        self.resize(300, 300)
        self.move(500, 200)
        self.label = QLabel('Select the active flow cytometer')
        self.label.setFont(QFont('Arial', 13, QFont.Bold))
        db.updateFc()
        self.labelClass = QLabel('\nConfigure a flow cytometer')
        self.aClassQ = QComboBox()
        for a in self.fcList:
            self.aClassQ.addItem(a)
        # todo del the fc name if id not in channels column fcid

        v = db.getFcValue()
        if v != []:
            self.aClassQ.setCurrentIndex(self.fcList.index(v))

        self.buttonAdd = QPushButton('Add a flow cytometer')
        self.buttonAdd.clicked.connect(self.addClass)
        self.buttonDel = QPushButton('Delete a flow cytometer')
        self.buttonDel.clicked.connect(self.delClass)
        self.ok = QPushButton('Update a flow cytometer')
        self.ok.clicked.connect(self.updateFc)
        self.ok1 = QPushButton('OK')
        self.ok1.clicked.connect(self.validate)
        self.closeB = QPushButton('Exit')
        self.closeB.clicked.connect(self.closeW)
        layout = QGridLayout()
        layout.addWidget(self.label, 0, 0, 1, 2)
        layout.addWidget(self.aClassQ, 1, 0, 1, 2)
        layout.addWidget(self.labelClass, 2, 0)
        layout.addWidget(self.ok, 4, 0,1,2)
        layout.addWidget(self.buttonAdd, 3, 0,1,2)
        layout.addWidget(self.buttonDel, 5, 0,1,2)
        layout.addWidget(self.closeB, 6, 1)
        layout.addWidget(self.ok1,6,0)
        self.setLayout(layout)
        self.show()

    def addClass(self):
        self.newWindow = CreateFc(parent=self)
        self.close()
    def delClass(self):
        self.newWwindow = DeleteFc(parent=self)
        self.close()
    def closeW(self):
        self.close()

    def updateFc(self):
        self.aClass = str(self.aClassQ.currentText())
        self.newWindow = Update(parent=self)
        self.newWindow.show()
        self.close()

    def validate(self):
        db.desactivateFc()
        db.saveFc(self.fcList[self.aClassQ.currentIndex()])
        db.removeUnusedParam()
        if db.getFcValue() !=[]:
            self.aClassQ.setCurrentIndex(self.fcList.index(db.getFcValue()))

    #todo faire une fenetre ou on peut rentrer les parametres de la base de donnée


class Update(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self)
        self.setWindowIcon(QIcon('logo.png'))
        self.parent = parent
        self.setWindowTitle('FC selection for update')
        self.resize(500, 100)
        self.move(500, 200)
        self.fcNames = QComboBox()
        for a in self.parent.fcList:
            self.fcNames.addItem(a)

        self.label2 = QLabel('Name ')


        self.ok = QPushButton('Ok')
        self.ok.clicked.connect(partial(self.inputData))
        self.layout = QGridLayout()
        self.layout.addWidget(self.label2, 1, 0)
        self.layout.addWidget(self.fcNames, 1, 1)
        self.layout.addWidget(self.ok, 2, 1)
        self.setLayout(self.layout)


        self.show()

    def inputData(self):
        name = self.parent.fcList[self.fcNames.currentIndex()]
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, "Data selection", "Select data file for flow cytometer configuration",
                                                  "All Files (*csv *fcs);;CSV Files (*.csv);;FCS Files (*.fcs)",
                                                  options=options)
        if file:
            self.file = file
            channels = r.getChannelsFromFile(file)
            self.window = UpdateFc(name, channels,True)


class CreateFc(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self)
        self.setWindowIcon(QIcon('logo.png'))
        self.parent = parent
        self.setWindowTitle('Create a new flow cytometer')
        self.resize(500, 100)
        self.move(500, 200)

        self.label2 = QLabel('Name ')
        self.textbox = QLineEdit(self)

        self.ok = QPushButton('Create')
        self.ok.clicked.connect(partial(self.createAction))
        self.layout = QGridLayout()
        self.layout.addWidget(self.label2, 1, 0)
        self.layout.addWidget(self.textbox, 1, 1)
        self.layout.addWidget(self.ok, 2, 1)
        self.setLayout(self.layout)

        self.show()

    def createAction(self):
        self.name = self.textbox.text()
        message = QMessageBox.question(self, 'Create FC', "Do you want to create a new flow cytometer \n" + self.name + '?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            exist = db.addFc(self.name)
            if exist:
                msg = QMessageBox.warning(self, 'Impossible to create the class.',
                                              "Class " + self.name + " alredy exsist ")
            else:
                self.parent.close()
                self.close()
                self.openFileNameDialog()

        self.close()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, "Select data file for flow cytometer configuration", "",
                                                "All Files (*csv *fcs);;CSV Files (*.csv);;FCS Files (*.fcs)",
                                                options=options)
        if file:
            self.file = file
        else :
            self.file = ''
            return
        j = '\n'
        channels = r.getChannelsFromFile(file)
        self.window = UpdateFc(self.name,channels)
        self.window.show()


class DeleteFc(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self)
        self.setWindowIcon(QIcon('logo.png'))

        self.parent = parent
        self.setWindowTitle('Delete a flow cytometer')
        self.resize(500, 100)
        self.move(500, 200)
        self.fcNames = QComboBox()
        for a in self.parent.fcList:
            self.fcNames.addItem(a)
        self.label2 = QLabel('Name ')
        self.ok = QPushButton('Ok')
        self.ok.clicked.connect(self.createAction)
        self.layout = QGridLayout()
        self.layout.addWidget(self.label2, 1, 0)
        self.layout.addWidget(self.fcNames, 1, 1)
        self.layout.addWidget(self.ok, 2, 1)
        self.setLayout(self.layout)
        self.show()

    def createAction(self):
        fcName = self.parent.fcList[self.fcNames.currentIndex()]
        message = QMessageBox.question(self, 'Delete FC', "Do you want to delete \n" + fcName + '?',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            db.delFc(fcName)
        msg= QMessageBox.information(self,'Flow cytometer deleted','The flow cytometer '+fcName+' is removed from the database.')
        self.window = ConfigFlowcytometer()
        self.window.show()
        self.close()


class UpdateFc(QWidget):
    def __init__(self, name,channels=None,update=False):
        QWidget.__init__(self)
        self.layout = QHBoxLayout(self)
        self.setWindowIcon(QIcon('logo.png'))
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.grid = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout.addWidget(self.scrollArea)
        self.fcList = db.getFlowcytometerList()
        self.state = None
        self.fc = name
        self.update = update
        self.setWindowTitle('Channels selection')
        self.channels = channels
        self.ref_channels,self.ref_desc = db.getRefChannels()

        self.resize(900, 400)
        self.move(500, 200)
        self.labels = []
        self.descLabel = []
        self.comboBox = []
        self.refButtons = []
        self.delButtons = []
        self.label = QLabel('In order to do the line gating, please indicate which \nof your channels correpond to the indicated channels below:')
        self.label.setFont(QFont('Arial', 13, QFont.Bold))
        self.grid.addWidget(self.label, 0, 0, 1, 2)
        i = 0
        self.aClassQ = QComboBox()

        for i in range(len(self.ref_channels)):
            self.labels.append(QLabel(self.ref_channels[i]))
            self.descLabel.append(QLabel(self.ref_desc[i]))
            self.comboBox.append(QComboBox(self))
            for j in range(len(self.channels)):
                self.comboBox[i].addItem(self.channels[j])

            self.grid.addWidget(self.labels[i], i + 1, 0)
            self.grid.addWidget(self.descLabel[i], i + 1, 1)
            self.grid.addWidget(self.comboBox[i], i + 1, 2)
        self.ok = QPushButton('Next')
        self.ok.clicked.connect(self.validate)
        self.grid.addWidget(self.ok, i + 3, 3)
        if update:
            self.updateFields()
        self.show()

    def validate(self):
        self.res = []
        #todo befor or after openning the good window, add in the data base
        # validate only if the choosen ref are not in double
        for i in range(len(self.ref_channels)):
            self.v=self.comboBox[i].currentIndex()
            if self.v not in self.res:
                self.res.append(self.comboBox[i].currentIndex())
            else :
                msg = QMessageBox.warning(self, 'Data Warning', 'A channel can be used only once per reference channel')
                return

        self.window = ChannelSelect(self.fc,self.ref_channels,self.channels,self.res,self.update)
        self.window.show()
        self.close()

    def updateFields(self):
        for i in range(len(self.ref_channels)):
            v=db.getCnFromRef(self.ref_channels[i],self.fc)
            if v in self.channels:
                self.comboBox[i].setCurrentIndex(self.channels.index(v))


class ChannelSelect(QWidget):

    def __init__(self, fc, ref_channels,channels,values,update):
        QWidget.__init__(self)
        self.layout = QHBoxLayout(self)
        self.setWindowIcon(QIcon('logo.png'))
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.grid = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout.addWidget(self.scrollArea)
        self.fc = fc
        self.resize(800, 800)
        self.move(500, 200)
        self.ref_channels = ref_channels
        self.channels = channels
        self.values = values
        self.update = update

        self.Title = QLabel('Select the channels used in your analysis')
        self.Title.setFont(QFont('Arial', 13, QFont.Bold))
        self.checkbox=[]
        self.label = []
        self.label2 = []
        for i in range(len(channels)):
            self.label.append(QLabel(channels[i]))
            self.checkbox.append(QCheckBox())
            if i in self.values:
                self.label2.append(QLabel(self.ref_channels[self.values.index(i)]))
                self.checkbox[i].setChecked(True)
                self.checkbox[i].setDisabled(True)
            else:
                self.label2.append(QLabel())
        self.ok = QPushButton('Validate')
        self.ok.clicked.connect(self.validate)
        self.l1 = QLabel('Channel name')
        self.l2 = QLabel('Is used')
        self.l3 = QLabel('Reference channel name')
        self.grid.addWidget(self.Title, 0, 0, 1, 3)
        self.grid.addWidget(self.l1, 2, 0)
        self.grid.addWidget(self.l2, 2, 1)
        self.grid.addWidget(self.l3, 2, 2)
        for i in range(len(self.channels)):
            self.grid.addWidget(self.label[i], i + 3, 0)
            self.grid.addWidget(self.checkbox[i], i + 3, 1)
            self.grid.addWidget(self.label2[i], i + 3, 2)
        self.grid.addWidget(self.ok, i+5, 2)

    def validate(self):
        #todo ajouter à la base de données toute les valeures
        channels = []
        ref = []

        for i in range(len(self.channels)):
            if self.checkbox[i].isChecked():
                channels.append(self.channels[i])
                if i in self.values:
                    ref.append(self.ref_channels[self.values.index(i)])
                else:
                    ref.append('')
        print(channels)
        print(ref)
        print(self.update)
        db.addChannels(self.fc,channels,ref,self.update)
        msg = QMessageBox.information(self, 'Success', "The flow cytometer is created: \nDon't forget to select it as the current flow cytometer!")

        self.window = ConfigFlowcytometer()
        self.window.show()
        self.close()

##########################################


class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super(CheckableComboBox, self).__init__(parent)
        self.setWindowIcon(QIcon('logo.png'))
        self.parent = parent
        self.maxCheckable = 100
        self.checked = 0
        self.setView(QListView(self))
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QStandardItemModel(self))

    def setMaxCheckable(self, nvalue):
        self.maxCheckable = nvalue

    def enoughSp(self):
        if self.checked == self.maxCheckable:
            self.parent.ok2.setEnabled(True)
        else:
            self.parent.ok2.setEnabled(False)

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)
        self.on_selectedItems()
        if self.parent.state is not None:
            self.enoughSp()

    def checkedItems(self):
        checkedItems = []
        cnt = 0
        for index in range(self.count()):

            item = self.model().item(index)
            if item.checkState() == Qt.Checked:
                cnt = cnt + 1
                checkedItems.append(item)
        self.checked = cnt
        for index in range(self.count()):
            item = self.model().item(index)
            if cnt == self.maxCheckable:
                if item.checkState() == Qt.Unchecked:
                    item.setEnabled(False)
            else:
                item.setEnabled(True)
        return checkedItems

    def on_selectedItems(self):
        selectedItems = self.checkedItems()
        self.lblSelectItem.setText("")
        for item in selectedItems:
            self.lblSelectItem.setText("{} \n {} "
                                       "".format(self.lblSelectItem.text(), item.text()))

    def getChecked(self):
        pos = []
        for index in range(self.count()):
            item = self.model().item(index)
            if item.checkState() == Qt.Checked:
                pos.append(index)
        return pos

###########################################################################
logf = open('error.log', 'a')

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        palette = QPalette()
        palette.setColor(QPalette.Button, Qt.gray)
        palette.setColor(QPalette.Window, QColor(199,229,242))
        palette.setColor(QPalette.WindowText, QColor(31,71,85))
        app.setPalette(palette)
        start = MainWindow()
        start.show()
        sys.exit(app.exec_())
    except Exception as e:
        logf.write(str(e))
        logf.close()
