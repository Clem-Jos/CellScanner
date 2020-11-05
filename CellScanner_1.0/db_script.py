from sqlite3 import *
from shutil import copyfile
import os

"""This file contains all functions linking the program and the database """


def execute(phrase):
    """This function create a connection to the database and execute a command.

    :param phrase: str SQL sentence to execute
    :return: list of str received from the database after command execution.
    """
    conn = connect('bd.db')
    res = []
    cursor = conn.cursor()
    cursor.execute(phrase)
    conn.commit()
    for row in cursor:
        res.append(row)
    conn.close()
    return res


def getClassNames(isapred=False):
    """This function extract class name from database, if it's for a prediction purpose the class should not be empty.

    :param isapred: Boolean, if True only class with at least 2 records with references will be selected. Because
                    necessary for a prediction.
    :return: List of class Names
    """
    classes = []
    if not isapred:
        phrase = """SELECT NAME FROM CLASS"""
    else:
        phrase = "select name from class inner join (select ID_CLASS,count(ID_RECORD)AS c from CLASS_RECORD where " \
                 "id_record in(select ID_RECORD from RECORD_REFERENCE) GROUP BY ID_CLASS)as T on class.ID = " \
                 "T.ID_CLASS where c>=2 "

    res = execute(phrase)
    for row in res:
        classes = classes + list(row)
    return classes


def getRecordNames(aClass, isapred=False):
    """ This function extract record names from the database for a specific class. If it is a prediction, only record
    with a reference will be extracted
    :param aClass: str class name
    :param isapred: boolean if True, only record from specified class with a reference will be selected.
    :return: List of record names
    """
    records = []
    if not isapred:
        phrase = "Select NAME FROM RECORD INNER JOIN (SELECT ID_RECORD FROM CLASS_RECORD INNER JOIN (SELECT ID FROM " \
                 "CLASS WHERE NAME ='" + aClass + "') AS X ON CLASS_RECORD.ID_CLASS = X.ID) AS Y ON RECORD.ID = " \
                                                  "Y.ID_RECORD ORDER BY NAME "
    else:
        phrase = "select Name from RECORD inner join (select * from RECORD_REFERENCE INNER JOIN CLASS_RECORD ON " \
                 "RECORD_REFERENCE.ID_RECORD = CLASS_RECORD.ID_RECORD WHERE CLASS_RECORD.ID_CLASS =(SELECT ID FROM " \
                 "CLASS WHERE NAME = '" + aClass + "') AND IS_A_REF ='T') as T on record.ID = T.ID_RECORD group by " \
                                                   "NAME ORDER BY NAME"
    res = execute(phrase)
    for row in res:
        records = records + list(row)
    return records


def getReferenceNames(aRecord, aClass):
    """This function extract reference name from database for a specific record :
    used to show reference in the update data function
    :param aClass:
    :param aRecord: str record name
    :return: List of reference names
    """
    references = []
    isARef = []
    phrase = "Select NAME,IS_A_REF FROM REFERENCE INNER JOIN (SELECT ID_REFERENCE,IS_A_REF FROM RECORD_REFERENCE " \
             "WHERE ID_RECORD IN (SELECT ID FROM RECORD WHERE NAME='" + aRecord + "') AND ID_CLASS =(SELECT ID FROM " \
                                                                                  "CLASS WHERE NAME ='" + aClass + \
             "')ORDER BY IS_A_REF DESC) AS Y ON REFERENCE.ID = Y.ID_REFERENCE "
    res = execute(phrase)
    for row in res:
        a = list(row)
        references.append(a[0])
        isARef.append(a[1])
    return references, isARef


def changeReferences(aRecord, newRefs):
    """This function update the active references for a specific record.
    :param aRecord: str record name
    :param newRefs: List of references to set as active.
    :return: Only change db content so nothing is return
    """
    phrase = "UPDATE RECORD_REFERENCE SET IS_A_REF='F' WHERE ID_RECORD =(SELECT ID FROM RECORD WHERE NAME ='" + \
             aRecord + "')"
    execute(phrase)
    for aRef in newRefs:
        phrase2 = "UPDATE RECORD_REFERENCE SET IS_A_REF='T' WHERE ID_REFERENCE =(SELECT ID FROM REFERENCE WHERE " \
                  "NAME ='" + aRef + "'); "
        execute(phrase2)


def addClass(className):
    """Create a new class in the database if not exist
    :param className: str for class name
    :return: boolean if it exist or not
    """
    if doesntExsist('CLASS', className):
        phrase = "INSERT INTO CLASS(NAME) VALUES ('" + className + "')"
        execute(phrase)
        return False
    else:
        return True


def addFc(fcName):
    if doesntExsist('FLOWCYTOMETER', fcName):
        phrase = "INSERT INTO FLOWCYTOMETER(NAME) VALUES ('" + fcName + "')"
        execute(phrase)
        return False
    else:
        return True


def addRecord(aClass, recordName):
    """Add a new record from a specific class in the database if it doesn't exist in the database
    :param aClass: str for class name
    :param recordName: str for record name
    :return: boolean if it exist or not
    """
    if doesntExsist('RECORD', recordName):  # or if class_record doesn't exist
        phrase = "INSERT INTO RECORD(NAME) VALUES ('" + recordName + "'); "
        execute(phrase)
    exist = execute('SELECT * FROM CLASS_RECORD WHERE ID_CLASS =(SELECT ID FROM CLASS WHERE NAME ="' + aClass +
                    '") AND ID_RECORD =(SELECT ID FROM RECORD WHERE NAME="' + recordName + '")')
    if not exist:
        phrase = "INSERT INTO CLASS_RECORD(ID_CLASS,ID_RECORD) VALUES ((SELECT ID FROM CLASS WHERE NAME ='" + aClass + \
                 "' ),(SELECT ID FROM RECORD WHERE NAME ='" + recordName + "')) "
        execute(phrase)
        return False
    return True


def addReference(aRecord, fileName, aClass):  # don't add a link if exist already #class
    """ Add a reference to a record, if already in the database, add only a link to the record. If already exist only a
    link with the record is created. If link already exist, ??? does it add the red or not???? #

    :param aClass:
    :param aRecord:
    :param fileName:
    :return:
    """
    fileName = fileName.replace('\\', '/')
    referenceName = fileName.split('/')[-1]
    referenceName = referenceName.replace(' ', '_')

    # give an option to chose the name of the reference popup

    if doesntExsist('REFERENCE', referenceName):
        if not os.path.exists('references/' + referenceName):
            copyfile(fileName, 'references/' + referenceName)  #
        phrase = "INSERT INTO REFERENCE(NAME) VALUES ('" + referenceName + "'); "
        execute(phrase)

    exist = execute(
        "SELECT * FROM RECORD_REFERENCE WHERE ID_RECORD IN (SELECT ID FROM RECORD WHERE NAME ='" + aRecord +
        "') AND ID_REFERENCE =(SELECT ID FROM REFERENCE WHERE NAME='" + referenceName +
        "') AND ID_CLASS =(SELECT ID FROM CLASS WHERE NAME='" + aClass + "')")
    if not exist:
        phrase = "INSERT INTO RECORD_REFERENCE(ID_RECORD,ID_CLASS,ID_REFERENCE,IS_A_REF) VALUES ((SELECT ID_RECORD " \
                 "FROM CLASS_RECORD WHERE ID_RECORD IN (SELECT ID FROM RECORD WHERE NAME ='" + aRecord + "' ) AND " \
                 "ID_CLASS=(SELECT ID FROM CLASS WHERE NAME ='" + aClass + "')),(SELECT ID FROM CLASS WHERE NAME ='" + \
                 aClass + "'),(SELECT ID FROM REFERENCE WHERE NAME ='" + referenceName + "'),'F') "
        execute(phrase)
        return False
    return True


def addChannels(fc,channels,ref, update):
    # get fc id from fc name
    phrase = "SELECT ID FROM FLOWCYTOMETER WHERE NAME='"+fc+"'"
    res = execute(phrase)
    fcid = res[0][0]
    if update:
        # del all values from table channels where fc id = fc id
        phrase = "DELETE FROM CHANNELS WHERE ID_FC ="+str(fcid)
        res = execute(phrase)
    #for each channels name
    for i in range(len(channels)):
        if ref[i] != '':
            #get the id from the ref name
            # crete the phrase
            phrase = "INSERT INTO CHANNELS (NAME,ID_FC,ID_REF_CN) VALUES ('"+channels[i]+"',"+str(fcid)+",(SELECT ID FROM REF_CHANNELS WHERE NAME='"+str(ref[i])+"'))"
        else:
            phrase = "INSERT INTO CHANNELS (NAME,ID_FC) VALUES ('" + channels[i] + "'," + str(fcid) + ")"
            #create the phrase
        res = execute(phrase)


def updateFc():
    phrase = "DELETE FROM FLOWCYTOMETER WHERE ID NOT IN (SELECT ID_FC FROM CHANNELS INNER JOIN FLOWCYTOMETER WHERE FLOWCYTOMETER.ID=CHANNELS.ID_FC GROUP BY ID_FC)"
    execute(phrase)
    return


def doesntExsist(table, name):
    """Ask the database if an item is in a specific table or not
    :param table: str for tale name
    :param name: str for item name
    :return: Bollean, True if the item doesn't exsist
    """
    phrase = "SELECT NAME FROM " + table + " WHERE NAME ='" + name + "'"
    res = execute(phrase)
    a = []
    for val in res:
        a = a + list(val)
    if not a:
        return True
    else:
        return False


def delClass(aClass):
    """Delete a class from the database, delete all link with record, delete record if the record is not liked to
    another class. Delete references, if not linked to any record.

    :param aClass: str class name
    :return: Changes done in the database, nothing to return
    """
    phrase = "DELETE FROM CLASS_RECORD WHERE ID_CLASS =(SELECT ID FROM CLASS WHERE NAME = '" + aClass + "'); "
    execute(phrase)
    phrase = "DELETE FROM CLASS WHERE NAME = '" + aClass + "'; "
    execute(phrase)

    phrase = "DELETE FROM RECORD WHERE  ID IN (SELECT ID FROM RECORD LEFT JOIN CLASS_RECORD ON RECORD.ID = " \
             "CLASS_RECORD.ID_RECORD WHERE CLASS_RECORD.ID_RECORD IS NULL); "
    execute(phrase)
    phrase = "DELETE FROM RECORD_REFERENCE WHERE ID_RECORD  IN (SELECT RECORD_REFERENCE.ID_RECORD FROM RECORD_" \
             "REFERENCE LEFT JOIN RECORD ON RECORD_REFERENCE.ID_RECORD = RECORD.ID WHERE RECORD.ID IS NULL); "
    execute(phrase)
    phrase = "DELETE FROM REFERENCE WHERE  ID IN (SELECT ID FROM REFERENCE LEFT JOIN RECORD_REFERENCE ON " \
             "REFERENCE.ID = RECORD_REFERENCE.ID_REFERENCE WHERE RECORD_REFERENCE.ID_REFERENCE IS NULL); "
    execute(phrase)
    delRefFile()


def delRecord(aClass, aRecord):
    """ Delete a record from a class, delete reference from the deleted record if not liked to other records or if the
    record is not liked to another class
    :param aClass:  str for class name
    :param aRecord: str for record name
    :return:  Changes done in the database, nothing to return
    """

    phrase = "DELETE FROM CLASS_RECORD WHERE ID_RECORD =(SELECT ID FROM RECORD WHERE NAME = '" + aRecord + \
             "')AND ID_CLASS =(SELECT ID FROM CLASS WHERE NAME ='" + aClass + "');"  # for a specific class ?
    execute(phrase)
    phrase = "DELETE FROM RECORD WHERE  ID IN (SELECT ID FROM RECORD LEFT JOIN CLASS_RECORD ON " \
             "RECORD.ID = CLASS_RECORD.ID_RECORD WHERE CLASS_RECORD.ID_RECORD IS NULL); "
    execute(phrase)
    phrase = "DELETE FROM RECORD_REFERENCE WHERE ID_RECORD  IN (SELECT RECORD_REFERENCE.ID_RECORD FROM " \
             "RECORD_REFERENCE LEFT JOIN RECORD ON RECORD_REFERENCE.ID_RECORD = RECORD.ID WHERE RECORD.ID IS NULL); "
    execute(phrase)
    phrase = "DELETE FROM REFERENCE WHERE  ID IN (SELECT ID FROM REFERENCE LEFT JOIN RECORD_REFERENCE ON " \
             "REFERENCE.ID = RECORD_REFERENCE.ID_REFERENCE WHERE RECORD_REFERENCE.ID_REFERENCE IS NULL); "
    execute(phrase)
    delRefFile()


def delRefFile():
    phrase = 'SELECT NAME FROM REFERENCE'
    filesdb = execute(phrase)
    filedb=[]
    files = os.listdir('./References/')
    for row in filesdb:
        a = list(row)
        filedb = filedb + a
    for f in files:
        if f not in filedb:
            os.remove('./References/'+f)


def delReference(aRecord, aReference, aClass):
    """From a record name, delete the reference associated to the record, and delete the reference from the database if
    not liked to another record.
    :param aClass:
    :param aRecord: str name of the record
    :param aReference: str name for a reference
    :return: Changes done in the database, nothing to return
    """
    phrase = "DELETE FROM RECORD_REFERENCE WHERE ID_REFERENCE =(SELECT ID FROM REFERENCE WHERE NAME = '" + aReference +\
             "') AND ID_RECORD IN (SELECT ID FROM RECORD WHERE NAME ='" + aRecord + \
             "') AND ID_CLASS=(SELECT ID FROM CLASS WHERE NAME='" + aClass + "');"
    execute(phrase)
    phrase = "DELETE FROM REFERENCE WHERE  ID IN (SELECT ID FROM REFERENCE LEFT JOIN RECORD_REFERENCE ON " \
             "REFERENCE.ID = RECORD_REFERENCE.ID_REFERENCE WHERE RECORD_REFERENCE.ID_REFERENCE IS NULL); "
    execute(phrase)
    if not execute('SELECT * FROM REFERENCE WHERE NAME ="' + aReference + '"'):
        os.remove('References/' + aReference)


def getCnFromRef(refName,fcName):
    phrase ="SELECT NAME from CHANNELS WHERE ID_FC =(SELECT ID FROM FLOWCYTOMETER WHERE NAME ='"+fcName+"') AND ID_REF_CN =(SELECT ID FROM REF_CHANNELS WHERE NAME ='"+refName+"')"
    res = execute(phrase)
    if res == []:
        return res
    else :
        return res[0][0]


def getReferences(aRecord, aClass):
    """This function extract active references (checked in the 'update data' window) from a record name.
    :param aClass:
    :param aRecord: str record name
    :return: list of reference names
    """
    references = []
    phrase = "Select NAME FROM REFERENCE INNER JOIN (Select ID_REFERENCE, IS_A_REF FROM RECORD_REFERENCE where " \
             "ID_CLASS =(select id from class where name ='" + aClass + "') and ID_RECORD=(select id from record " \
             "where name ='" + aRecord + "'))AS Y ON REFERENCE.ID = Y.ID_REFERENCE WHERE IS_A_REF='T' "

    res = execute(phrase)
    for row in res:
        a = list(row)
        references = references + a
    return references


def getChannels():
    """This function extract the channels list from the database according to the active flowcytometer."""
    # get the flowcytometer number active
    channels=[]
    toreplace=[]
    replaced=[]
    dict={}
    phrase = "SELECT VALUE FROM PARAMETER WHERE ID ='fc' AND ACTIVE=1"
    res = execute(phrase)
    if res == []:
        channels=[]
        replaced = []
        dict = {}
    else :
        fc = res[0][0]

    # get the list of param + the list changed channels
        phrase2 = "SELECT NAME,ID_REF_CN FROM CHANNELS WHERE ID_FC ="+fc
        res2 = execute(phrase2)
        for row in res2:
            a = list(row)
            channels.append(a[0])
            toreplace.append(a[1])
        phrase3 = "SELECT NAME,ID FROM REF_CHANNELS "
        res3 = execute(phrase3)
        Lname = []
        Lid = []
        replaced=channels.copy()
        for row in res3:
            a = list(row)
            Lname.append(a[0])
            Lid.append(a[1])
        for i in range(len(Lid)):
            dict[channels[toreplace.index(Lid[i])]] = Lname[i]
        for key,value in dict.items():
            replaced = [value if x == key else x for x in replaced]

    return channels,replaced, dict


def getFlowcytometerList():
    phrase = "SELECT NAME FROM FLOWCYTOMETER"
    res = execute(phrase)
    fc = []
    for row in res:
        a = list(row)
        fc.append(a[0])
    return fc


def getRefChannels():
    phrase = "SELECT NAME,DESC FROM REF_CHANNELS"
    res = execute(phrase)
    channels = []
    desc = []
    for row in res:
        a = list(row)
        channels.append(a[0])
        desc.append(a[1])
    return channels, desc


def getSpecies():
    """Ask in the database all species name referenced for the species auto-completer.
    :return: List of bacterial species names
    """
    phrase = "SELECT NAME FROM REF_SPECIES WHERE ID_RECORD IS NULL"
    res = execute(phrase)
    classes = []
    for row in res:
        classes = classes + list(row)
    return classes


def getParamValue(param):
    """Get a parameter value from the database.
    :param param: Parameter name
    :return: string parameter value
    """
    phrase = "SELECT VALUE FROM PARAMETER WHERE ACTIVE =1 AND ID ='" + param + "'"
    res = execute(phrase)
    if res==[]:
        return 'NULL'
    return res[0][0]


def getFcValue():
    v = getParamValue('fc')
    phrase = "SELECT NAME FROM FLOWCYTOMETER WHERE ID ="+v
    res = execute(phrase)
    if res == []:
        return []
    else:
        return res[0][0]


def activeDefaultParam():
    """Set all default parameter as active in the database
    :return: Changes done in the database, nothing to return
    """
    phrase = "UPDATE PARAMETER SET ACTIVE = 1 WHERE STANDARD=1;"
    execute(phrase)
    phrase = "UPDATE PARAMETER SET ACTIVE = 0 WHERE STANDARD=0  "
    execute(phrase)


def desactivateParam():
    """Desactivate in the database all parameters that are set active.
    :return: Changes done in the database, nothing to return
    """
    phrase = "UPDATE PARAMETER SET ACTIVE = 0 WHERE ACTIVE = 1 AND ID NOT IN ('fc')"
    execute(phrase)


def desactivateFc():
    phrase = "UPDATE PARAMETER SET ACTIVE = 0 WHERE ACTIVE = 1 AND ID ='fc'"
    execute(phrase)


def removeUnusedParam():
    """Delete from the database all parameter values that are not used and not initial parameters
    :return: Changes done in the database, nothing to return
    """
    phrase = "DELETE FROM PARAMETER  WHERE ACTIVE = 0  AND STANDARD = 0"
    execute(phrase)


def updateReferenceInfo():
    # select reference from ref directory
    refFiles = [a for a in os.listdir('./references') if os.path.isfile(os.path.join('./references', a))]
    refFilesStr = '("'+'","'.join(refFiles)+'")'
    phrase = 'DELETE FROM REFERENCE WHERE NAME NOT IN '+refFilesStr
    execute(phrase)

    # remove all line in db that are not in the ref file list.


def saveParam(paramName, paramValue):
    """Update a parameter in the parameter table in the database
    :param paramName: Name of the parameter
    :param paramValue: new value of the parameter
    :return: Changes done in the database, nothing to return
    """
    phrase = "INSERT INTO PARAMETER VALUES('" + paramName + "','" + str(paramValue) + "',1,0)"
    execute(phrase)


def saveFc(value):
    phrase = "SELECT ID FROM FLOWCYTOMETER WHERE NAME ='"+value+"'"
    res = execute(phrase)
    if res:
        saveParam('fc',res[0][0])


def delFc(fcName):
    phrase = "DELETE FROM CHANNELS WHERE ID_FC =(SELECT ID FROM FLOWCYTOMETER WHERE NAME ='"+fcName+"')"
    execute(phrase)
    phrase = "DELETE FROM FLOWCYTOMETER WHERE NAME ='"+fcName+"'"
    execute(phrase)