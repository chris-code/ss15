import unittest
import os.path
import pandas as pd

def fileExists(filePath):
    return os.path.isfile(filePath)

def fileJsonType(fileName):
    return(fileName.lower().endswith('.json'))

def fileCSVType(fileName):
    return(fileName.lower().endswith('.csv'))

def correctTrainFormat(fileName):
    if(fileJsonType(fileName)):
        train = pd.read_json(fileName)
        return (len(train.columns))
    
def correctTrainColumns(fileName):
    if(fileJsonType(fileName)):
        train = pd.read_json(fileName)
        return('id' in train.columns and 'cuisine' in train.columns and 'ingredients' in train.columns)

class MyTest(unittest.TestCase):
    def fileTraining(self):
        """Check whether a file exists at the given path"""
        self.assertEqual(fileExists('train.json'), True, "Correct")
        self.assertEqual(fileExists('test.json'), True)
        self.assertEqual(fileExists('tes.json'),False)
        
    def fileTrainingJson(self):
        """Checks whether the file at the given path is a .json type"""
        self.assertEqual(fileJsonType('train.json'), True, "Correct type")
        self.assertEqual(fileJsonType('test.json'), True, "Correct type")
        self.assertEqual(fileJsonType('test.jso'), False, "False type")
    
    def fileTrainingCSV(self):
        """Check whether the file given at the path is a correct .csv file"""
        self.assertEqual(fileCSVType('results.csv'), True, "Correct type")
        
    def testTraining(self):
        """Test the format of the json files"""
        self.assertEqual(correctTrainFormat('train.json'), 3, "Corrrect quantity of columns")
        self.assertEqual(correctTrainColumns('train.json'), True, "Correct Format")
        
if __name__ == "__main__":
    unittest.main()