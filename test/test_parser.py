__version__ = '0.1.0'


from parser import Parse

def main():
    #testParseForm1()
    #testParseForm2()
    #testParseForm3()
    testParseForm4()

def testParseForm1():
    parseClass = Parse(1)
    TrainSet = parseClass.getTrain()
    TestSet = parseClass.getTest()
    
    #print(TrainSet['No'])
    #print(TrainSet['Words'])
    #print(TrainSet['Topic'])

def testParseForm2():
    parseClass = Parse(2)
    TrainSet = parseClass.getTrain()
    TestSet = parseClass.getTest()
    
    #print(TrainSet[1][500])
    #print(TestSet[1][100])


def testParseForm3():
    parseClass = Parse(3)
    TrainSet = parseClass.getTrain()
    TestSet = parseClass.getTest()
    
    print(TrainSet[0])
    print(TestSet[0])

def testParseForm4():
    parseClass = Parse(4)
    TrainSet = parseClass.getTrain()
    TestSet = parseClass.getTest()
    
    print(TrainSet[0])
    print(TestSet[0])
    

main()

        
    
