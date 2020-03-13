def createFileNames(count,identifiers=['kothaddekha','openslrbn6k3seco'],
                lblids=['ImageArray','LabelArray','LenthArray']):
    
    fileNames = []

    for i in range(count):
        fn = []
        for n in lblids:
            fn.append('no_'+str(i)+"_"+identifiers[0]+"_"+n+"_"+identifiers[1]+'.npy')

        fileNames.append(fn) 

    return fileNames

def main_func():

    fileNames = createFileNames(3)

    print(fileNames)


if __name__ == "__main__":
    main_func()