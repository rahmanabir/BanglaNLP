from xml.dom.minidom import parse
import xml.dom.minidom
import re
import pandas as pd
import csv

# dekhabet to ipa look up table
dekhabet = [
    [' ',' '],
    ['a','a'],
    ['e','j'],
    ['e','æ'],
    ['e','e'],
    ['i','i'],
    ['o','ɔ'],
    ['o','o'],
    ['u','u'],
    ['s','s'],
    ['r','r'],
    ['k','k'],
    ['sh','ʃ'],
    ['ch','c'],
    ['n','n'],
    ['m','m'],
    ['t','t'],
    ['d','d'],
    ['l','l'],
    ['j','ɟ'],
    ['h','ʰ'],
    ['h','h'],
    ['b','b'],
    ['p','p'],
    ['g','g'],
    ['ng','ŋ'],
    ['r','ɾ']
]

tokenlookup = [' ', 'a', 'e', 'i', 'o', 'u', 'g', 'p', 'b', 'h', 's', 't', 'j', 'l', 'm', 'n', 'r', 'd', 'k', 'ng', 'ch', 'sh']

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse('speech_corpus.xml')
root = DOMTree.documentElement
sentences = root.getElementsByTagName("sentences")
dk = []
ipa = []
for tup in dekhabet:
    dk.append(tup[0])
    ipa.append(tup[1])
# tokenlookup = list(set(dk))
# print(tokenlookup)


def ipa2dekhabet(text):
    text = re.sub("ːː", ":", text)
    text = text.lstrip("/[").rstrip("]/")
    text = text.strip("'-!$")
    # print(text)
    converted_dekhabet = ''
    tokens = []
    for char in text:
        try:
            a = ipa.index(char)
            b = dk[a]
            c = tokenlookup.index(b)
            tokens.append(c)
            converted_dekhabet+=dk[ipa.index(char)]
        except:
            pass
    # print(converted_dekhabet)
    return converted_dekhabet, tokens
    # print(tokens)

# ipa2dekhabet('ʃi.t̪er ʃu.ru.t̪e d̪e.ʃer ha.or bil cɔ.ran.cɔl')


def ConvertIPA2Dekhabet():
    # lists to store sentence details
    s_id = []
    ortho = []
    ipa = []
    dekhabet = []
    tokens = []
    i = 0   # sentence iterator
    # Print detail of each sentence
    for sentence in sentences:
        while i <= 12000:  # 12073 sentence entries in xml
            sent_id = sentence.getElementsByTagName('sent_id')[i].childNodes[0].data
            orthograph = sentence.getElementsByTagName('orthograph')[i].childNodes[0].data
            phonetic_form = sentence.getElementsByTagName('phonetic_form')[i].childNodes[0].data
            dekhabet_form, token_form = ipa2dekhabet(phonetic_form)
            # print(type(token_form),token_form)

            s_id.append(sent_id)
            ortho.append(orthograph)
            ipa.append(phonetic_form)
            dekhabet.append(dekhabet_form)
            tokens.append(token_form)
            i += 1
            if (i % 500 == 0):
                print(i,'items saved in csv')
                df = pd.DataFrame(list(zip(s_id, ortho, ipa, dekhabet, tokens)),columns=['sID', 'Bangla', 'IPA', 'Dekhabet', 'Tokens'])
                df.to_csv('dekhabet_dataLabels.csv', index=False)
    print("WRITE to CSV COMPLETE")


def FindUniqueChars():
    lookUp = []
    i = 0   # sentence iterator
    # Print detail of each sentence
    for sentence in sentences:
        while i <= 10000:  # 12073 sentence entries in xml
            phonetic_form = sentence.getElementsByTagName('phonetic_form')[i]
            uniqueipa = list(set(phonetic_form.childNodes[0].data))
            print(i+':',uniqueipa)
            for unique in uniqueipa:
                if unique not in lookUp:
                    lookUp.append(unique)
                    print('New Added:',unique)

            i += 1
            print(len(lookUp))


    print(lookUp)
    print("Number of unique chars: ", len(lookUp))
    lookUp_file = open("uniqueIPA.txt", "w+")
    lookUp_file.writelines('\n'.join(lookUp) + '\n')
    lookUp_file.close()

def MakeMatchingCSV(csvf='dekhabet_dataLabels.csv', txtf='data_RangedAudiofileList_1to2.txt', outf='dekhabet_dataLabelsRanged.csv'):
    rangedlist = []
    with open(txtf, 'r') as txtFile:
        reader = csv.reader(txtFile)
        for row in reader:
            txtline = re.sub("data/crblp/wav/", "", row[0])
            txtline = re.sub(".wav", "", txtline)
            rangedlist.append(txtline)
    txtFile.close()
    print('rangedlist length:',len(rangedlist))

    csvID = []
    csvBangla = []
    csvIPA = []
    csvDekha = []
    csvTokens = []
    with open(csvf, 'r') as csvFile:
        reader = csv.reader(csvFile)
        i = 0
        for row in reader:
            if row[0] in rangedlist:
                csvID.append(row[0])
                csvBangla.append(row[1])
                csvIPA.append(row[2])
                csvDekha.append(row[3])
                csvTokens.append(row[4])
                i+=1
                if (i % 250 == 0):
                    print(i,'items saved in csv')
                    df = pd.DataFrame(list(zip(csvID, csvBangla, csvIPA, csvDekha, csvTokens)),columns=['sID', 'Bangla', 'IPA', 'Dekhabet', 'Tokens'])
                    df.to_csv(outf, index=False)
        print(len(csvID))
        # print(csvDekha[1])
    csvFile.close()

def FindMaxMinCSVToken(csvf='dekhabet_dataLabelsRanged.csv'):
    max = 0
    min = 1000
    str = ''
    lenlist = []

    with open(csvf, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if row[0]=='sID':
                print(row)
            else:
                # lenlist.append(len(row[4]))
                count = row[4].count(',')
                if max < count:
                    max = count
                if min > count:
                    min = count
    csvFile.close()
    print('Str:',str)
    print('Min:',min)
    print('Max:',max)

def splitcsv(csvf='dekhabet_dataLabelsRanged.csv'):
    labels = []
    with open(csvf, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            ctext = []
            text = row[4]
            text = text.strip("'-!$[]")
            text = text.split(',')
            for t in text:
                if t=='Tokens':
                    pass
                else:
                    ctext.append(int(t))
            labels.append(ctext)
    csvFile.close()
    print(labels[0])
    print(labels[1])
    labels.pop(0)
    print(labels[0])
    print(labels[1])

def convertTokens2Dekhabet(input,adjust=0):
    out = ''
    for i in input:
        out += tokenlookup[i-adjust]
    return out

# ConvertIPA2Dekhabet()
# MakeMatchingCSV()
# FindMaxMinCSVToken()
# splitcsv()


