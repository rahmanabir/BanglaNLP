from xml.dom.minidom import parse
import xml.dom.minidom
from ipa_to_cmubet import IPA_to_CMUBET  # IPA to CMUBET conversion function
from ipa_to_cmubet import IPA_Repair  # IPA repair function
import pandas as pd

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse("speech_corpus.xml")
root = DOMTree.documentElement

# Get all the sentences in the root
sentences = root.getElementsByTagName("sentences")

# lists to store sentence details
s_id = []
ipa = []
lookUp = []
i = 0   # sentence iterator
# Print detail of each sentence
for sentence in sentences:
    while i <= 10000:  # 12073 sentence entries in xml
        phonetic_form = sentence.getElementsByTagName('phonetic_form')[i]
        uniqueipa = list(set(phonetic_form.childNodes[0].data))
        print(uniqueipa)
        for unique in uniqueipa:
            if unique in lookUp:
                print("Already Exists")
            else:
                lookUp.append(unique)
        i += 1
        # for j in uniqueipa:
        #     print(j)

print(lookUp)
print("Number of unique chars: ", len(lookUp))
lookUp_file = open("uniqueIPA.txt", "w+")
lookUp_file.writelines('\n'.join(lookUp) + '\n')
lookUp_file.close()
