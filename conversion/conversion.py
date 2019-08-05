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
ortho = []
ipa = []
cmubet = []

i = 0   # sentence iterator
# Print detail of each sentence
for sentence in sentences:
    while i <= 12072:  # 12073 sentence entries in xml
        sent_id = sentence.getElementsByTagName('sent_id')[i]
        orthograph = sentence.getElementsByTagName('orthograph')[i]
        phonetic_form = sentence.getElementsByTagName('phonetic_form')[i]
        ipa_repaired = IPA_Repair(phonetic_form.childNodes[0].data)
        cmubet_form = IPA_to_CMUBET(ipa_repaired)

        s_id.append(sent_id.childNodes[0].data)
        ortho.append(orthograph.childNodes[0].data)
        ipa.append(ipa_repaired)
        cmubet.append(cmubet_form)
        i += 1
        if (i % 500 == 0):
            print(i)
            df = pd.DataFrame(list(zip(s_id, ortho, ipa, cmubet)),
                              columns=['sID', 'Bangla', 'IPA', 'CMUBET'])
            df.to_csv('final_dataset.csv', index=False)


print("WRITE to CSV COMPLETE")
