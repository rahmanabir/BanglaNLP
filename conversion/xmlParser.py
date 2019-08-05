from xml.dom.minidom import parse
import xml.dom.minidom
from ipa_to_cmubet import IPA_to_CMUBET
from csvWriter import savetoCSV

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse("speech_corpus.xml")

root = DOMTree.documentElement
# if collection.hasAttribute("shelf"):
#  print "Root element : %s" % collection.getAttribute("shelf")

# Get all the movies in the collection
sentences = root.getElementsByTagName("sentences")
i = 0
# Print detail of each movie.
for sentence in sentences:
    while i <= 10:
        orthograph = sentence.getElementsByTagName('orthograph')[i]
        #print("Orthograph: %s" % orthograph.childNodes[0].data)
        phonetic_form = sentence.getElementsByTagName('phonetic_form')[i]
        #print("Phonetic_form: %s" % phonetic_form.childNodes[0].data)
        # print(IPA_to_CMUBET(phonetic_form.childNodes[0].data))
        cmubet_form = IPA_to_CMUBET(phonetic_form.childNodes[0].data)
        savetoCSV(orthograph.childNodes[0].data,
                  phonetic_form.childNodes[0].data, cmubet_form)
        i += 1


print("WRITE COMPLETE")
