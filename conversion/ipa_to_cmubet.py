import re

# cmubet to ipa look up table import
from dataTables import CMUBET
# import CMUBET  # fix folder structure

# CMUBET symbol set
m_CMUBET = CMUBET.data

# IPA <-> CMUBET lookup tables
# IPA symbols not in CMUBET are included from Arpabet
i2c_lookup = {
    # Modified in CMUBET
    "tʃ": "CH",
    "i:": "IY",
    "i": "IY",
    "ɪ": "IH",
    # from https://en.wikipedia.org/wiki/Arpabet
    "ə": "AX",
    "aɪ": "AY",
    "aʊ": "AW",
    "ɝ": "ER",
    "ɚ": "AXR",
    "dʒ": "JH",
    "m̩": "EM",
    "n̩": "EN",
    "ŋ̍": "EN",  # "ENG",
    "ɫ": "L",
    "ɫ̩": "L",  # "EL",
    "r": "R",
    "ɾ": "DX",
    "ɾ̃": "DX",  # "NX",
    "ʔ": "Q",
    # from http://courses.csail.mit.edu/6.345//notes/IPA/
    "e": "EY",
    "ɦ": "HH",  # "HV",
    "o": "OW",
    "ɨ": "IX",
    # add to nearest phoneme
    "a": "AA",
    "ɜ": "ER",
    "ɒ": "O",
}
for CMUBET_symbol, IPA_symbol in m_CMUBET.items():
    i2c_lookup[IPA_symbol] = CMUBET_symbol

# needs to read from file
#textIPA = "pi.roɟ.pur ɟe.lar ka.u.kʰa.li bɔn.d̪ɔ.re ek.ti mat̪.t̪ro ba.li.ka bid̪.d̪a.lɔj a.cʰe ɟa es.bi ʃɔr.ka.ri uc.co.ba.li.ka bid̪.d̪a.lɔj na.me po.ri.ci.t̪o"

# textIPA = ' '.join(w.replace('ɟ', 'ʤ')
#                   for w in textIPA.split())

# debug print
# print(textIPA)

# textIPA = ' '.join(w.replace('.', '')
#                   for w in textIPA.split())

# Debug print
# print(textIPA)


def IPA_to_CMUBET(text):

    # text = re.sub(r".", "", text)
    text = ' '.join(w.replace('ɟ', 'ʤ')
                    for w in text.split())

    text = ' '.join(w.replace('.', '')
                    for w in text.split())

    text = re.sub("ːː", ":", text)
    text = text.lstrip("/[").rstrip("]/")
    text = text.strip("'-!$")
    text += " "
    CMUBET_lst = []
    i = 0
    while i < len(text) - 1:
        if text[i:i+2] in i2c_lookup.keys():
            CMUBET_lst.append(i2c_lookup[text[i:i+2]])
            i += 1
        elif text[i] in i2c_lookup.keys():
            CMUBET_lst.append(i2c_lookup[text[i]])
        i += 1
    return " ".join(CMUBET_lst)


# debug print
# print(IPA_to_CMUBET(textIPA))
