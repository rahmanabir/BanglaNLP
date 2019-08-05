import csv


def savetoCSV(orthograph, ipa, cmubet):
    with open('data_Bengali.csv', mode='a') as csv_file:

        #fieldnames = ['Orthograph', 'IPA', 'CMUBET']
        writer = csv.writer(csv_file)

        writer.writeheader()
        writer.writerow({orthograph, ipa, cmubet})
