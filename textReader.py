from geopy.geocoders import Nominatim
#pip install locationtagger
import nltk
import spacy

# essential entity models downloads
nltk.downloader.download('maxent_ne_chunker')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger')


if __name__ == '__main__':
    ### BASIC STEP BY STEP:
        # 1. Split OCR output (which I assume is just one big string of \n delimited text
        # 2. Separate said words into categories: parsedList[]
        #     Categories:
        #       [0]CatalogNumber	[1]Specimen_voucher<MGCL #>	[2]Family	[3]Genus<Phoebis>	[4]Species<sennae>	[5]Subspecies<sennae>	[6]Sex<female/male>
        #       [7]Country	[8]State	[9]County	[10]Locality name	[11]Elevation min	[12]Elevation max	[13]Elevation unit	[14]Collectors	[15]Latitude	[16]Longitude
        #       [17]Georeferencing source	[18]Georeferencing precision	[19]Questionable label data	Do not publish
        #       [20]Collecting event start	[21]Collecting event end	[22]Date verbatim           *** NOTE: Collecting event start/end are the same, in MM/DD/YYYY format
        #       [23]Remarks public	[24]Remarks private	    [25]Cataloged date***can collect ourselves	[26]Cataloger First	Cataloger last***can collect ourselves
        #       [27]Prep type 1	Prep count 1	[28]Prep type 2	Prep number 2	[29]Prep type 3	Prep number 3	[30]Other record number
        #       [31]Other record source     [32]publication     [33]publication
        # 3. Input parsed list into a CSV format, where the first line is the label names and then the rest of it is the information
        # Consistencies:
        #   - The species name will always be first, followed by UF tag
        #   - Date will always have a year and month - date is tricky, but we can say that if there's 2 periods/commas, there's a day -
        #   if not, and there's only one (or only one space), it's a day
        #   - For questionable label, maybe we can put if the general confidence rating is low, don't publish
        #   - Labels 2, 23 - 33 are not part of the labels!


    ### Can be replaced with output from OCR
    sampleString = "Phoebis sennae\nsennae\nUF\nFLMNH\nMGCL 1163652\nCUBA: GRANMA\nEl Banco, Mpio. Buey Arriba\n1000m, Turquino massif\n2,xi,1994; L D & J Y Miller\n& L R Hernandez sta. 1994-43\nEX.SA. MAESTRA Allyn Museum Acc. 1994-16"
    f = open("vsheet.csv", "r")
    f.readline()    # Get past the tags
    print(f.readline())
    of = open("esheet.csv", "r")
    labels = of.readline()
    listOfLabels = labels.split(",")
    # print(len(listOfLabels))

    listOfStrings = sampleString.split()
    copy = sampleString.split()
    parsedList = [""] * 22

    # Category 0
    parsedList[0] = "#########"

    # Categories 3 - 6
    for x in range(3, 6):
        parsedList[x] = listOfStrings[0]
        listOfStrings.pop(0)

    listOfStrings.pop(0)    # Get rid of UF
    listOfStrings.pop(0)    # Get rid of FLMNH

    # Category 1
    voucher = ""
    for x in range(2):
        voucher += listOfStrings[0] + " "
        listOfStrings.pop(0)
    parsedList[1] = voucher[:-1]    # [:-1] takes care of the extra space

    index = 0
    if "coll." in listOfStrings:
        # eoc = End of Collectors; marks the end of the section detailing the collectors, and the beginning of the next label
        eoc = listOfStrings.index("coll.")

    # Extracting Location Entities:
    joinedStrings = " ".join(listOfStrings)
    placeEntity = locationtagger.find_locations(text=joinedStrings)

    # Getting all countries
    print("The countries in text : ")
    print(placeEntity.countries)

    # Getting all states
    print("The states in text : ")
    print(placeEntity.regions)

    # Getting all cities
    print("The cities in text : ")
    print(placeEntity.cities)

    print("\n")
    print(listOfStrings)
    print("")
    print(listOfLabels)
    print(parsedList)
    print("")

    # sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
    # dictionary_path = "newDict2.txt"
    # dictionary_path2 = "bigramDict.txt"
    # eng_Dict = "en-80k.txt"

    # sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, separator="$")
    # sym_spell.load_dictionary(dictionary_path2, term_index=0, count_index=1, separator="$")
    # sym_spell.load_dictionary(eng_Dict, term_index=0, count_index=1)
    # sym_spell.load_bigram_dictionary(dictionary_path2, term_index=0, count_index=1, separator="$")
