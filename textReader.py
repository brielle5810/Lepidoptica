from geopy.geocoders import Nominatim
import nltk
import pandas as pd
import numpy as np
import spacy
from datetime import datetime
import pycountry
import locationtagger
import re
import roman
import datefinder


def find_more_dates(text):
    date_patterns = [
        # DD.MM.YYYY | DD,MM,YYYY | DD/MM/YYYY | DD-MM-YYYY   OR   MM.DD.YYYY | MM,DD,YYYY | MM/DD/YYYY | MM-DD-YYYY
        r"\d{2}[.,/-]\d{2}[.,/-]\d{4};*",
        # YYYY   OR   'YY
        r"(\d{4});?(?![a-zA-Z-])",
        r"'(\d{2});?(?![a-zA-Z-])",
        # Roman numerals format
        # DD.MM.YYYY where MM = roman numerals
        r"(?:\d{1,2}[.,])?(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)*[.,]\d{4};*",
        # DD MM YYYY where MM = date name (abbreviate or otherwise)
        ### CAN I REPLACE THESE WITH /w ? who's to say.....
        r"(?:\d{1,2} )?(?:jan|JAN|feb|FEB|mar|MAR|apr|APR|may|MAY|jun|JUN|jul|JUL|aug|AUG|sept|SEPT|oct|OCT|nov|NOV|dec|DEC|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)*[., ]{1,3}\d{4};*",
        r"(?:\d{1,2} )?(?:January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)*[., ]{1,3}\d{4};*"
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        print(matches)
        if matches:
            for index, date in enumerate(matches):
                if date.endswith(';'):
                    matches[index] = date[:-1]
                if len(date) == 2:
                    matches[index] = "'" + date
            return matches
    return []

def is_valid_date(date_str, format):
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False

def split_date(date_str):
    day = month = year = ""
    date_list = [date_str]

    if re.search("\.", date_str):
        date_list = date_str.split(".")
    elif re.search(",", date_str):
        date_list = date_str.split(",")
    elif re.search("/", date_str):
        date_list = date_str.split("/")
    elif re.search("-", date_str):
        date_list = date_str.split("-")
    elif re.search(" ", date_str):
        date_list = date_str.split(" ")

    if len(date_list) == 3:
        day = date_list[0] + "/"
        month = date_list[1] + "/"
        year = date_list[2]
    elif len(date_list) == 2:
        month = date_list[0] + "/"
        year = date_list[1]
    elif len(date_list) == 1:
        year = date_list[0]
        if len(date_list[0]) == 3 and date_list[0][0] == "'":
            year = date_list[0].replace("'", "19")

    if (len(date_list) >= 2) and re.search("i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii", month):
        month = month[:-1]  # Get rid of slash
        numeral_List = re.findall("i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii", month)
        roman_numeral = "".join(numeral_List)
        month = str(roman.fromRoman(roman_numeral.upper())) + "/"
        print(month)
    elif (len(date_list) >= 2) and re.search("[^a-zA-Z]", month):
        month = month[:-1]  # Get rid of slash
        try:
            datetime_object = datetime.strptime(month, "%B")  # Uses the full month name format
        except ValueError:
            try:
                datetime_object = datetime.strptime(month, "%b")  # Uses the abbreviated month name format
            except ValueError:
                return None  # Return None for invalid month names
        month = datetime_object.month + "/"

    date_string = f"{month}{day}{year}"
    return date_string


# essential entity models downloads
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')


if __name__ == '__main__':
    ### BASIC STEP BY STEP:
        # 1. Split OCR output (which I assume is just one big string of \n delimited text
        # 2. Separate said words into categories: parsedList[]
        #     Categories:
        #       [0]CatalogNumber	[1]Specimen_voucher<MGCL #>	[2]Family	[3]Genus<Phoebis>	[4]Species<sennae>	[5]Subspecies<sennae>	[6]Sex<female/male>
        #       [7]Country	[8]State	[9]County	[10]Locality name	[11]Elevation min	[12]Elevation max	[13]Elevation unit	[14]Collectors	[15]Latitude	[16]Longitude
        #       [17]Georeferencing source	[18]Georeferencing precision	[19]Questionable label data [20]Do not publish
        #       [21]Collecting event start	[22]Collecting event end	[23]Date verbatim           *** NOTE: Collecting event start/end are the same, in MM/DD/YYYY format
        #       [24]Remarks public	[25]Remarks private	    [26]Cataloged date***can collect ourselves	[27]Cataloger First [28]Cataloger last***can collect ourselves
        #       [29]Prep type 1 [30]Prep count 1	[31]Prep type 2	    [32]Prep number 2	[33]Prep type 3 [34]Prep number 3	[35]Other record number
        #       [36]Other record source     [37]publication     [38]publication
        # 3. Input parsed list into a CSV format, where the first line is the label names and then the rest of it is the information
        # Consistencies:
        #   - The species name will always be first, followed by UF tag
        #   - Date will always have a year and month - date is tricky, but we can say that if there's 2 periods/commas, there's a day -
        #   if not, and there's only one (or only one space), it's a day
        #   - For questionable label, maybe we can put if the general confidence rating is low, don't publish
        #   - Labels 2, 24 - 39 are not part of the labels!
        # 4. fillna(): Fills NA/NaN values for remainder of dataframe


    ### Can be replaced with output from OCR
    sampleString = "Phoebis sennae\nsennae\nUF\nFLMNH\nMGCL 1163652\nCUBA: GRANMA\nEl Banco, Mpio. Buey Arriba\n1000m, Turquino massif\n'98; L D & J Y Miller\n& L R Hernandez sta. 1994-43\nEX.SA. MAESTRA Allyn Museum Acc. 1994-16"
    print("Sample string:\n", sampleString, "\n")

    listOfStrings = sampleString.split()
    print("List of strings: ", listOfStrings)

### MAKE THIS A FOR LOOP AT SOME POINT ###
    # data is the template list of metadata attributes; there'll be a list for each image
    # Category 0: CatalogNumber can be filled in off the top (#########)
    data = [["#########", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN", "NaN"]]

    # Create the pandas DataFrame: df
    # df will contain the parsed metadata for each image
    df = pd.DataFrame(data, columns=['CatalogNumber', 'Specimen_voucher', 'Family', 'Genus', 'Species', 'Subspecies', 'Sex', 'Country', 'State', 'County', 'Locality name', 'Elevation min', 'Elevation max', 'Elevation unit', 'Collectors', 'Latitude', 'Longitude', 'Georeferencing source', 'Georeferencing precision', 'Questionable label data', 'Do not publish', 'Collecting event start', 'Collecting event end', 'Date verbatim', 'Remarks public', 'Remarks private', 'Cataloged date', 'Cataloger First', 'Cataloger last', 'Prep type 1', 'Prep count 1', 'Prep type 2', 'Prep number 2', 'Prep type 3', 'Prep number 3', 'Other record number', 'Other record source', 'publication', 'publication1'])
    print("df", df)
    currentIndex = 0    # Refers to the current ocr output we're parsing (0 - (n-1)) where n is the number of photos uploaded

    # The order that the categories are filled in is, at first, determined by the order the text is parsed from the photo
    # Some items, (genus, species, and subspecies) always come first
    # Categories 3 - 6: Genus, Species, and Subspecies
    for x in range(3, 6):
        if listOfStrings[currentIndex] != "UF" and listOfStrings[currentIndex] != "FLMNH":
            df.iloc[currentIndex, x] = listOfStrings[0]
            listOfStrings.pop(0)

    listOfStrings.pop(0)    # Get rid of UF
    listOfStrings.pop(0)    # Get rid of FLMNH

    # Category[1]: Specimen Voucher
    voucher = ""
    for x in range(2):
        voucher += listOfStrings[0] + " "
        listOfStrings.pop(0)
    df.loc[currentIndex, 'Specimen_voucher'] = voucher[:-1]     # [:-1] takes care of the extra space

    ### Extracting Location Entities: Categories[7 - 10] ###
    joinedStrings = " ".join(listOfStrings)
    placeEntity = locationtagger.find_locations(text=joinedStrings)

    # Getting all countries
    print("The countries in text : ")
    print(placeEntity.countries)
    df.loc[currentIndex, 'Country'] = ", ".join(placeEntity.countries)

    # Getting all states
    print("The states in text : ")
    print(placeEntity.regions)

    # Getting all cities
    print("The cities in text : ")
    print(placeEntity.cities)

    ### NOW BEGINS THE DATE SAGA ###
    print("\nJoined strings: ", joinedStrings)

    datefinder_output = []#datefinder.find_dates(joinedStrings)

    if not datefinder_output:
        dates_found = find_more_dates(joinedStrings)

        for date_str in dates_found:
            df.loc[currentIndex, 'Date verbatim'] = date_str

            # Categories[21 - 22] (they're the same)
            proper_date = split_date(date_str)
            df.loc[currentIndex, 'Collecting event start'] = proper_date
            df.loc[currentIndex, 'Collecting event end'] = proper_date
    else:
        print("Dates found: ")
        for dates in datefinder_output:
            print(dates)
            # Categories[21 - 22]
            ### NOTE: If we use datefinder, we can't transcribe it verbatim
            df.loc[currentIndex, 'Collecting event start'] = dates.strftime('%m/%d/%Y')
            df.loc[currentIndex, 'Collecting event end'] = dates.strftime('%m/%d/%Y')
            break

    # Category[26]: Date cataloged (current date at time of program running)
    df.loc[currentIndex, 'Cataloged date'] = datetime.now().strftime("%m-%d-%Y")

    print("\n")
    print("Updated listOfStrings: ", listOfStrings)
    print("")
    pd.set_option('display.max_columns', None)
    print("Updated df:\n", df)

    # sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
    # dictionary_path = "newDict2.txt"
    # dictionary_path2 = "bigramDict.txt"
    # eng_Dict = "en-80k.txt"

    # sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, separator="$")
    # sym_spell.load_dictionary(dictionary_path2, term_index=0, count_index=1, separator="$")
    # sym_spell.load_dictionary(eng_Dict, term_index=0, count_index=1)
    # sym_spell.load_bigram_dictionary(dictionary_path2, term_index=0, count_index=1, separator="$")

### DATE TESTS ###
    # MM DD YYYY
    # date_format_1 = "%m.%d.%Y"
    # date_format_2 = "%m,%d,%Y"
    # date_format_3 = "%m/%d/%Y"
    # # MM (Abbreviated or full name), YYYY
    # date_format_4 = "%B, %Y"
    # date_format_5 = "%b., %Y"
    # # MM (Abbreviated or full name) DD, YYYY
    # date_format_6 = "%B %d, %Y"
    # date_format_7 = "%b %d, %Y"
    # # DD MM (Abbreviated or full name), YYYY
    # date_format_8 = "%d %B, %Y"
    # date_format_9 = "%d %b, %Y"
    # # DD MM YYYY
    # date_format_10 = "%d.%m.%Y"
    # date_format_11 = "%d,%m,%Y"
    # date_format_12 = "%d/%m/%Y"
    # # YYYY
    # date_format_13 = "%Y"

    # if is_valid_date(date_str, date_format_1):
    #     print(f"{date_str} is a valid date in the format {date_format_1}")
    # elif is_valid_date(date_str, date_format_2):
    #     print(f"{date_str} is a valid date in the format {date_format_2}")
    # elif is_valid_date(date_str, date_format_3):
    #     print(f"{date_str} is a valid date in the format {date_format_3}")
    # elif is_valid_date(date_str, date_format_4):
    #     print(f"{date_str} is a valid date in the format {date_format_4}")
    # elif is_valid_date(date_str, date_format_5):
    #     print(f"{date_str} is a valid date in the format {date_format_5}")
    # elif is_valid_date(date_str, date_format_6):
    #     print(f"{date_str} is a valid date in the format {date_format_6}")
    # elif is_valid_date(date_str, date_format_7):
    #     print(f"{date_str} is a valid date in the format {date_format_7}")
    # elif is_valid_date(date_str, date_format_8):
    #     print(f"{date_str} is a valid date in the format {date_format_8}")
    # elif is_valid_date(date_str, date_format_9):
    #     print(f"{date_str} is a valid date in the format {date_format_9}")
    # elif is_valid_date(date_str, date_format_10):
    #     print(f"{date_str} is a valid date in the format {date_format_9}")
    # elif is_valid_date(date_str, date_format_11):
    #     print(f"{date_str} is a valid date in the format {date_format_9}")
    # elif is_valid_date(date_str, date_format_12):
    #     print(f"{date_str} is a valid date in the format {date_format_9}")
    # else:
    #     print(f"{date_str} is not a valid date in the tested formats.")
