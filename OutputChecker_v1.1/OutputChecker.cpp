#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
using namespace std;

namespace fs = filesystem;


vector<string> parseCSV(string path) {

    //return input.substr(1, input.length() - 2);
    //cout << path;
    vector<string> strings;

    ifstream infile(path, std::ifstream::binary);

    string lineToRead;
    

    size_t beginFrom;
    size_t location;


    string entry;


    while (getline(infile, lineToRead)) {
        //cout << lineToRead << endl;
        istringstream line(lineToRead);

        while (getline(line, entry, ',')) {
            strings.push_back(entry);
            //cout << "String: " << entry << endl;
        }

    }


    infile.close();
    return strings;
}

int main()
{
    
    string path = "./CorrectValues";
    vector<string> fileNames;
    string solutionsFileName;
    for (const auto& entry : fs::directory_iterator(path)) {

        solutionsFileName = entry.path().string();

    }
    //cout << solutionsFileName;
    path = "./OutputFiles";
    for (const auto& entry : fs::directory_iterator(path)) {
 
        //string tempPath = entry.path().string();
 
        //fileNames.push_back(path.append(tempPath));
        fileNames.push_back(entry.path().string());
    }


    vector<string> correctValues = parseCSV(solutionsFileName);
    vector<string> tempValues;
    int numErrors = 0;
    int numMatched = 0;
    int numAligned = 0;
    bool found = false;

    for (int i = 0; i < fileNames.size(); i++) {

        numErrors = 0;
        numMatched = 0;
        numAligned = 0;

        tempValues = parseCSV(fileNames[i]);

        for (int j = 0; j < tempValues.size(); j++) {
            found = false;
            if ((tempValues[j].compare("") == 0) && !(correctValues[j].compare("") == 0)) {
                continue;
            }
            //Check if term appears in list
            //int cnt = count(correctValues.begin(), correctValues.end(), tempValues[j]);
            //cout << "String: " << tempValues[j] << endl;

            if (tempValues[j].compare(correctValues[j]) == 0) {
                numMatched++;
                numAligned++;
                continue;
            }
            for (int k = 0; k < correctValues.size(); k++) {
                if (tempValues[j].compare(correctValues[k]) == 0) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                numErrors++;
            }
            else {
                numMatched++;
            }
            
        }
        cout << "Printing stats for output file  " << fileNames[i] << endl << endl;

        cout << "Percentage of correct words: " << fixed << setprecision(2) << (static_cast<double>(numMatched * 100)/correctValues.size()) << "%" << endl;
        cout << "Number of errors: " << (numErrors) << endl;
        cout << "Error Rate: " << fixed << setprecision(2) << (static_cast<double>(numErrors * 100)/correctValues.size()) << "%" << endl;
        cout << "Percentage of correct words in correct locations: " << fixed << setprecision(2) << (static_cast<double>(numAligned * 100)/correctValues.size()) << "%" << endl << endl;
    }


    return 0;

}
