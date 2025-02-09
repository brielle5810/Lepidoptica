#include <iostream>
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
 
        string tempPath = entry.path().string();
 
        fileNames.push_back(path.append(tempPath));
    }


    vector<string> correctValues = parseCSV(solutionsFileName);
    vector<string> tempValues;
    int numErrors = 0;
    int numMatched = 0;
    int numAligned = 0;

    for (int i = 0; i < fileNames.size(); i++) {

        numErrors = 0;
        numMatched = 0;
        numAligned = 0;

        tempValues = parseCSV(solutionsFileName);

        for (int j = 0; j < tempValues.size(); j++) {
            if (tempValues[j] == "") {
                continue;
            }
            //Check if term appears in list
            int cnt = count(correctValues.begin(), correctValues.end(), tempValues[j]);

            if (tempValues[j] == correctValues[j]) {
                numMatched++;
                numAligned++;
            }
            else if (cnt == 0) {
                numErrors++;
            }
            else {
                numMatched++;
            }
            
        }
        cout << "Printing stats for output file  " << fileNames[i] << endl << endl;

        cout << "Percentage of correct words: " << (numMatched/correctValues.size()) << endl;
        cout << "Number of errors: " << (numErrors) << endl;
        cout << "Error Rate: " << (numErrors/correctValues.size()) << endl;
        cout << "Percentage of correct words in correct locations: " << (numAligned/correctValues.size()) << endl << endl;
    }


    return 0;

}
