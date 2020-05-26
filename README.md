# Disaster Response Pipeline Project
This project is the development of web application which classifies disaster related text messages to the appropriate tags and categories, using ETL and machine learning techniques and pipelines.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files and Folders Description
1. app/templates: This folder contains the html files needed to visualize web app.
2. data/* : This folder contains data used for training the classifier ('disaster_categories.csv' and 'disaster_messages.csv'), the cleaned database 'DisasterResponse.db' and the 'process_data.py' script which cleans and prepares input data for training.
3. models/* : This folder contains "train_classifier.py" script, which is run to start training with 2 command line arguments: database location and output model filename.
4. app/run.py: This script runs flask app on local server, which classify text inputs to multiple labels.


Optional: You can use "requirements.txt" to pip install all libraries needed to run this project.
