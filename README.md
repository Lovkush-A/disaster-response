# Disaster Response Pipeline Project

### Summary
An ETL and ML pipeline is created to classify text communications, during a natural disaster, into various categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3a. Go to http://0.0.0.0:3001/
3b. On udacity, go to https://view6914b2f4-3001.udacity-student-workspaces.com/


### Directory structure

```
app - contains files to create the webapp
data - contains raw and processed data, and python file to do processing
models - contains python file to do modelling and saved models in form of pkl files
```



