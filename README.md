# Steam Recommender System

## How to run:

**Disclaimer**: Dataset and libraries are extrememly large so a storage size of about **30GB** is requried

## Backend (First Time Use):
```console
cd Back-End\Methods\ 
python -m venv venv
```

**For a specific version you can also do** 
```console
py -3.xx -m venv venv
```


**Install Libraries**  
```console
pip install -r requirements.txt
```

**Run Server** 
```console
py server.py
```


## Frontend (First Time Use):
**Open a Second Terminal**

```console
cd Front-End/
npm install
npm start
```


# Game Recommender Usage Instructions

1. Click search bar to open list of available games (inputs)
    - Type in the search bar to filter games by name, *and/or*
    - Scroll through the dropdown menu to find games, *and*
    - Select games by clicking on each title.

2. Click outside the dropdown to close the menu

3. Click the "Load Recommendations" button to get recommendations

4. Wait for loading bar to indicate completion (disappear)

5. View results:
    - Interact with each modal to see a curated list of 10 games (sorted by method)
    - View summary table with the recommendations of each model

*Note*: Modals can be opened by clicking on the buttons with each model labelled on it. Modals can be closed by clicking the red exit button on the top right of the display. 