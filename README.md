# NYC Air Quality & Public Health Correlation

#### By Stephanie Pavon (sp8089), Xavier Beltran (xb2101), Philip Vetsas (pmv264), Dongting Gao (dg4528)

## Structure

All code is contained in `BigdataProject.ipynb`. All resources necessary to run the code are contained in the `resources` directory.

## Data

The project makes use of two datasets:

1. NYC [ Air Quality ]( https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r/about_data )

This file should be named `Air_Quality_20251201.csv` and be placed in the `resources` folder.

2. [ Hospital Inpatient Discharges ]( https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/gnzp-ekau/about_data )

This file should be named `Hospital_Inpatient_Discharges_2016.csv` and be placed in the `resources` folder.

There are a few other files in the `resources` folder: some GeoJSONs. The GeoJSONs are necessary to run the geospatial visualizations, and must be included as well for all the code to work.

## Requirements

The code was developed in Python 3.12. 

To install the necessary packages, run the following command:

`pip install -r requirements.txt`

Dependencies of note include:

- numpy
- pandas
- matplotlib
- geopandas

## Running the code

To run the code, open the `BigdataProject.ipynb` file in in a Jupyter Notebook editor, like VS Code, run `jupyter notebook BigDataProject.ipynb` and view in your web browser, or add the project files to Google Drive and run with Colab.
