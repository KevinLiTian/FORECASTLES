# FORECASTLES

## OVERVIEW OF AIM
We used *time series* models in an attempt to model:
1. the number of service user counts over time per shelter
2. the number of service user counts over time in Toronto neighbourhoods with shelters

## OVERVIEW OF FILES
`timeseries_SARIMA.ipynb` is a comprehensive approach to analyzing shelter occupancy data using the SARIMA model. This model integrates data from multiple years to analyze and forecast overall shelter occupancy trends. It involves data preprocessing, including aggregation of daily data, and applies the SARIMA model to predict total service user count. This analysis provides a macro-level view of shelter occupancy, identifying broad patterns and trends over time. 

`timeseries_SARIMA_per_location.ipynb` notebook takes a more granular approach. It focuses on location-specific analysis, applying the SARIMA model to individual shelters based on location IDs. This method allows for a detailed understanding of occupancy trends at each shelter across different locations. Together, these notebooks offer a dual perspective on shelter occupancy analysis – one providing an overarching view and the other offering detailed, location-specific insights.

`timeseries_decompositional.ipynb` notebook demonstrates a detailed approach to analyzing and forecasting shelter occupancy data using the Prophet modeling framework. The data is decomposed into individual location datasets based on their unique location IDs. Each location's data undergoes a time series analysis using Prophet, a tool well-suited for handling time series data with its logistic growth modeling. The model incorporates yearly, weekly, and daily seasonality to accurately capture the occupancy patterns at each shelter. The process includes creating future time frames for predictions, allowing for a year-long forecast for each location. This approach enables the model to provide specific forecasts for each shelter, taking into account their unique occupancy trends and variations.

`timeseries_SARIMAX_neighbour.ipynb` : an analysis of the service user counts in neighbourhoods using the SARIMA model. This notebook includes the initial preprocessing of the data, creation of functions to perform data visualisation, applying SARIMA modelling and model evaluations. The SARIMA modelling used for this analysis uses library `pmdarima: ARIMA estimators for Python` which is a package that makes available for Python several ARIMA modelling functions. 

## ACKNOWLEDGEMENTS
These were developed as part of RBC Borealis AI - Let's Solve It Mentorship Programme. 

## CITATIONS
Artley, Brendan. “Time Series Forecasting with ARIMA , SARIMA and SARIMAX.” Medium, 12 May 2022, towardsdatascience.com/time-series-forecasting-with-arima-sarima-and-sarimax-ee61099e78f6.

Smith, Taylor G., et al. pmdarima: ARIMA estimators for Python, 2017-, http://www.alkaline-ml.com/pmdarima.

# Forecasting Shelter Space Needs in Toronto

## Introduction
Welcome to our project, led by the Homies team, focusing on forecasting shelter space needs in Toronto.

### Team Members
- [Kimlin Chin](https://github.com/KCtt457)
- [Kevin Li](https://github.com/KevinLiTian)
- [Michelle Lin](https://github.com/michellemlin)
- [Abhijoy Mandal](https://github.com/Abhijoy-Mandal)
- [Donna Qi](https://github.com/donnaqi)

## Homelessness in Toronto
"Homelessness is a profound assault on dignity, social inclusion and the right to life" ~ United Nations. In Toronto, homelessness has been declared a city emergency, with over 10,000 people affected daily.

## Motivation
Our goal is to optimize shelter usage in Toronto and increase access, as many are turned away daily despite shelters not being at full capacity.

## Project Aims
1. Predict service user counts for existing shelters to optimize resource allocation.
2. Forecast demand in neighborhoods without shelters to guide new shelter construction.

## Data

- **City of Toronto’s Open Data Portal:**
  - [Daily Shelter & Overnight Service Occupancy & Capacity dataset](https://open.toronto.ca/dataset/daily-shelter-overnight-service-occupancy-capacity/)
  - [Central Intake calls dataset](https://open.toronto.ca/dataset/central-intake-calls/)
  - [Neighbourhood Profiles](https://open.toronto.ca/dataset/neighbourhood-profiles/)
  - [Neighbourhood GEOJSON](https://open.toronto.ca/dataset/neighbourhoods/)

## Map Visualization - Demo
We developed a Shiny app to visualize predictions, demonstrating user count forecasts for various shelters and neighborhoods in Toronto.

## Limitations and Future Directions
- Expansion of the model beyond Toronto.
- Integration of more stable and diverse data sets.
- Focusing on demographic-specific trends in homelessness.
- Investigating the relationship between housing prices and homelessness.

## Conclusion
Our model is a significant step towards data-driven management of shelter spaces, aiding in decision-making for budgeting and planning for future years. It's a proactive tool for addressing the growing needs of the homeless population in Toronto.
