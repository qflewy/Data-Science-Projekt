# Data-Science-Projekt
University Project of 4 Students

### Introduction

Fuel prices in Germany affect a large share of the population, as many people rely on their cars for commuting and everyday life. 
According to DESTATIS, the average person travels about 15.5 km per day by car. Recent geopolitical events have shown how quickly fuel prices can change. 
We analyze fuel prices across Germany and investigate the key factors driving their variation.

### Research Questions
- Intraday patterns in fuel prices
- Speed of fuel price adjustments to oil price changes
- Influence of special location factors on price differences and their stability
- Price differences between brand and non-brand gas stations
- Diesel vs. E10 price anomalies
- Predicting the optimal weekly refueling time
- Impact of proximity to other stations on fuel prices
- Impact of extreme weather events on fuel prices


### Data
##### Tankerkönig - historical fuel prices (Germany)
Provides daily CSV files with fuel prices (E5, E10, Diesel) and gas station data across Germany. 
Data is structured by date: prices/YYYY/MM/YYYY-MM-DD-prices.csv and stations/YYYY/MM/YYYY-MM-DD-stations.csv.

##### Yahoo Finance – Oil Prices (via yfinance)
Provides historical and real-time commodity market data. 
Used in this dashboard to retrieve crude oil price data (e.g. Brent, WTI) as a macroeconomic reference for fuel price analysis.

##### OpenStreetMap – Map Tiles & Geodata
Free, editable geographic database maintained by a global community. 
Used in this dashboard as the base map layer for displaying gas station locations across Germany.

##### Open-Meteo – Weather Data
Free weather API providing historical and forecast data including temperature, precipitation, wind speed, and more. 
Used in this dashboard to correlate weather conditions with fuel consumption patterns.

### Data Pipeline
For Tankerkönig, we cloned the git repo with the historical data and used the provided csv files for further analyses. 
Except two files that were corrupted, we didn't need to do any data cleaning.
For our other source, we just used the respective python libraries (e.g. yfinance) to gather our required data.


### Website
We decided to use Dash for our website, because it provided better performance and customizability compared to Streamlit. 
The website was build modular, such as every page having its own data folder. In addition, every page has its own file and separate figure and callback files.
We used Render to host our website which has proven to be the wrong decision, because of performance issues.

#### How to use the website
Since the Render site has long loading time, it's best to clone the website repo and use the website locally.
On our landing page you find an introduction to our topic, some data fun facts and our research questions. 
On the top right, you can find a menu bar where you can navigate to the pages for each broader topic. 
For example, you can click on "Fuel Up" to have an interactive graph that shows you when the best time to fuel up is. 
You can select a fuel type and give your location.


### Use of LLMs
In our code, we marked the lines created with an LLM explicitly. 
When we worked with an LLM to refine a function, we wrote a comment that this specific function was created with the help of a LLM.
Also, we used codex to make our code pep8 conform.
