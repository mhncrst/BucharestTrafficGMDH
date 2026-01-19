# Traffic Prediction Project using GMDH

## Existing Research

### Estimating Missing Weather Data for Agricultural Simulations Using Group Method of Data Handling  
*M. C. Acock, Ya. A. Pachepsky, Journal of Applied Meteorology (1988-2005)*

- Addresses the problem of missing weather data.  
- Hypothesis: missing daily weather variables can be estimated using measurements from the days immediately before and after the missing day.  
- Dataset: daily solar radiation, max/min temperature, and wind run collected in Stoneville, Mississippi, from May–September, 1982–1992.  
- Reason for using GMDH: relationships among weather variables across multiple days can be highly nonlinear.  
- Observations: the day immediately following the missing day provided the most informative input; accuracy worsened with more than one missing day.  
- Results: GMDH clearly outperformed climatology and persistence methods, achieving lower errors and stronger correlations with actual weather variables.  

---

### Polynomial Theory of Complex Systems  
*A.G. Ivakhnenko*

- Complex systems can be effectively modeled using polynomial functions, structured hierarchically to capture nonlinear dependencies.  
- Formalizes GMDH as a self-organizing approach that builds polynomial regression networks layer by layer, automatically selecting the most relevant variables and terms.  
- Only the best-performing polynomials are retained with each layer.  
- Considered the cornerstone of modern GMDH research.  

---

## Data Used

### TomTom Traffic Flow API

Variables obtained from API calls:
- `currentSpeed` – observed speed on segment  
- `freeFlowSpeed` – normal speed in no traffic conditions  
- `currentTravelTime` – real travel time  
- `freeFlowTravelTime` – ideal travel time  
- `confidence` – estimation confidence level  
- `roadClosure` – closure/blockage indicator  
- `coordinates` – latitude and longitude for each segment  

Note: Some variables are highly correlated, so using all of them might confuse the selection process.

---

## Our Approach

Start with core explanatory variables:
- `currentSpeed`  
- `freeFlowSpeed`  
- `time of day`  

Later improvements may include:
- `freeFlowTravelTime`  
- `roadClosure`  

---

## Model & Necessary Resources

- **Model:** GMDH (Group Method of Data Handling)  
- **Hardware:** CPU (low processing demands), RAM 4–8 GB (thousands of observations × several variables)  
- **Storage:** a few hundred MB for JSON/CSV traffic data files and intermediate results  
- **Secondary model for comparison:** Multiple Linear Regression  
- **Evaluation criteria:** accuracy, comparing current speed, free flow speed, current time, free flow time with actual values  

---

## Methods of Evaluation

We will use two main evaluation methods:

- **Mean Absolute Error (MAE):** shows the average accuracy of the model, treating all errors linearly.  
- **Root Mean Square Error (RMSE):** similar to MAE, but penalizes large errors more strongly because of squaring.  

Combining both should give an accurate approximation of model efficiency. Adding another metric such as **R²** to test explanatory power is not excluded.  
