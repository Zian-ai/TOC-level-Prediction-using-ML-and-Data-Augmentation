# TOC-level-Prediction-using-ML-and-Data-Augmentation
Prediction of TOC level in a reservoir using Optimized Machine Learning and WGAN data Augmentation. 

**OVERVIEW**
This study explores how future climate change may affect Total Organic Carbon (TOC) levels in key reservoirs by employing advanced machine-learning techniques combined with synthetic data augmentation. Models are trained on historical observations and climate projections to generate reliable forecasts of TOC under different climate change scenarios. The results highlight potential changes in water quality driven by varying climate pathways and demonstrate a robust framework for informed reservoir management in a changing climate.

**METHODOLOGY**
1. **Data Collection**  
   - Gather TOC observations data and Input data (Precipitation, Minimum Temperature, Maximum Temperature and Relative Humidity) from the period of 2011–2024 and collect 12 GCM from CMIP6 climate projections for 2025 to 2100.   

2. **WGAN data augmentation**  
   - Generate synthetic data using a WGAN trained for 3,000 epochs, targeting a synthetic-to-real ratio of 5:1.   
   - Validate the realism of the synthetic data relative to the real data using distributional similarity tests: the Kolmogorov–Smirnov test, the Anderson–Darling test, and the energy distance test.
     
3. **Nine ML Model training & Hyperparamter Optimization using Optuna**  
   - Split the Real data 80/20 (train/test).  
   - Optimize hyperparameters for nine ML algorithms (DT, RF, GB, XGB, SVR, RR, LL, M5, M5-SGB) using Optuna.

4. **TOC level projection of 12 GCM under Climate Change scenario (SSP2-4.5 and SSP5-8.5)**  
   - Deploy best model to forecast TOC level from 2025–2100 under SSP2-4.5 & SSP5-8.5 across 12 GCMs.

5. **Trend analysis**  
   - Detect monotonic changes with Mann–Kendall and quantify slopes with Sen’s estimator.  

**INSTALL REQUIREMENTS**
-Install the required packages: python, tensorflow, numpy, matplotlib packages

**DATA**
-Data will be made available on request. 

**USAGE**
1. To run the project, follow these steps:
2. Ensure you have the necessary dependencies installed.
3. Modify each python file to set the correct paths for your data and model directories.
4. Run each python script to start the training process.

**ABOUT THE AUTHOR**
Christian Joseph Siose, Kangwon National University
Email: christiansiose18@gmail.com
