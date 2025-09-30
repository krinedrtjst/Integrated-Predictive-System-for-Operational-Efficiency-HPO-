-----

# Integrated Predictive System for Operational Efficiency (HPO)

**A data science project to enhance grid stability by optimizing hydroelectric power dispatch in response to intermittent renewable energy sources.**

This project, developed for the InovaONS challenge, demonstrates an end-to-end MLOps solution that leverages machine learning and mathematical optimization to ensure the stability of Brazil's National Interconnected System (SIN).

-----

## 📋 Table of Contents

  * [Overview](https://www.google.com/search?q=%23-overview)
  * [Core Problem](https://www.google.com/search?q=%23-core-problem)
  * [The Solution: HPO](https://www.google.com/search?q=%23-the-solution-hpo)
  * [Project Pipeline](https://www.google.com/search?q=%23-project-pipeline)
  * [Key Results](https://www.google.com/search?q=%23-key-results)
  * [Technology Stack](https://www.google.com/search?q=%23-technology-stack)
  * [How to Run](https://www.google.com/search?q=%23-how-to-run)
  * [Future Improvements](https://www.google.com/search?q=%23-future-improvements)

-----

## 🔭 Overview

The increasing integration of intermittent energy sources (like wind and solar) poses a significant challenge to the stability of the electrical grid. This project, the **Hydropower Optimizer (HPO)**, provides a robust data-driven framework to mitigate this volatility. It uses a predictive model to forecast energy demand and an optimization model to calculate the ideal real-time dispatch from hydroelectric plants, effectively turning them into a flexible reserve to balance the grid.

-----

## 🌪️ Core Problem

  * **Intermittency**: Wind and solar power generation can fluctuate dramatically, creating sudden "ramps" (surges or drops) in the energy supply.
  * **Grid Instability**: These ramps can destabilize the grid's **frequency** and **voltage**, risking power quality and security.
  * **Operational Constraints**: Hydroelectric plants must operate within strict **environmental and physical limits**, such as minimum water flow and maximum generation capacity.

-----

## ✨ The Solution: HPO

The HPO system is an integrated pipeline that transforms raw operational data into optimized, real-time dispatch commands.

1.  🧠 **Prediction First**: It uses a **Long Short-Term Memory (LSTM)** neural network to accurately forecast the total grid load for the next hour based on historical data, weather patterns, and operational variables.
2.  ⚙️ **Optimize Second**: It then feeds this prediction into a **Linear Programming (PuLP)** model. This model calculates the optimal amount of hydroelectric power to generate, minimizing any gap between supply and demand while respecting all operational constraints.

This turns hydroelectric power into a precise, data-driven tool for ensuring grid stability.

-----

## 🚀 Project Pipeline

The project is structured as a series of Jupyter Notebooks that simulate a complete MLOps workflow:

1.  **ETL and Data Integration**: Extracts data from various sources (INMET, ONS, CCEE), cleans it, and merges it into a unified dataset. The pipeline is built to be resilient, using simulated data if an API fails.
2.  **Predictive Modeling (LSTM)**: A TensorFlow/Keras LSTM model is trained on the integrated data to predict `carga_mw_subsistema` (subsystem load).
3.  **Rigorous Validation**: The model's accuracy is validated, achieving a **Root Mean Squared Error (RMSE) of 9.33 MW**, which is a very low error margin for a national grid.
4.  **Advanced Optimization (PuLP)**: A formal optimization model calculates the ideal hourly dispatch, ensuring it meets the predicted demand while staying within the plant's minimum (`MIN_GERACAO_OBRIGATORIA`) and maximum (`MAX_GERACAO_CAPACIDADE`) generation limits.
5.  **MLOps and Deployment Simulation**: The final step simulates deploying the trained model and scaler. A function `predict_and_optimize_realtime` mimics a real-world API endpoint that would provide the optimized dispatch value to the power plant's operational system.

-----

## 📈 Key Results

  * **High-Accuracy Prediction**: The LSTM model achieved a low **RMSE of 9.33 MW** on the test set, demonstrating its reliability for operational planning.
  * **Optimal Dispatch**: The PuLP optimization model successfully identified the ideal hydroelectric generation required to meet demand, with a final error of **0.00 MW** in the simulation.
  * **End-to-End MLOps Workflow**: The project successfully simulates the entire lifecycle of a machine learning model, from data ingestion to a production-ready API concept.

-----

## 🛠️ Technology Stack

  * **Data Manipulation**: **Pandas**, **NumPy**
  * **Machine Learning**: **TensorFlow**, **Keras**, **Scikit-learn**
  * **Optimization**: **PuLP**
  * **MLOps (Serialization)**: **Joblib**
  * **Environment**: **Python 3.13**, Jupyter Notebook

-----

## ▶️ How to Run

1.  **Clone the repository.**
2.  **Set up the environment**: Ensure you have Python and the required libraries installed. You can install them via `pip`:
    ```bash
    pip install pandas numpy tensorflow scikit-learn pulp joblib
    ```
3.  **Place your data**: Make sure the `dados_hpo_integrados_*.csv` file is located at the path specified in the notebooks.
4.  **Run the notebooks sequentially**:
      * `01_ETL_Integration.ipynb`
      * `02_Predictive_Modeling_LSTM.ipynb`
      * `03_Optimization_PuLP.ipynb`
      * `04_MLOps_Deployment.ipynb`

-----

## 🔮 Future Improvements

  * **Hyperparameter Tuning**: Implement automated tuning with **Keras Tuner** to further reduce the model's RMSE by optimizing parameters like LSTM units, time steps, and learning rate.
  * **CI/CD Pipeline**: Develop a full CI/CD pipeline to automate model retraining, testing, and deployment when new data becomes available or model performance degrades.
  * **Real-Time Dashboard**: Create a visualization dashboard to monitor key metrics like predicted vs. actual load, optimized vs. actual dispatch, and grid frequency in real-time.
