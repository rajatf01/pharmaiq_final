# PharmaIQ
PharmaIQ is an explainable inventory intelligence system for small and medium-sized pharmaceutical enterprises (SMEs). It was developed as a Master's project (ISE 298) at San José State University under the supervision of Dr. Behin Elahi.

## Overview
Pharmaceutical SMEs must keep medicines in stock while limiting losses from expiry and supplier disruption, and many lack access to enterprise planning software. PharmaIQ addresses this gap with four linked analyses: demand forecasting, expiry-risk classification, supplier risk scoring, and order-quantity optimization. Results appear in a dashboard, and each recommendation includes an explanation [specify method, e.g., SHAP feature attributions] so that planners can review the reasoning behind it.

## System Components

| Component | Method |
|---|---|
| Demand forecasting | Random Forest regression |
| Expiry-risk classification | Gradient Boosting classifier |
| Supplier risk assessment | Rule-based scoring |
| Order-quantity decisions | Economic Order Quantity (EOQ) model |
| Financial analysis | Cost comparison of baseline and recommended policies |
| User interface | Streamlit dashboard (six tabs) |

## Validation
The system was evaluated on 350 SKUs and 10 suppliers. Under this dataset, the recommended policy yielded a projected 53.3% reduction in inventory cost compared with [define baseline]. This figure comes from a simulated evaluation and has not been tested in a live pharmacy or distribution setting.

## Installation
[Python version, clone command, dependency installation command]

## Usage
[Command to launch the dashboard, plus the required CSV format]

## Repository Structure
[Directory listing with one-line descriptions]
