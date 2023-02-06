# Maternal Early Warning System
**This project is supervised by the Duke Institute for Healthcare Innovation and Duke University Hospital System. The project builds a perinatal early warning system, which predicts the following eight outcomes of Severe Maternal Morbidity (SMM) using time-series data of perinatal encounters. Outcomes include Hemorrhage, Sepsis, Eclampsia, Embolism, Acute Renal Failure, Acute Respiratory Distress Syndrome, and Acute Heart Failure, and Disseminated Intravascular Coagulation.**

## **About the Project**

The maternal mortality rate in the United States has increased from 9.9 per 100,000 births in 1999 to 26.4 per 100,000 births in 2015, making it one of the highest maternal mortality rates among industrialized nations. This highest risk is among non-Hispanic black women who had a mortality rate of 46 per 100,000 births in 2014. More common, however, is severe maternal morbidity (SMM), which has been shown to be associated with increased risk for maternal mortality. The Center for Disease Control and Prevention defines SMM by 18 indicators commonly identified as peripartum complications. Severe maternal morbidity, like maternal mortality, has also significantly increased over the last 2 decades, from 49.5 per 10,000 births in 1993 to 144.0 per 10,000 births in 2014.

At Duke University Hospital (DUH), ~80% of our patients are considered high risk, and therefore at increased risk for SMM. Currently, there are no systems to quickly identify women experiencing SMM. While there are a few commercial prediction systems available, none have been studied or shown to improve outcomes. Delay in diagnosis has been identified as a factor in leading to SMM and maternal mortality. In non-pregnant patients, there are a plethora of early warning systems and clinical deterioration algorithms useful in identifying patients in need of escalation of care. However, the unique physiology of pregnancy makes adoption of these algorithms in the obstetric population difficult. Often well-validated clinical deterioration systems in non-pregnant patients excluded pregnant patients in the development process and perform poorly when adapted to the obstetric population making them ineffective.

DIHI created a trans-disciplinary team of obstetric physicians, and technical staff to build an automated Maternal Early Warning System for 8 clinician-prioritized cases of SMM, to be used as an alert dashboard. 

## **About my Role**

I have authored all code in this repository, and have been responsible for all dataset building and modeling for the project, supervised by Mark Sendak and Michael Gao (DIHI). Note: This repository is in progress, and is not complete.  

## **Included in this Repository**

### Dataset Pipeline 
The datasets used for these predictions are hour-by-hour input data including measures of patient flowsheets, lab orders/collections/results, demographics, prenatal visits, and comorbidities. 


### Modeling Experiments 
A significant challenge for this task is data imbalance, driven primarily by two factors: 1) SMMs occur for very few patients 2) Because we are making hourly predictions of SMM occurance in the following N hours, even these patients only have positive values for the N hours of an encounter prior to a SMM condition being met, and negative for all encounter hours prior. 

To help mitigate these effects, I designed the following experiments: 

1. Undersampling the minority class 
2. Weighting minority samples 
3. Weighting minority samples with a distribution 

### Automated Documentation and Evaluation 
Automated documentation and evaluation code, including analysis of population subgroup performance

### Example Runs 

## Repository Guide 




