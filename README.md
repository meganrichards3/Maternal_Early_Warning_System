# Maternal_Early_Warning_System
**This project is supervised by the Duke Institute for Healthcare Innovation and Duke University Hospital System. The project builds a perinatal early warning system, which predicts the following eight outcomes of Severe Maternal Morbidity (SMM) using time-series data of perinatal encounters. Outcomes include Hemorrhage, Sepsis, Eclampsia, Embolism, Acute Renal Failure, Acute Respiratory Distress Syndrome, and Acute Heart Failure, and Disseminated Intravascular Coagulation.**

**About the Project**

The maternal mortality rate in the United States has increased from 9.9 per 100,000 births in 1999 to 26.4 per 100,000 births in 2015, making it one of the highest maternal mortality rates among industrialized nations. This highest risk is among non-Hispanic black women who had a mortality rate of 46 per 100,000 births in 2014. More common, however, is severe maternal morbidity (SMM), which has been shown to be associated with increased risk for maternal mortality. The Center for Disease Control and Prevention defines SMM by 18 indicators commonly identified as peripartum complications. Severe maternal morbidity, like maternal mortality, has also significantly increased over the last 2 decades, from 49.5 per 10,000 births in 1993 to 144.0 per 10,000 births in 2014.

At Duke University Hospital (DUH), ~80% of our patients are considered high risk, and therefore at increased risk for SMM. Currently, there are no systems to quickly identify women experiencing SMM. While there are a few commercial prediction systems available, none have been studied or shown to improve outcomes. Delay in diagnosis has been identified as a factor in leading to SMM and maternal mortality. In non-pregnant patients, there are a plethora of early warning systems and clinical deterioration algorithms useful in identifying patients in need of escalation of care. However, the unique physiology of pregnancy makes adoption of these algorithms in the obstetric population difficult. Often well-validated clinical deterioration systems in non-pregnant patients excluded pregnant patients in the development process and perform poorly when adapted to the obstetric population making them ineffective.

DIHI created a trans-disciplinary team of obstetric physicians, and technical staff to build an automated Maternal Early Warning System for 8 clinician-prioritized cases of SMM, to be used as an alert dashboard. 

**About my Role**

I have authored all code in this repository, and have been responsible for all dataset building and modeling for the project, supervised by Mark Sendak and Michael Gao (DIHI). Phenotype definitions were created by the following clinical members of the MEWS team: , and the project has been managed by Will Knechtle. 

**Included in this Repository**

This respository includes code used to build:

1. Time-specific indicators for the 8 phenotypes 
2. Time-series datasets processed from the following portions of the EHR: patient flowsheets, problem list data, snomed codes, prenatal visit information, and demographic information. 
3. Modeling experiments for gradient-boosted trees and recurrant neural networks
4. Automated documentation and evaluation code, including analysis of population subgroup performance



