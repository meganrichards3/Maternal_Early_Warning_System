import pandas as pd 

if __name__ == '__main__': 
    df = pd.DataFrame({
        'ENCOUNTER_ID': [1,1,1,2,2,3,3,3,3], 
        'PATIENT_ID': [0, 0, 0, 1, 1, 3, 3, 3, 3], 
        'Hour': [1,2,3,1,2,1,2,3,4],
        'Heart_Rate': [60,65,70,67,72,65,70,67,72],  
        'Label': [0,0,1,0,0,0,0,0,0]
    })
    df.to_csv("data/fake_dataset.csv")