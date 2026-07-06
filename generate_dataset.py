"""Generate Crop_recommendation.csv with realistic data (2200 records, 22 crops)."""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Each crop: (N_mean, N_std, P_mean, P_std, K_mean, K_std,
#             temp_mean, temp_std, hum_mean, hum_std,
#             ph_mean, ph_std, rain_mean, rain_std, count)
CROP_PARAMS = {
    'rice':        (80,10, 45,10, 40,10, 23,2,  82,4,  6.5,0.5, 200,30, 100),
    'maize':       (78,10, 48,10, 20,5,  22,3,  65,8,  6.2,0.5, 85,20,  100),
    'chickpea':    (40,8,  68,10, 80,10, 18,3,  16,5,  7.0,0.4, 80,20,  100),
    'kidneybeans': (20,5,  67,10, 20,5,  20,3,  21,4,  5.7,0.4, 105,20, 100),
    'pigeonpeas':  (20,5,  67,10, 20,5,  27,3,  48,8,  5.8,0.4, 149,30, 100),
    'mothbeans':   (21,5,  48,8,  20,5,  28,3,  53,8,  3.5,0.3, 51,15,  100),
    'mungbean':    (20,5,  47,8,  20,5,  28,3,  85,5,  6.7,0.4, 49,15,  100),
    'blackgram':   (40,8,  67,10, 19,5,  30,3,  65,8,  7.0,0.4, 68,15,  100),
    'lentil':      (18,5,  68,10, 19,5,  24,3,  64,8,  6.9,0.4, 46,15,  100),
    'pomegranate': (18,5,  18,5,  40,8,  21,3,  90,5,  6.0,0.4, 107,20, 100),
    'banana':      (100,10,82,10, 50,8,  27,2,  80,5,  6.0,0.4, 105,20, 100),
    'mango':       (20,5,  27,5,  30,5,  31,3,  50,8,  5.8,0.4, 95,20,  100),
    'grapes':      (23,5,  132,10,200,15,23,3,  81,5,  6.0,0.4, 69,15,  100),
    'watermelon':  (99,10, 17,5,  50,8,  25,3,  85,5,  6.5,0.4, 51,15,  100),
    'muskmelon':   (100,10,17,5,  50,8,  28,3,  92,5,  6.5,0.4, 25,10,  100),
    'apple':       (21,5,  134,10,199,15,22,3,  92,5,  5.9,0.4, 113,20, 100),
    'orange':      (20,5,  16,5,  10,3,  22,3,  92,5,  7.0,0.4, 110,20, 100),
    'papaya':      (50,8,  59,8,  50,8,  33,3,  92,5,  6.7,0.4, 143,25, 100),
    'coconut':     (22,5,  16,5,  30,5,  27,3,  94,5,  5.9,0.4, 176,30, 100),
    'cotton':      (118,10,46,8,  43,8,  24,3,  79,5,  6.9,0.4, 80,20,  100),
    'jute':        (78,10, 46,8,  40,8,  24,3,  79,5,  6.7,0.4, 175,30, 100),
    'coffee':      (101,10,28,5,  29,5,  25,3,  58,8,  6.8,0.4, 158,25, 100),
}

rows = []
for crop, p in CROP_PARAMS.items():
    n = p[14]
    N    = np.clip(np.random.normal(p[0],  p[1],  n), 0,   140)
    P    = np.clip(np.random.normal(p[2],  p[3],  n), 5,   145)
    K    = np.clip(np.random.normal(p[4],  p[5],  n), 5,   205)
    temp = np.clip(np.random.normal(p[6],  p[7],  n), 8,   44)
    hum  = np.clip(np.random.normal(p[8],  p[9],  n), 14,  100)
    ph   = np.clip(np.random.normal(p[10], p[11], n), 3.5, 9.9)
    rain = np.clip(np.random.normal(p[12], p[13], n), 20,  300)
    for i in range(n):
        rows.append([round(N[i],2), round(P[i],2), round(K[i],2),
                     round(temp[i],2), round(hum[i],2), round(ph[i],2),
                     round(rain[i],2), crop])

df = pd.DataFrame(rows, columns=['N','P','K','temperature','humidity','ph','rainfall','label'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'Crop_recommendation.csv')
df.to_csv(out, index=False)
print(f"Dataset saved: {out}  shape={df.shape}")
print(df['label'].value_counts())
