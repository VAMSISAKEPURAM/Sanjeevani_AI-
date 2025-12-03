import pandas as pd
import numpy as np
import random

# ------------------------------------------------------------
# Crop–Disease Mapping
# ------------------------------------------------------------
disease_map = {
    "Tomato": [
        "Target Spot",
        "Tomato mosaic virus",
        "Tomato YellowLeaf Curl Virus",
        "Bacterial spot",
        "Early blight",
        "Late blight",
        "Leaf Mold",
        "Septoria leaf spot",
        "Spider mites Two spotted spider mite"
    ],
    "Potato": ["Early blight", "Late blight"],
    "Paddy": ["Blast", "Brown spot"],
    "Groundnut": ["Rust", "Leaf spot"]
}

# ------------------------------------------------------------
# Distribution Plan
# ------------------------------------------------------------
distribution = {
    "Tomato": 0.50,
    "Potato": 0.20,
    "Paddy": 0.15,
    "Groundnut": 0.15
}

total_samples = 5000
time_slots = ["Morning", "Afternoon", "Evening", "Night"]
dataset = []

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def generate_sample(label):
    """Generate weather data based on 'Good' or 'Bad' label"""
    if label == "Good":
        temp = round(random.uniform(20, 32), 1)
        humidity = round(random.uniform(50, 80), 1)
        wind = round(random.uniform(2, 10), 1)
        rain = 0.0
        tod = random.choice(["Morning", "Evening"])
    else:
        temp = round(random.uniform(18, 40), 1)
        humidity = round(random.uniform(30, 95), 1)
        wind = round(random.uniform(5, 15), 1)
        rain = round(random.choice([0.0, 0.0, random.uniform(0.1, 5)]), 1)
        tod = random.choice(["Afternoon", "Night"])
    return temp, humidity, wind, rain, tod

# ------------------------------------------------------------
# Generate Balanced Data
# ------------------------------------------------------------
for crop, perc in distribution.items():
    crop_total = int(total_samples * perc)
    diseases = disease_map[crop]
    samples_per_disease = crop_total // len(diseases)
    good_per_disease = samples_per_disease // 2
    bad_per_disease = samples_per_disease - good_per_disease

    for disease in diseases:
        # Generate GOOD samples
        for _ in range(good_per_disease):
            t, h, w, r, tod = generate_sample("Good")
            dataset.append([t, h, w, r, tod, crop, disease, "Good"])
        # Generate BAD samples
        for _ in range(bad_per_disease):
            t, h, w, r, tod = generate_sample("Bad")
            dataset.append([t, h, w, r, tod, crop, disease, "Bad"])

# ------------------------------------------------------------
# Create DataFrame and Save
# ------------------------------------------------------------
df = pd.DataFrame(dataset, columns=[
    "temperature", "humidity", "wind_speed", "rainfall",
    "time_of_day", "crop_type", "disease_type", "label"
])

df.to_csv("balanced_crop_pesticide_dataset_test.csv", index=False)

# Summary
print("✅ Dataset created successfully!\n")
print(f"Total Samples: {len(df)}")
print(df['crop_type'].value_counts())
print("\nLabel Distribution:\n", df['label'].value_counts())
print("\nPreview:")
print(df.sample(10))
