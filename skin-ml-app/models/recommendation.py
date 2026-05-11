import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Skin_Type_dataset.csv')

LABEL_MAPS = {
    'Gender':         {'Female': 0, 'Male': 1},
    'Hydration_Level':{'High': 0, 'Low': 1, 'Medium': 2},
    'Oil_Level':      {'High': 0, 'Low': 1, 'Medium': 2},
    'Sensitivity':    {'High': 0, 'Low': 1, 'Medium': 2},
}
FEATURE_COLS = ['Age', 'Gender_enc', 'Hydration_enc', 'Oil_enc', 'Sensitivity_enc', 'Humidity', 'Temperature']

SKINCARE_ADVICE = {
    'Dry': {
        'description': 'Your skin lacks sufficient moisture and may feel tight or flaky.',
        'routine': [
            'Gentle cream cleanser (avoid foaming)',
            'Alcohol-free hydrating toner',
            'Hyaluronic acid serum (2 layers)',
            'Rich moisturizing cream',
            'SPF 30+ broad-spectrum sunscreen'
        ],
        'ingredients_to_use': ['Hyaluronic Acid', 'Ceramides', 'Glycerin', 'Shea Butter', 'Squalane', 'Niacinamide'],
        'ingredients_to_avoid': ['Alcohol Denat', 'Harsh Sulfates (SLS)', 'Strong Fragrances', 'Benzoyl Peroxide', 'Salicylic Acid (high %)'],
        'tips': [
            'Apply moisturizer on slightly damp skin to lock in moisture',
            'Use a humidifier in dry environments',
            'Avoid hot showers – use lukewarm water',
            'Drink at least 8 glasses of water daily',
            'Use overnight sleeping masks 2-3x per week'
        ],
        'color': '#4a90e2',
        'icon': '💧'
    },
    'Oily': {
        'description': 'Your skin produces excess sebum, leading to shine and enlarged pores.',
        'routine': [
            'Salicylic acid gel cleanser (BHA)',
            'Alcohol-free astringent toner',
            'Niacinamide + Zinc serum',
            'Oil-free lightweight moisturizer',
            'Mattifying SPF 30+ sunscreen'
        ],
        'ingredients_to_use': ['Salicylic Acid (BHA)', 'Niacinamide', 'Zinc PCA', 'Green Tea Extract', 'Retinol', 'AHA'],
        'ingredients_to_avoid': ['Heavy Oils (Coconut)', 'Petrolatum', 'Thick Creams', 'Isopropyl Myristate'],
        'tips': [
            'Do not over-cleanse – max 2x per day',
            'Use oil-blotting papers throughout the day',
            'Apply a clay mask 1-2x per week',
            'Use non-comedogenic products only',
            'Keep hands away from your face'
        ],
        'color': '#f5a623',
        'icon': '✨'
    },
    'Normal': {
        'description': 'Your skin is well-balanced with good moisture levels and few imperfections.',
        'routine': [
            'Gentle foaming cleanser',
            'Balancing hydrating toner',
            'Vitamin C antioxidant serum (AM)',
            'Lightweight moisturizer',
            'SPF 30+ sunscreen (AM)'
        ],
        'ingredients_to_use': ['Vitamin C', 'Retinol', 'Peptides', 'Light AHA/BHA', 'Antioxidants', 'SPF'],
        'ingredients_to_avoid': [],
        'tips': [
            'Focus on prevention and anti-aging',
            'Maintain your balanced routine consistently',
            'Annual skin check-up recommended',
            'Adjust products seasonally',
            'Stay consistent with SPF'
        ],
        'color': '#7ed321',
        'icon': '🌿'
    },
    'Combination': {
        'description': 'Your skin has an oily T-zone (forehead, nose, chin) with normal or dry cheeks.',
        'routine': [
            'Balancing gel cleanser',
            'Hydrating toner (avoid alcohol)',
            'Niacinamide serum (all zones)',
            'Gel moisturizer on T-zone / Cream on dry areas',
            'SPF 30+ sunscreen'
        ],
        'ingredients_to_use': ['Niacinamide', 'Hyaluronic Acid', 'Salicylic Acid (T-zone)', 'Light Plant Oils (cheeks)'],
        'ingredients_to_avoid': ['Heavy creams on T-zone', 'Harsh stripping cleansers', 'Too-rich oils all over'],
        'tips': [
            'Multi-mask technique: clay on T-zone, hydrating on cheeks',
            'Use lighter products on forehead and nose',
            'Apply richer creams only to dry areas',
            'Consider zone-specific serums',
            'Blot T-zone without touching dry areas'
        ],
        'color': '#9b59b6',
        'icon': '⚖️'
    }
}

def encode_input(inp):
    return [
        float(inp.get('Age', 30)),
        LABEL_MAPS['Gender'].get(inp.get('Gender', 'Female'), 0),
        LABEL_MAPS['Hydration_Level'].get(inp.get('Hydration_Level', 'Medium'), 2),
        LABEL_MAPS['Oil_Level'].get(inp.get('Oil_Level', 'Medium'), 2),
        LABEL_MAPS['Sensitivity'].get(inp.get('Sensitivity', 'Low'), 1),
        float(inp.get('Humidity', 50.0)),
        float(inp.get('Temperature', 20.0)),
    ]

def get_recommendations(input_data, n=5):
    df = pd.read_csv(DATA_PATH)
    df['Gender_enc']      = df['Gender'].map(LABEL_MAPS['Gender'])
    df['Hydration_enc']   = df['Hydration_Level'].map(LABEL_MAPS['Hydration_Level'])
    df['Oil_enc']         = df['Oil_Level'].map(LABEL_MAPS['Oil_Level'])
    df['Sensitivity_enc'] = df['Sensitivity'].map(LABEL_MAPS['Sensitivity'])

    le = LabelEncoder()
    df['Skin_Type_enc'] = le.fit_transform(df['Skin_Type'])

    X = df[FEATURE_COLS].values
    y = df['Skin_Type_enc'].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Predict skin type
    clf = KNeighborsClassifier(n_neighbors=11)
    clf.fit(X_s, y)
    x_enc = encode_input(input_data)
    x_s   = scaler.transform([x_enc])
    pred  = clf.predict(x_s)[0]
    proba = clf.predict_proba(x_s)[0]
    predicted_skin_type = le.inverse_transform([pred])[0]

    proba_dict = {cls: round(float(p)*100, 1) for cls, p in zip(le.classes_, proba)}

    # Find nearest neighbours
    nn = NearestNeighbors(n_neighbors=n+1, metric='euclidean')
    nn.fit(X_s)
    distances, indices = nn.kneighbors(x_s)
    distances = distances[0][1:]  # exclude self
    indices   = indices[0][1:]

    # Similarity as percentage (1 - normalized_distance)
    max_dist = distances.max() if distances.max() > 0 else 1
    similarities = [round((1 - d/max_dist)*100, 1) for d in distances]

    similar_profiles = []
    for idx, sim in zip(indices, similarities):
        row = df.iloc[idx]
        similar_profiles.append({
            'rank': len(similar_profiles)+1,
            'Age': int(row['Age']),
            'Gender': row['Gender'],
            'Hydration_Level': row['Hydration_Level'],
            'Oil_Level': row['Oil_Level'],
            'Sensitivity': row['Sensitivity'],
            'Humidity': round(float(row['Humidity']), 1),
            'Temperature': round(float(row['Temperature']), 1),
            'Skin_Type': row['Skin_Type'],
            'skin_score': round(float(row['skin_score']), 1),
            'skin_age': round(float(row['skin_age']), 1),
            'similarity': sim,
        })

    advice = SKINCARE_ADVICE.get(predicted_skin_type, SKINCARE_ADVICE['Normal'])

    return {
        'predicted_skin_type': predicted_skin_type,
        'probabilities': proba_dict,
        'similar_profiles': similar_profiles,
        'skincare_advice': advice,
    }
