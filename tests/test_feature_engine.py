from features.feature_engine import FeatureEngine

engine = FeatureEngine()


deck = [
    26000000,  # Knight
    26000015,  # Fireball
    26000021,  # Hog Rider
    26000032,  # Musketeer
    26000014,  # Mini P.E.K.K.A.
    26000010,  # Skeletons
    27000003,  # Cannon
    28000011,  # The Log
]

features = engine.extract_features(deck)

for key, value in features.items():
    print(f"{key:25}: {value}")