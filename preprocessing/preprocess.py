import sqlite3
import pandas as pd
# import os
from pathlib import Path
from .deck_features import add_deck_features

def generate_augmented_dataset():
    print("=========================================")
    print("   RUNNING ADVANCED DATA PREPROCESSOR    ")
    print("=========================================\n")
    
    # 1. Connect to the exact SQLite database file
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DB_PATH = PROJECT_ROOT / "data" / "clashroyale.db"

    print(f"Connecting to '{DB_PATH}'...")

    if not DB_PATH.exists():
        print(f"CRITICAL ERROR: '{DB_PATH}' does not exist!")
        return None

    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Match your exact table 'battles' and column schemas
        df = pd.read_sql_query("SELECT player_deck, opponent_deck, win FROM battles", conn)
    except Exception as e:
        print(f"Database Query Error: {e}")
        return None
    finally:
        conn.close()
        
    if df.empty:
        print("Error: The database warehouse contains 0 records.")
        return None
        
    print(f"Successfully loaded {len(df)} match rows from warehouse database.")

    print("Generating engineered deck features...")
    df = add_deck_features(df)

    #=================================================================================
    # # 3. Handle teacher's expanded JSON nesting dictionary gracefully
    # print("Loading 'meta_archetypes_library_expanded.json' mapping registry...")
    # try:
    #     with open("meta_archetypes_library_expanded.json", "r") as f:
    #         raw_json_data = json.load(f)
            
    #     if "archetypes" in raw_json_data:
    #         archetype_lib = {name: frozenset(data["cards"]) for name, data in raw_json_data["archetypes"].items()}
    #     else:
    #         archetype_lib = load_archetypes("meta_archetypes_library_expanded.json")
    # except FileNotFoundError:
    #     print("Error: 'meta_archetypes_library_expanded.json' is missing from this directory.")
    #     return None

    # # 4. Use your teacher's built-in archetype encoder
    # print("Classifying matches and building macro-archetype structures...")
    # df = add_archetype_columns(
    #     df=df, 
    #     archetypes=archetype_lib, 
    #     p1_cards_col='p1_cards_list', 
    #     p2_cards_col='p2_cards_list', 
    #     threshold=7
    # )
    
    # # 5. Apply target encoding with Laplacian smoothing
    # np.random.seed(42)
    # train_mask = np.random.rand(len(df)) < 0.8
    
    # print("Computing Bayesian-smoothed matchup win-rates across pairs...")
    # df = matchup_win_rate_features(df, label_col='win', train_mask=train_mask, smoothing=5)
    #=============================================================================================

    # 6. Housekeeping
    # df = df.drop(columns=['p1_cards_list', 'p2_cards_list'])
    
    print(f"\nMatrix transformation complete! Total data dimensions: {df.shape}")
    return df


# --- FORCE EXECUTION OUT IN THE OPEN ---
augmented_df = generate_augmented_dataset()

if augmented_df is not None:
    output_path = "matches_with_features.parquet"   
    augmented_df.to_parquet(output_path, index=False)
    print(f"SUCCESS: Unified dataset file safely locked into '{output_path}'!\n")
else:
    print("Pipeline failed to generate the augmented dataset.")