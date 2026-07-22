import requests
from bs4 import BeautifulSoup
import json
import time

# Comprehensive repository of 22 handwritten meta-game archetypes
EXISTING_ARCHETYPES = {
    # === HOG RIDER & CYCLES ===
    "hog_2_6_classic": ["hog-rider", "musketeer", "ice-golem", "skeletons", "ice-spirit", "cannon", "the-log", "fireball"],
    "hog_quake_firecracker": ["hog-rider", "firecracker", "knight", "skeletons", "ice-spirit", "tesla", "the-log", "earthquake"],
    "hog_exegaze_heavy": ["hog-rider", "executioner", "valkyrie", "goblins", "ice-spirit", "tornado", "the-log", "rocket"],
    "miner_poison_control": ["miner", "poison", "knight", "goblins", "ice-spirit", "tesla", "the-log", "delivery"],
    "goblin_drill_cycle": ["goblin-drill", "bomber", "firecracker", "wall-breakers", "knight", "skeletons", "tesla", "the-log"],

    # === BEATDOWN & HEAVY ARCHETYPES ===
    "golem_night_witch_classic": ["golem", "night-witch", "baby-dragon", "mega-minion", "lumberjack", "tornado", "lightning", "barbarian-barrel"],
    "golem_healer_elixir_pump": ["golem", "elixir-collector", "battle-healer", "electro-dragon", "night-witch", "tornado", "barbarian-barrel", "rage"],
    "lava_loon_classic": ["lava-hound", "balloon", "mega-minion", "minions", "tombstone", "barbarians", "fireball", "zap"],
    "lava_miner_clone": ["lava-hound", "miner", "skeleton-barrel", "flying-machine", "clone", "bats", "barbarians", "zap"],
    "egiant_bowler_tornado": ["electro-giant", "bowler", "baby-dragon", "mega-minion", "golden-knight", "tornado", "lightning", "barbarian-barrel"],
    "giant_graveyard_beatdown": ["giant", "graveyard", "bowler", "night-witch", "bats", "arrows", "giant-snowball", "musketeer"],
    "electro_giant_lightning_cycle": ["electro-giant", "lightning", "bomber", "cannon-cart", "golden-knight", "baby-dragon", "barbarian-barrel", "tornado"],

    # === BRIDGE SPAM ARCHETYPES ===
    "pekka_bridge_spam_classic": ["pekka", "battle-ram", "bandit", "royal-ghost", "electro-wizard", "magic-archer", "zap", "poison"],
    "pekka_ram_rider_bolted": ["pekka", "ram-rider", "bandit", "electro-wizard", "royal-ghost", "barbarian-barrel", "lightning", "giant-snowball"],
    "mega_knight_spam_miners": ["mega-knight", "bandit", "royal-ghost", "electro-wizard", "miner", "wall-breakers", "bats", "zap"],
    "royal_recruits_hogs_split": ["royal-recruits", "royal-hogs", "flying-machine", "zappies", "goblin-cage", "barbarian-barrel", "arrows", "electro-spirit"],

    # === SPELL BAIT ARCHETYPES ===
    "log_bait_classic_2_9": ["princess", "goblin-barrel", "goblin-gang", "knight", "skeletons", "inferno-tower", "the-log", "rocket"],
    "log_bait_valk_tesla": ["princess", "goblin-barrel", "goblin-gang", "valkyrie", "tesla", "the-log", "rocket", "ice-spirit"],
    "mega_knight_zap_bait": ["mega-knight", "goblin-barrel", "skeleton-barrel", "miner", "goblin-gang", "bats", "spear-goblins", "zap"],
    "mortar_bait_miner": ["mortar", "miner", "goblin-gang", "spear-goblins", "minion-horde", "minions", "the-log", "fireball"],

    # === SIEGE DECK ARCHETYPES ===
    "xbow_2_9_classic": ["x-bow", "tesla", "knight", "archers", "skeletons", "ice-spirit", "the-log", "fireball"],
    "ice_bow_rocket_turtle": ["x-bow", "tesla", "knight", "ice-wizard", "tornado", "the-log", "rocket", "skeletons"]
}

def scrape_and_merge_meta_decks(limit=60):
    print("=========================================")
    print("     CONNECTING TO ROYALEAPI ENGINE      ")
    print("=========================================\n")
    
    url = "https://royaleapi.com/decks/popular?time=7d"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Connection rejected by server. Status code: {response.status_code}")
            print("Defaulting entirely to your 22 hand-crafted archetypes library.")
            save_library(EXISTING_ARCHETYPES)
            return
    except Exception as e:
        print(f"Network error occurred: {e}")
        print("Defaulting entirely to your 22 hand-crafted archetypes library.")
        save_library(EXISTING_ARCHETYPES)
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    deck_elements = soup.find_all('div', class_='deck_item')
    print(f"Found {len(deck_elements)} potential deck containers on the page layout.")
    
    merged_archetypes = EXISTING_ARCHETYPES.copy()
    scraped_count = 0
    
    for idx, deck_element in enumerate(deck_elements):
        if scraped_count >= limit:
            break
            
        card_tags = deck_element.find_all('img', class_='deck_card')
        card_list = []
        
        for card in card_tags:
            card_name = card.get('alt', '').lower().strip().replace(" ", "-")
            if card_name:
                card_list.append(card_name)
                
        card_list = sorted(list(set([c.replace("/", "-") for c in card_list])))
        
        if len(card_list) == 8:
            scraped_count += 1
            deck_key = f"royaleapi_meta_{scraped_count}"
            merged_archetypes[deck_key] = card_list

    print(f"\nExtraction complete: Scraped {scraped_count} live trending decks.")
    print(f"Total archetypes currently locked into your matrix index: {len(merged_archetypes)}")
    
    save_library(merged_archetypes)

def save_library(library_data):
    with open("meta_archetypes_library.json", "w") as f:
        json.dump(library_data, f, indent=4)
    print("\nMaster map successfully compiled and stored inside 'meta_archetypes_library.json'!")

if __name__ == "__main__":
    scrape_and_merge_meta_decks()