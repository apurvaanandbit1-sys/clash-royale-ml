import json
import yaml
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Clash Royale Intrinsic Matchup Predictor API",
    description="Production API for Clash Royale deck matchup prediction and card embedding analysis.",
    version="1.0.0"
)

# Global runtime state
onnx_session: Optional[ort.InferenceSession] = None
card_to_idx: Dict[str, int] = {}
idx_to_card: Dict[int, str] = {}
meta_deck_cache: Dict[str, float] = {}
embeddings_2d: Dict[str, List[float]] = {}
num_cards: int = 0

class PredictRequest(BaseModel):
    p1_deck: List[int] = Field(..., description="List of 8 card IDs for Player A")
    p2_deck: List[int] = Field(..., description="List of 8 card IDs for Player B")
    trophy_diff: float = Field(0.0, description="Trophy difference (Player A - Player B). Defaults to 0.0 for intrinsic matchup.")

class PredictResponse(BaseModel):
    win_probability: float
    categorical_band: str
    cached: bool
    model_version: str = "sprint13_onnx_v1.0"

class SimilarityRequest(BaseModel):
    deck: List[int] = Field(..., description="List of 8 card IDs")
    top_k: int = Field(5, description="Number of nearest decks to retrieve")

@app.on_event("startup")
def startup_event():
    global onnx_session, card_to_idx, idx_to_card, num_cards, embeddings_2d, meta_deck_cache
    
    # 1. Load card library
    card_lib_path = PROJECT_ROOT / "data" / "card_library.json"
    if card_lib_path.exists():
        with open(card_lib_path, "r") as f:
            cards_lib = json.load(f)
        sorted_ids = sorted(list(cards_lib.keys()))
        card_to_idx = {str(cid): idx for idx, cid in enumerate(sorted_ids)}
        idx_to_card = {idx: str(cid) for idx, cid in enumerate(sorted_ids)}
        num_cards = len(card_to_idx)
    else:
        num_cards = 122
        card_to_idx = {str(i): i for i in range(122)}
        idx_to_card = {i: str(i) for i in range(122)}
        
    # 2. Load ONNX model
    onnx_path = PROJECT_ROOT / "models" / "onnx" / "matchup_model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model missing at {onnx_path}. Run scripts/export_onnx.py first.")
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # 3. Load card embeddings & precompute 2D projection (tied to sprint13_matchup_model.pt)
    import torch
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint13_matchup_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt)
        if "encoder.embeddings.weight" in state_dict:
            emb_weights = state_dict["encoder.embeddings.weight"].numpy()
            
            # t-SNE 2D projection
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, num_cards-1))
            coords_2d = tsne.fit_transform(emb_weights)
            for i in range(len(coords_2d)):
                cid = idx_to_card.get(i, str(i))
                embeddings_2d[cid] = [float(coords_2d[i, 0]), float(coords_2d[i, 1])]

    # 4. Precompute top meta decks cache
    meta_deck_cache = {}

def get_categorical_band(prob: float) -> str:
    if prob >= 0.58:
        return "Strongly Favored"
    elif prob >= 0.52:
        return "Slightly Favored"
    elif prob >= 0.48:
        return "Even Matchup"
    elif prob >= 0.42:
        return "Slightly Unfavored"
    else:
        return "Strongly Unfavored"

def deck_key(p1_deck: List[int], p2_deck: List[int]) -> str:
    s1 = "-".join(sorted([str(c) for c in p1_deck]))
    s2 = "-".join(sorted([str(c) for c in p2_deck]))
    return f"{s1}_{s2}"

@app.post("/predict", response_model=PredictResponse)
def predict_matchup(req: PredictRequest):
    if len(req.p1_deck) != 8 or len(req.p2_deck) != 8:
        raise HTTPException(status_code=400, detail="Decks must contain exactly 8 card IDs.")
        
    key = deck_key(req.p1_deck, req.p2_deck)
    if key in meta_deck_cache and req.trophy_diff == 0.0:
        prob = meta_deck_cache[key]
        return PredictResponse(
            win_probability=prob,
            categorical_band=get_categorical_band(prob),
            cached=True
        )
        
    p1_indices = [card_to_idx.get(str(c), 0) for c in req.p1_deck]
    p2_indices = [card_to_idx.get(str(c), 0) for c in req.p2_deck]
    
    p1_arr = np.array([p1_indices], dtype=np.int64)
    p2_arr = np.array([p2_indices], dtype=np.int64)
    td_arr = np.array([[req.trophy_diff]], dtype=np.float32)
    
    outputs = onnx_session.run(None, {
        "p1_deck": p1_arr,
        "p2_deck": p2_arr,
        "trophy_diff": td_arr
    })
    
    prob = float(outputs[0][0][0])
    
    # Cache intrinsic predictions
    if req.trophy_diff == 0.0:
        meta_deck_cache[key] = prob
        
    return PredictResponse(
        win_probability=prob,
        categorical_band=get_categorical_band(prob),
        cached=False
    )

@app.get("/embeddings")
def get_card_embeddings():
    """
    Returns 2D t-SNE coordinate projections of all 122 card embeddings.
    NOTE: Embeddings are explicitly tied to frozen checkpoint 'sprint13_matchup_model.pt'
    due to rotational invariance of average pooled Deep Sets space across retraining runs.
    """
    return {
        "checkpoint": "sprint13_matchup_model.pt",
        "projection_method": "t-SNE",
        "num_cards": len(embeddings_2d),
        "coordinates": embeddings_2d
    }

# Call startup_event directly at module load
startup_event()

@app.get("/health")
def health_check():
    return {"status": "ok", "onnx_model_loaded": onnx_session is not None}
