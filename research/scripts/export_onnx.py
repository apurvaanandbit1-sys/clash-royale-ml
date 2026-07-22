import json
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel

class ONNXMatchupModelWrapper(nn.Module):
    """
    Wrapper around MatchupModel that computes sigmoid win probabilities
    and supports fixed trophy difference input for intrinsic matchup evaluation.
    """
    def __init__(self, model: MatchupModel):
        super().__init__()
        self.model = model

    def forward(self, p1_deck: torch.Tensor, p2_deck: torch.Tensor, trophy_diff: torch.Tensor) -> torch.Tensor:
        logits = self.model(p1_deck, p2_deck, trophy_diff)
        return torch.sigmoid(logits)

def export_onnx():
    print("=== Exporting MatchupModel to ONNX Format ===")
    
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
        
    card_lib_path = PROJECT_ROOT / ds_config["data"]["card_library_path"]
    with open(card_lib_path, "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint13_matchup_model.pt"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} missing.")
        sys.exit(1)
        
    encoder = DeckEncoder(num_cards=num_cards, embedding_dim=16)
    head = BradleyTerryHead(projection_dim=16)
    skill = SkillDifference(hidden_dim=8)
    model = MatchupModel(encoder, head, skill)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device("cpu")))
    model.eval()
    
    onnx_wrapper = ONNXMatchupModelWrapper(model)
    onnx_wrapper.eval()
    
    onnx_dir = PROJECT_ROOT / "models" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "matchup_model.onnx"
    
    # Dummy inputs
    dummy_p1 = torch.zeros((1, 8), dtype=torch.int64)
    dummy_p2 = torch.ones((1, 8), dtype=torch.int64)
    dummy_td = torch.zeros((1, 1), dtype=torch.float32)
    
    torch.onnx.export(
        onnx_wrapper,
        (dummy_p1, dummy_p2, dummy_td),
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["p1_deck", "p2_deck", "trophy_diff"],
        output_names=["win_probability"],
        dynamic_axes={
            "p1_deck": {0: "batch_size"},
            "p2_deck": {0: "batch_size"},
            "trophy_diff": {0: "batch_size"},
            "win_probability": {0: "batch_size"}
        }
    )
    
    print(f"[+] ONNX model exported successfully to: {onnx_path}")
    
    # Verify ONNX Runtime Execution
    print("\n--- Verifying ONNX Runtime Inference ---")
    session = ort.InferenceSession(str(onnx_path))
    
    input_p1 = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    input_p2 = np.array([[8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int64)
    input_td = np.array([[0.0]], dtype=np.float32)
    
    ort_inputs = {
        "p1_deck": input_p1,
        "p2_deck": input_p2,
        "trophy_diff": input_td
    }
    
    ort_outputs = session.run(None, ort_inputs)
    prob_onnx = float(ort_outputs[0][0][0])
    
    # PyTorch verification
    with torch.no_grad():
        prob_pt = float(onnx_wrapper(torch.tensor(input_p1), torch.tensor(input_p2), torch.tensor(input_td)).numpy()[0][0])
        
    print(f"  • ONNX Runtime Predicted Probability: {prob_onnx:.6f}")
    print(f"  • PyTorch Predicted Probability:      {prob_pt:.6f}")
    print(f"  • Absolute Difference:               {abs(prob_onnx - prob_pt):.8f}")
    
    assert abs(prob_onnx - prob_pt) < 1e-5, "ONNX and PyTorch outputs mismatch!"
    print("  • ONNX Verification Status:          PASSED")

if __name__ == "__main__":
    export_onnx()
