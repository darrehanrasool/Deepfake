"""Standalone forensic verification inference for arbitrary media files."""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import torch
import torch.nn.functional as F

from university.model import ST2SyncNet
from university.dataset import MultimodalDeepfakeDataset


def verify_video_authenticity(
    video_path: str,
    model_weights_path: Path,
    device: torch.device,
    num_frames: int = 8
) -> Dict[str, Any]:
    """Evaluates an arbitrary video file using trained ST2-SyncNet weights."""
    print(f"\n[*] Evaluating media authenticity for: {video_path}")
    model = ST2SyncNet().to(device)

    if model_weights_path.exists():
        weights = torch.load(model_weights_path, map_location=device)
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            model.load_state_dict(weights['model_state_dict'])
        else:
            model.load_state_dict(weights)
        print("  [✓] Loaded trained model weights cleanly.")
    else:
        print(f"  [-] Warning: Model weights not found at {model_weights_path}. Using initial weights.")

    model.eval()

    temp_df = pd.DataFrame([{'video_path': str(video_path), 'label': 0, 'dataset': 'Test', 'actor_group': 'test'}])
    temp_ds = MultimodalDeepfakeDataset(temp_df, num_frames=num_frames, is_train=False)
    sample = temp_ds[0]

    spatial = sample['spatial'].unsqueeze(0).to(device)
    temporal = sample['temporal'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(spatial, temporal)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    prob_real = float(probs[0] * 100)
    prob_fake = float(probs[1] * 100)
    verdict = 'REAL (Authentic Human Capture)' if prob_real >= 50.0 else 'FAKE (Deepfake / Synthetic Manipulation)'
    confidence = max(prob_real, prob_fake)

    print('=' * 75)
    print('          DEEPFAKE-AI MULTIMODAL FORENSIC VERDICT')
    print('=' * 75)
    print(f'  • Final Verdict     : {verdict}')
    print(f'  • Confidence        : {confidence:.2f}%')
    print(f'  • Probability REAL  : {prob_real:.2f}%')
    print(f'  • Probability FAKE  : {prob_fake:.2f}%')
    print('=' * 75)

    return {
        'verdict': verdict,
        'confidence': confidence,
        'prob_real': prob_real,
        'prob_fake': prob_fake
    }
