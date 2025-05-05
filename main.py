import cv2
import numpy as np
import torch
import json
from typing import Dict, List, Optional, Union
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from datetime import datetime
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ---------------------- SYSTEM CONSTANTS ---------------------- #
CHUNK_SIZE = 1000  # frames to process at once
SCENE_CHANGE_THRESHOLD = 0.3
TEMPORAL_METRICS = {
    "optical_flow": "Optical flow magnitude between frames",
    "frame_difference": "Pixel-wise absolute difference",
    "ssim": "Structural Similarity Index",
    "edge_consistency": "Edge overlap ratio"
}

# ---------------------- IMPROVED MESSAGE BUS ---------------------- #
class MessageBus:
    def __init__(self):
        self.commands = []
        self.results = {}
        self.subscribers = {}
    
    def publish(self, command_type: str, data: Dict):
        self.commands.append({"type": command_type, **data})
        self.notify_subscribers(command_type)
    
    def subscribe(self, agent, command_types: List[str]):
        for cmd_type in command_types:
            if cmd_type not in self.subscribers:
                self.subscribers[cmd_type] = []
            self.subscribers[cmd_type].append(agent)
    
    def notify_subscribers(self, command_type: str):
        for agent in self.subscribers.get(command_type, []):
            agent.process_commands()

message_bus = MessageBus()

# ---------------------- SERIALIZATION UTILS ---------------------- #
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)

# ---------------------- MANAGING AGENT ---------------------- #
class ManagingAgent:
    def __init__(self):
        message_bus.subscribe(self, ["report_ready"])
    
    def interact(self):
        print("\nVideo Evaluation System")
        print("="*40)
        
        # Metric selection
        print("\nAvailable Temporal Metrics:")
        for i, (k, v) in enumerate(TEMPORAL_METRICS.items(), 1):
            print(f"{i}. {k}: {v}")
        
        selected = input("\nSelect metrics (comma-separated numbers): ").strip().split(',')
        metrics = [list(TEMPORAL_METRICS.keys())[int(i)-1] for i in selected if i.isdigit()]
        
        # Reference input
        reference_text = input("\nEnter reference text description: ").strip()
        
        # Video input
        video_path = input("\nEnter video file path: ").strip()
        
        # Dispatch tasks
        message_bus.publish("evaluate_temporal", {
            "video_path": video_path,
            "metrics": metrics
        })
        
        message_bus.publish("evaluate_semantic", {
            "video_path": video_path,
            "reference_text": reference_text
        })
        
        message_bus.publish("evaluate_dynamic", {
            "video_path": video_path
        })
        
        message_bus.publish("evaluate_generalization", {
            "video_path": video_path
        })
    
    def process_commands(self):
        if len(message_bus.results) == 4:  # All results received
            ReportingAgent().generate_report()

# ---------------------- TEMPORAL COHERENCE AGENT ---------------------- #
class TemporalCoherenceAgent:
    def __init__(self):
        message_bus.subscribe(self, ["evaluate_temporal"])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def calculate_optical_flow(self, prev: np.ndarray, curr: np.ndarray) -> float:
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))
    
    def calculate_frame_difference(self, prev: np.ndarray, curr: np.ndarray) -> float:
        diff = cv2.absdiff(prev, curr)
        return float(np.mean(diff))
    
    def calculate_ssim(self, prev: np.ndarray, curr: np.ndarray) -> float:
        return float(ssim(prev, curr, data_range=curr.max() - curr.min()))
    
    def calculate_edge_consistency(self, prev: np.ndarray, curr: np.ndarray) -> float:
        e1 = cv2.Canny(prev, 100, 200)
        e2 = cv2.Canny(curr, 100, 200)
        inter = np.logical_and(e1, e2)
        union = np.logical_or(e1, e2)
        return float(np.sum(inter) / np.sum(union)) if np.sum(union) > 0 else 1.0
    
    def process_video_chunk(self, cap, metrics: List[str]) -> Dict[str, float]:
        results = {m: [] for m in metrics}
        prev_frame = None
        
        for _ in range(CHUNK_SIZE):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                if "optical_flow" in metrics:
                    results["optical_flow"].append(self.calculate_optical_flow(prev_frame, gray))
                if "frame_difference" in metrics:
                    results["frame_difference"].append(self.calculate_frame_difference(prev_frame, gray))
                if "ssim" in metrics:
                    results["ssim"].append(self.calculate_ssim(prev_frame, gray))
                if "edge_consistency" in metrics:
                    results["edge_consistency"].append(self.calculate_edge_consistency(prev_frame, gray))
            
            prev_frame = gray
        
        return {k: float(np.mean(v)) if v else 0.0 for k, v in results.items()}
    
    def evaluate(self, video_path: str, metrics: List[str]):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            total_results = {m: [] for m in metrics}
            while True:
                chunk_results = self.process_video_chunk(cap, metrics)
                if not any(chunk_results.values()):
                    break
                for m in metrics:
                    total_results[m].append(chunk_results[m])
            
            cap.release()
            final_results = {m: float(np.mean(v)) for m, v in total_results.items() if v}
            message_bus.results["temporal"] = final_results
            message_bus.publish("report_ready", {})
            
        except Exception as e:
            logger.error(f"Temporal evaluation failed: {str(e)}")
            message_bus.results["temporal"] = {"error": str(e)}
    
    def process_commands(self):
        for cmd in message_bus.commands:
            if cmd["type"] == "evaluate_temporal":
                self.evaluate(cmd["video_path"], cmd["metrics"])

# ---------------------- SEMANTIC CONSISTENCY AGENT ---------------------- #
class SemanticConsistencyAgent:
    def __init__(self):
        message_bus.subscribe(self, ["evaluate_semantic"])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            use_fast=True  # Fixes the slow processor warning
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.sbert = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device
        )
    
    def process_frame(self, frame: np.ndarray) -> str:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def evaluate(self, video_path: str, reference_text: str):
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_rate = max(1, total_frames // 8)  # Sample 8 frames
            
            captions = []
            for i in range(0, total_frames, sample_rate):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    captions.append(self.process_frame(frame))
            
            cap.release()
            
            summary = " ".join(captions)
            ref_emb = self.sbert.encode(reference_text, convert_to_tensor=True)
            sum_emb = self.sbert.encode(summary, convert_to_tensor=True)
            score = float(util.cos_sim(ref_emb, sum_emb).item())
            
            message_bus.results["semantic"] = {
                "score": score,
                "summary": summary,
                "reference": reference_text
            }
            message_bus.publish("report_ready", {})
            
        except Exception as e:
            logger.error(f"Semantic evaluation failed: {str(e)}")
            message_bus.results["semantic"] = {"error": str(e)}
    
    def process_commands(self):
        for cmd in message_bus.commands:
            if cmd["type"] == "evaluate_semantic":
                self.evaluate(cmd["video_path"], cmd["reference_text"])

# ---------------------- DYNAMIC SCENE AGENT ---------------------- #
class DynamicSceneAgent:
    def __init__(self):
        message_bus.subscribe(self, ["evaluate_dynamic"])
        # Parameters for feature detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        # Parameters for optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def evaluate(self, video_path: str):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Initialize metrics
            metrics = {
                'scene_changes': 0,
                'total_frames': 0,
                'flow_magnitudes': [],
                'brightness_changes': [],
                'object_movements': [],
                'prev_frame': None,
                'prev_gray': None,
                'prev_brightness': None,
                'prev_features': None
            }
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and HSV
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                current_brightness = np.mean(hsv[:,:,2])  # Value channel
                
                if metrics['prev_frame'] is not None:
                    # 1. Scene Change Detection
                    hist1 = cv2.calcHist([metrics['prev_gray']], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                    if hist_score > SCENE_CHANGE_THRESHOLD:
                        metrics['scene_changes'] += 1
                    
                    # 2. Optical Flow Analysis (Camera Movement)
                    flow = cv2.calcOpticalFlowFarneback(
                        metrics['prev_gray'], gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                    metrics['flow_magnitudes'].append(np.mean(mag))
                    
                    # 3. Lighting Change Detection
                    brightness_change = abs(current_brightness - metrics['prev_brightness'])/metrics['prev_brightness']
                    metrics['brightness_changes'].append(brightness_change)
                    
                    # 4. Object Movement Analysis
                    if metrics['prev_features'] is not None:
                        features, status, _ = cv2.calcOpticalFlowPyrLK(
                            metrics['prev_gray'], gray, 
                            metrics['prev_features'], None, 
                            **self.lk_params
                        )
                        if features is not None:
                            good_new = features[status==1]
                            good_old = metrics['prev_features'][status==1]
                            distances = np.sqrt(np.sum((good_new - good_old)**2, axis=1))
                            metrics['object_movements'].extend(distances)
                
                # Update tracking features every 5 frames
                if metrics['total_frames'] % 5 == 0:
                    metrics['prev_features'] = cv2.goodFeaturesToTrack(
                        gray, mask=None, **self.feature_params
                    )
                
                # Update previous frame data
                metrics['prev_frame'] = frame
                metrics['prev_gray'] = gray
                metrics['prev_brightness'] = current_brightness
                metrics['total_frames'] += 1
            
            cap.release()
            
            # Calculate final metrics
            change_ratio = metrics['scene_changes'] / metrics['total_frames'] if metrics['total_frames'] > 0 else 0
            avg_flow = np.mean(metrics['flow_magnitudes']) if metrics['flow_magnitudes'] else 0
            flow_variance = np.var(metrics['flow_magnitudes']) if metrics['flow_magnitudes'] else 0
            brightness_change_freq = np.mean([1 if x > 0.2 else 0 for x in metrics['brightness_changes']]) if metrics['brightness_changes'] else 0
            avg_object_movement = np.mean(metrics['object_movements']) if metrics['object_movements'] else 0
            object_movement_var = np.var(metrics['object_movements']) if metrics['object_movements'] else 0
            
            # Generate assessment
            assessment = self.generate_assessment(
                change_ratio,
                flow_variance,
                brightness_change_freq,
                object_movement_var
            )
            
            message_bus.results["dynamic"] = {
                "scene_change_ratio": float(change_ratio),
                
                "flow_variance": float(flow_variance),
                "brightness_change_frequency": float(brightness_change_freq),
                "avg_object_movement": float(avg_object_movement),
                "object_movement_variance": float(object_movement_var),
                "assessment": assessment
            }
            message_bus.publish("report_ready", {})
            
        except Exception as e:
            logger.error(f"Dynamic scene evaluation failed: {str(e)}")
            message_bus.results["dynamic"] = {"error": str(e)}
    
    def generate_assessment(self, change_ratio, flow_var, brightness_freq, obj_movement_var):
        score = 0
        weights = {
            'scene_change': 0.3,
            'camera_movement': 0.25,
            'lighting': 0.2,
            'object_movement': 0.25
        }
        
        # Score components
        if change_ratio < 0.1:
            score += weights['scene_change'] * 1.0
        elif change_ratio < 0.3:
            score += weights['scene_change'] * 0.6
        
        if flow_var < 0.05:
            score += weights['camera_movement'] * 1.0
        elif flow_var < 0.1:
            score += weights['camera_movement'] * 0.7
        
        if brightness_freq < 0.1:
            score += weights['lighting'] * 1.0
        elif brightness_freq < 0.3:
            score += weights['lighting'] * 0.5
        
        if obj_movement_var < 0.01:
            score += weights['object_movement'] * 1.0
        elif obj_movement_var < 0.05:
            score += weights['object_movement'] * 0.5
        
        # Final assessment
        if score > 0.8:
            return "Excellent dynamic scene handling"
        elif score > 0.6:
            return "Good handling with minor artifacts"
        elif score > 0.4:
            return "Moderate handling - noticeable artifacts"
        else:
            return "Poor handling - frequent artifacts during changes"
    
    def process_commands(self):
        for cmd in message_bus.commands:
            if cmd["type"] == "evaluate_dynamic":
                self.evaluate(cmd["video_path"])
# ---------------------- GENERALIZATION AGENT ---------------------- #
class GeneralizationAgent:
    def __init__(self):
        message_bus.subscribe(self, ["evaluate_generalization"])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def evaluate(self, video_path: str):
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Sample frames evenly
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = np.linspace(0, total_frames-1, 20, dtype=int)
            
            for i in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(cv2.resize(frame, (224, 224))))
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames processed")
            
            inputs = self.clip_processor(images=frames, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            avg_feature = features.mean(dim=0)
            baseline = torch.zeros_like(avg_feature)
            score = float(1 - torch.nn.functional.cosine_similarity(avg_feature, baseline, dim=0).item())
            
            message_bus.results["generalization"] = {
                "novelty_score": score,
                "assessment": "Typical content" if score < 0.3 else 
                            "Some novel features" if score < 0.6 else 
                            "Highly novel content"
            }
            message_bus.publish("report_ready", {})
            
        except Exception as e:
            logger.error(f"Generalization evaluation failed: {str(e)}")
            message_bus.results["generalization"] = {"error": str(e)}
    
    def process_commands(self):
        for cmd in message_bus.commands:
            if cmd["type"] == "evaluate_generalization":
                self.evaluate(cmd["video_path"])

# ---------------------- REPORTING AGENT ---------------------- #
class ReportingAgent:
    def generate_report(self):
        if not all(k in message_bus.results for k in ["temporal", "semantic", "dynamic", "generalization"]):
            return
        
        report = {
            "metadata": {
                "system_version": "1.0",
                "evaluation_date": datetime.now().isoformat()
            },
            "quantitative_metrics": message_bus.results,
            "qualitative_assessment": {
                "temporal_quality": self.assess_temporal(message_bus.results["temporal"]),
                "semantic_alignment": self.assess_semantic(message_bus.results["semantic"]),
                "scene_continuity": message_bus.results["dynamic"].get("assessment", "N/A"),
                "content_familiarity": message_bus.results["generalization"].get("assessment", "N/A")
            }
        }
        
        print("\n=== VIDEO EVALUATION REPORT ===")
        print(json.dumps(report, indent=2, cls=NumpyEncoder))
        
        # Save to file
        with open("video_evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        print("\nReport saved to 'video_evaluation_report.json'")
    
    def assess_temporal(self, metrics: Dict) -> str:
        if "error" in metrics:
            return "Evaluation failed"
        
        avg_coherence = np.mean([v for k, v in metrics.items() if k != "error"])
        return "Smooth" if avg_coherence > 0.7 else "Moderate" if avg_coherence > 0.4 else "Choppy"
    
    def assess_semantic(self, metrics: Dict) -> str:
        if "error" in metrics:
            return "Evaluation failed"
        return "Strong alignment" if metrics["score"] > 0.7 else "Moderate alignment" if metrics["score"] > 0.4 else "Weak alignment"

# ---------------------- SYSTEM INITIALIZATION ---------------------- #
def initialize_system():
    agents = [
        ManagingAgent(),
        TemporalCoherenceAgent(),
        SemanticConsistencyAgent(),
        DynamicSceneAgent(),
        GeneralizationAgent()
    ]
    return agents

# ---------------------- MAIN EXECUTION ---------------------- #
if __name__ == "__main__":
    print("Initializing Video Evaluation System...")
    agents = initialize_system()
    
    # Start interaction
    for agent in agents:
        if isinstance(agent, ManagingAgent):
            agent.interact()
            break
