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
