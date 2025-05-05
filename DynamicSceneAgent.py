class DynamicSceneAgent:
    def __init__(self):
        message_bus.subscribe(self, ["evaluate_dynamic"])
    
    def evaluate(self, video_path: str):
        try:
            cap = cv2.VideoCapture(video_path)
            scene_changes = 0
            prev_frame = None
            total_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is not None:
                    hist1 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([gray], [0], None, [256], [0, 256])
                    score = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
                    if score > SCENE_CHANGE_THRESHOLD:
                        scene_changes += 1
                
                prev_frame = gray
                total_frames += 1
            
            cap.release()
            
            change_ratio = float(scene_changes / total_frames) if total_frames > 0 else 0.0
            message_bus.results["dynamic"] = {
                "change_ratio": change_ratio,
                "assessment": "Smooth" if change_ratio < 0.1 else 
                              "Moderate changes" if change_ratio < 0.3 else 
                              "Frequent changes"
            }
            message_bus.publish("report_ready", {})
            
        except Exception as e:
            logger.error(f"Dynamic scene evaluation failed: {str(e)}")
            message_bus.results["dynamic"] = {"error": str(e)}
    
    def process_commands(self):
        for cmd in message_bus.commands:
            if cmd["type"] == "evaluate_dynamic":
                self.evaluate(cmd["video_path"])
