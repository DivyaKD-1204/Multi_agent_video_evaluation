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
