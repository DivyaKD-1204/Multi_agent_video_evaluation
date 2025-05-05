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
