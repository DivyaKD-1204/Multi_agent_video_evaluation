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
