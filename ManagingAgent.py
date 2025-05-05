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
