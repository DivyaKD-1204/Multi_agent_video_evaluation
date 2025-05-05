# Multi_agent_video_evaluation

##  Installation

I have already shared the setup_guide and specified the important installations to be done. You just need to install those packages and run the code.


# Multi-Agent Video Evaluation System
A modular, scalable multi-agent system designed to evaluate the quality of both real-life and AI-generated videos based on:

- **Temporal Coherence**
- **Semantic Consistency**
- **Dynamic Scene Handling**
- **Generalization across Contexts**

##  Agents Overview
| Agent Name         | Responsibility                                               |
|--------------------|--------------------------------------------------------------|
| ManagingAgent      | Manages all the agents and communicates with user            |
| TemporalAgent      | Measures consistency across video frames                     |
| SemanticAgent      | Compares frame semantics against video description           |
| DynamicsAgent      | Detects abrupt or unnatural changes in motion or objects     |
| GeneralizationAgent| Tests performance across different domains or styles         |
| ReportingAgent     | It reports after gathering all necessary evaluation results  |
   
##  Metrics





## Output
{
  "video_name": "your_video.mp4",
  "temporal_coherence": 0.82,
  "semantic_consistency": 0.75,
  "dynamic_scene_score": 0.68,
  "generalization_score": 0.80
}

## Interactive Workflow
1. Select metrics from available options
2. Provide video description/prompt
3. System processes video automatically
4. View comprehensive JSON report
