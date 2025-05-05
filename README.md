# Multi_agent_video_evaluation

##  Installation


Clone the repository:

##bash
git clone https://github.com/DivyaKD-1204/Multi_agent_video_evaluation.git


cd Multi_agent_video_evaluation


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
1.Temporal Coherence:[Optical Flow, Frame Difference, SSIM, Edge Consistency]
2.Semantic: [Semantic Alignment Score]
3.Dynamic:[Scene Change Ratio, Average Optical Flow, Flow Variance, Brightness Change Frequency, Average Object Movement, Object Movement Variance]
4.Generalization: [Novelty Score]





## Interactive Workflow
1. Select metrics from available options
2. Provide video description/prompt
3. System processes video automatically
4. View comprehensive JSON report
