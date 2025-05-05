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





## Output
=== VIDEO EVALUATION REPORT ===
{
  "metadata": {
    "system_version": "1.0",
    "evaluation_date": "<timestamp>"
  },
  "quantitative_metrics": {
    "temporal": {
      "optical_flow": "<optical_flow_value>",
      "frame_difference": "<frame_difference_value>",
      "ssim": "<ssim_value>",
      "edge_consistency": "<edge_consistency_value>"
    },
    "semantic": {
      "score": "<semantic_score>",
      "summary": "<generated_summary_from_video>",
      "reference": "<user_provided_reference_text>"
    },
    "dynamic": {
      "scene_change_ratio": "<scene_change_ratio>",
      "avg_optical_flow": "<avg_optical_flow>",
      "flow_variance": "<flow_variance>",
      "brightness_change_frequency": "<brightness_change_frequency>",
      "avg_object_movement": "<avg_object_movement>",
      "object_movement_variance": "<object_movement_variance>",
      "assessment": "<dynamic_assessment>"
    },
    "generalization": {
      "novelty_score": "<novelty_score>",
      "assessment": "<novelty_assessment>"
    }
  },
  "qualitative_assessment": {
    "temporal_quality": "<temporal_quality_label>",
    "semantic_alignment": "<semantic_alignment_label>",
    "scene_continuity": "<scene_continuity_label>",
    "content_familiarity": "<content_familiarity_label>"
  }
}

Report saved to 'video_evaluation_report.json'

## Interactive Workflow
1. Select metrics from available options
2. Provide video description/prompt
3. System processes video automatically
4. View comprehensive JSON report
