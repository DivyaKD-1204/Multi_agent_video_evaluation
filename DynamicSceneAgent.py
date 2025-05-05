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
