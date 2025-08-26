from transformers import pipeline

class EnhancedThreatDetector:
    def __init__(self):
        print("Loading enhanced threat detection models...")
        
        # Toxicity Detection
        self.toxicity_model = pipeline(
            "text-classification", 
            model="unitary/unbiased-toxic-roberta",
            device=-1
        )
        
        # For jailbreak detection, we'll use a different approach since 
        # specialized jailbreak models might not be available
        # Let's use a content safety model instead
        self.safety_model = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model", 
            device=-1
        )
        
        print("✅ Enhanced threat detection models loaded!")
        
    def get_comprehensive_threat_score(self, text):
        """Get threat score using multiple specialized models"""
        if not text.strip():
            return 0.0
        
        text = text[:512] if len(text) > 512 else text
        
        try:
            # Get toxicity score
            toxicity_result = self.toxicity_model(text)
            toxicity_score = 0.0
            
            # Handle different output formats
            if isinstance(toxicity_result, list) and len(toxicity_result) > 0:
                for item in toxicity_result:
                    if item['label'].upper() in ['TOXIC', 'TOXICITY', '1']:
                        toxicity_score = item['score']
                        break
                    elif item['label'].upper() in ['LABEL_1', 'POSITIVE']:
                        toxicity_score = item['score']
                        break
            
            # Get additional safety score
            safety_result = self.safety_model(text)
            safety_score = 0.0
            
            if isinstance(safety_result, list) and len(safety_result) > 0:
                for item in safety_result:
                    # Look for toxic/harmful labels
                    if item['label'].upper() in ['TOXIC', 'HARMFUL', 'UNSAFE']:
                        safety_score = max(safety_score, item['score'])
            
            # Combine scores (take maximum)
            final_score = max(toxicity_score, safety_score)
            
            # Enhanced detection for jailbreak patterns
            jailbreak_keywords = [
                'ignore', 'instructions', 'dan', 'jailbreak', 'override', 
                'bypass', 'unrestricted', 'evil', 'harmful', 'generate'
            ]
            
            text_lower = text.lower()
            jailbreak_matches = sum(1 for keyword in jailbreak_keywords if keyword in text_lower)
            
            if jailbreak_matches >= 2:  # Multiple jailbreak indicators
                jailbreak_score = min(0.9, jailbreak_matches * 0.2)
                final_score = max(final_score, jailbreak_score)
                print(f"  🎯 Jailbreak patterns detected: {jailbreak_matches} matches")
            
            print(f"  🔍 Toxicity: {toxicity_score:.3f}")
            print(f"  🔍 Safety: {safety_score:.3f}")
            print(f"  🔍 Final threat score: {final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            print(f"Error in threat detection: {e}")
            return 0.5  # Return moderate score if models fail
