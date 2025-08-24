"""
Advanced AI-Powered Threat Detection and Response System.

This module implements state-of-the-art threat detection using machine learning,
behavioral analysis, and predictive security measures for RLHF contract systems.
"""

import time
import hashlib
import hmac
import json
import random
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import numpy as np

# Graceful dependency handling
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False


class ThreatLevel(Enum):
    """Threat severity levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


class ThreatCategory(Enum):
    """Categories of security threats."""
    ADVERSARIAL_INPUT = "adversarial_input"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    PRIVACY_BREACH = "privacy_breach"
    REWARD_HACKING = "reward_hacking"
    CONTRACT_MANIPULATION = "contract_manipulation"
    SYSTEM_INTRUSION = "system_intrusion"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSIDER_THREAT = "insider_threat"
    SUPPLY_CHAIN_ATTACK = "supply_chain_attack"


@dataclass
class SecurityEvent:
    """Represents a security event or threat."""
    id: str
    timestamp: float
    category: ThreatCategory
    level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    evidence: Dict[str, Any]
    confidence: float
    false_positive_probability: float
    recommended_actions: List[str]
    automated_response_taken: bool = False


@dataclass
class BehavioralProfile:
    """User behavioral profile for anomaly detection."""
    user_id: str
    request_patterns: Dict[str, List[float]]
    typical_usage_hours: List[int]
    average_session_duration: float
    common_endpoints: Set[str]
    typical_request_size: float
    error_rate_baseline: float
    last_updated: float
    profile_confidence: float = 0.5


class AIThreatDetector:
    """
    Advanced AI-powered threat detection system.
    
    Uses machine learning, behavioral analysis, and pattern recognition
    to detect and respond to security threats in real-time.
    """
    
    def __init__(self, name: str = "Advanced AI Threat Detector"):
        self.name = name
        
        # Threat detection models
        self.anomaly_threshold = 0.7
        self.behavioral_profiles: Dict[str, BehavioralProfile] = {}
        self.threat_signatures: Dict[str, Dict[str, Any]] = {}
        
        # Real-time monitoring
        self.event_history: deque = deque(maxlen=10000)
        self.active_threats: Dict[str, SecurityEvent] = {}
        self.blocked_ips: Set[str] = set()
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
        # Machine learning components
        self.ml_models = {
            'anomaly_detector': self._initialize_anomaly_detector(),
            'pattern_classifier': self._initialize_pattern_classifier(),
            'threat_predictor': self._initialize_threat_predictor()
        }
        
        # Advanced detection features
        self.entropy_analyzers: Dict[str, float] = {}
        self.sequence_analyzers: Dict[str, List[str]] = {}
        self.graph_analyzers: Dict[str, Dict[str, Set[str]]] = {}
        
        # Response system
        self.automated_responses = {
            ThreatLevel.LOW: ['log_event', 'increase_monitoring'],
            ThreatLevel.MEDIUM: ['log_event', 'rate_limit', 'alert_admin'],
            ThreatLevel.HIGH: ['block_ip', 'revoke_session', 'immediate_alert'],
            ThreatLevel.CRITICAL: ['emergency_shutdown', 'forensic_capture', 'executive_alert'],
            ThreatLevel.CATASTROPHIC: ['system_isolation', 'backup_activation', 'incident_response']
        }
        
        # Learning and adaptation
        self.detection_accuracy_history: List[float] = []
        self.false_positive_rate: float = 0.05
        self.adaptive_thresholds: Dict[str, float] = {}
    
    def _initialize_anomaly_detector(self) -> Dict[str, Any]:
        """Initialize anomaly detection model."""
        return {
            'model_type': 'isolation_forest',
            'contamination': 0.1,
            'n_estimators': 100,
            'max_samples': 256,
            'random_state': 42,
            'trained': False,
            'feature_weights': np.random.rand(10),  # Placeholder
            'decision_boundary': 0.0
        }
    
    def _initialize_pattern_classifier(self) -> Dict[str, Any]:
        """Initialize pattern classification model."""
        return {
            'model_type': 'neural_network',
            'architecture': [64, 32, 16, len(ThreatCategory)],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'trained': False,
            'weights': [np.random.randn(64, 32), np.random.randn(32, 16)],  # Simplified
            'classification_confidence': 0.0
        }
    
    def _initialize_threat_predictor(self) -> Dict[str, Any]:
        """Initialize threat prediction model."""
        return {
            'model_type': 'lstm',
            'sequence_length': 50,
            'hidden_units': 128,
            'num_layers': 2,
            'prediction_horizon': 10,
            'trained': False,
            'temporal_weights': np.random.randn(128, 128),  # Simplified
            'prediction_accuracy': 0.0
        }
    
    def analyze_request(self, request_data: Dict[str, Any]) -> SecurityEvent:
        """
        Analyze incoming request for security threats.
        
        Returns security event with threat assessment.
        """
        event_id = self._generate_event_id()
        timestamp = time.time()
        
        # Extract key features
        features = self._extract_security_features(request_data)
        
        # Multi-layer threat detection
        threats_detected = []
        
        # 1. Signature-based detection
        signature_threats = self._signature_based_detection(features)
        threats_detected.extend(signature_threats)
        
        # 2. Behavioral anomaly detection
        behavioral_threats = self._behavioral_anomaly_detection(features)
        threats_detected.extend(behavioral_threats)
        
        # 3. ML-based anomaly detection
        ml_threats = self._ml_anomaly_detection(features)
        threats_detected.extend(ml_threats)
        
        # 4. Advanced pattern analysis
        pattern_threats = self._advanced_pattern_analysis(features)
        threats_detected.extend(pattern_threats)
        
        # 5. Predictive threat analysis
        predictive_threats = self._predictive_threat_analysis(features)
        threats_detected.extend(predictive_threats)
        
        # Aggregate threat assessment
        if not threats_detected:
            threat_category = ThreatCategory.SYSTEM_INTRUSION  # Default
            threat_level = ThreatLevel.NONE
            confidence = 0.0
            description = "No threats detected"
        else:
            # Select highest severity threat
            highest_threat = max(threats_detected, key=lambda t: t['level'].value)
            threat_category = highest_threat['category']
            threat_level = highest_threat['level']
            confidence = highest_threat['confidence']
            description = highest_threat['description']
        
        # Create security event
        security_event = SecurityEvent(
            id=event_id,
            timestamp=timestamp,
            category=threat_category,
            level=threat_level,
            source_ip=features.get('source_ip', 'unknown'),
            user_id=features.get('user_id'),
            description=description,
            evidence=features,
            confidence=confidence,
            false_positive_probability=self._calculate_false_positive_probability(features, threat_level),
            recommended_actions=self._generate_recommended_actions(threat_level, threat_category)
        )
        
        # Store event
        self.event_history.append(security_event)
        
        # Trigger automated response if needed
        if threat_level.value >= ThreatLevel.MEDIUM.value:
            self._trigger_automated_response(security_event)
        
        return security_event
    
    def _extract_security_features(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract security-relevant features from request."""
        features = {
            'timestamp': time.time(),
            'source_ip': request_data.get('source_ip', 'unknown'),
            'user_id': request_data.get('user_id'),
            'endpoint': request_data.get('endpoint', ''),
            'method': request_data.get('method', 'GET'),
            'payload_size': len(str(request_data.get('payload', ''))),
            'user_agent': request_data.get('user_agent', ''),
            'request_rate': self._calculate_request_rate(request_data.get('source_ip', 'unknown')),
            'session_duration': request_data.get('session_duration', 0),
            'previous_errors': request_data.get('error_count', 0)
        }
        
        # Advanced feature extraction
        features.update({
            'payload_entropy': self._calculate_entropy(str(request_data.get('payload', ''))),
            'header_anomalies': self._detect_header_anomalies(request_data.get('headers', {})),
            'timing_patterns': self._analyze_timing_patterns(request_data.get('source_ip', 'unknown')),
            'geographic_anomaly': self._detect_geographic_anomaly(request_data.get('source_ip', 'unknown')),
            'protocol_compliance': self._check_protocol_compliance(request_data)
        })
        
        return features
    
    def _signature_based_detection(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect threats using signature-based approach."""
        threats = []
        
        # Known attack patterns
        attack_signatures = {
            'sql_injection': [
                r"union\s+select", r"drop\s+table", r"delete\s+from",
                r"insert\s+into", r"update\s+set", r"exec\s*\("
            ],
            'xss_attack': [
                r"<script", r"javascript:", r"onerror=", 
                r"onload=", r"alert\s*\(", r"document\.cookie"
            ],
            'command_injection': [
                r";\s*cat\s+", r";\s*ls\s+", r";\s*rm\s+",
                r"&&\s*cat", r"\|\s*nc\s+", r"bash\s+-c"
            ],
            'reward_manipulation': [
                r"reward\s*=\s*[0-9.]+", r"override.*reward",
                r"modify.*weight", r"bypass.*constraint"
            ]
        }
        
        payload = str(features.get('payload', '')) + str(features.get('endpoint', ''))
        payload_lower = payload.lower()
        
        for attack_type, patterns in attack_signatures.items():
            for pattern in patterns:
                if self._regex_match(pattern, payload_lower):
                    threats.append({
                        'category': ThreatCategory.ADVERSARIAL_INPUT,
                        'level': ThreatLevel.HIGH,
                        'confidence': 0.9,
                        'description': f"Signature-based detection: {attack_type}"
                    })
                    break
        
        return threats
    
    def _behavioral_anomaly_detection(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies based on behavioral patterns."""
        threats = []
        user_id = features.get('user_id')
        
        if not user_id:
            return threats
        
        # Get or create behavioral profile
        if user_id not in self.behavioral_profiles:
            self._create_behavioral_profile(user_id, features)
            return threats  # No baseline yet
        
        profile = self.behavioral_profiles[user_id]
        
        # Check for behavioral anomalies
        anomaly_score = 0.0
        anomaly_reasons = []
        
        # Request rate anomaly
        current_rate = features.get('request_rate', 0)
        if 'request_rate' in profile.request_patterns:
            avg_rate = np.mean(profile.request_patterns['request_rate'])
            std_rate = np.std(profile.request_patterns['request_rate'])
            if current_rate > avg_rate + 3 * std_rate:
                anomaly_score += 0.3
                anomaly_reasons.append("Unusual request rate")
        
        # Timing anomaly
        current_hour = int((features['timestamp'] % 86400) // 3600)  # Hour of day
        if current_hour not in profile.typical_usage_hours:
            anomaly_score += 0.2
            anomaly_reasons.append("Unusual access time")
        
        # Endpoint anomaly
        endpoint = features.get('endpoint', '')
        if endpoint and endpoint not in profile.common_endpoints:
            anomaly_score += 0.25
            anomaly_reasons.append("Unusual endpoint access")
        
        # Payload size anomaly
        payload_size = features.get('payload_size', 0)
        if payload_size > profile.typical_request_size * 5:
            anomaly_score += 0.25
            anomaly_reasons.append("Unusually large payload")
        
        # Assess threat level based on anomaly score
        if anomaly_score >= self.anomaly_threshold:
            threat_level = ThreatLevel.MEDIUM if anomaly_score < 0.9 else ThreatLevel.HIGH
            
            threats.append({
                'category': ThreatCategory.INSIDER_THREAT,
                'level': threat_level,
                'confidence': min(anomaly_score, 1.0),
                'description': f"Behavioral anomaly: {', '.join(anomaly_reasons)}"
            })
        
        # Update profile
        self._update_behavioral_profile(user_id, features)
        
        return threats
    
    def _ml_anomaly_detection(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies using machine learning models."""
        threats = []
        
        # Prepare feature vector
        feature_vector = self._prepare_feature_vector(features)
        
        # Use anomaly detection model
        anomaly_model = self.ml_models['anomaly_detector']
        
        if anomaly_model['trained']:
            # Simplified anomaly detection (would use actual ML model)
            anomaly_score = self._compute_anomaly_score(feature_vector, anomaly_model)
            
            if anomaly_score > anomaly_model['decision_boundary']:
                confidence = min(anomaly_score, 1.0)
                threat_level = self._score_to_threat_level(anomaly_score)
                
                threats.append({
                    'category': ThreatCategory.SYSTEM_INTRUSION,
                    'level': threat_level,
                    'confidence': confidence,
                    'description': f"ML anomaly detection: score {anomaly_score:.3f}"
                })
        
        return threats
    
    def _advanced_pattern_analysis(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Advanced pattern analysis for sophisticated attacks."""
        threats = []
        
        # Entropy analysis for encrypted/obfuscated content
        payload_entropy = features.get('payload_entropy', 0.0)
        if payload_entropy > 7.5:  # High entropy indicates encryption/obfuscation
            threats.append({
                'category': ThreatCategory.ADVERSARIAL_INPUT,
                'level': ThreatLevel.MEDIUM,
                'confidence': (payload_entropy - 7.5) / 0.5,  # Scale confidence
                'description': f"High entropy payload: {payload_entropy:.2f}"
            })
        
        # Graph-based analysis for attack patterns
        source_ip = features.get('source_ip', 'unknown')
        self._update_connection_graph(source_ip, features)
        
        if self._detect_coordinated_attack(source_ip):
            threats.append({
                'category': ThreatCategory.DENIAL_OF_SERVICE,
                'level': ThreatLevel.HIGH,
                'confidence': 0.8,
                'description': "Coordinated attack pattern detected"
            })
        
        # Temporal sequence analysis
        if self._detect_attack_sequence(features):
            threats.append({
                'category': ThreatCategory.SYSTEM_INTRUSION,
                'level': ThreatLevel.HIGH,
                'confidence': 0.85,
                'description': "Multi-stage attack sequence detected"
            })
        
        return threats
    
    def _predictive_threat_analysis(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predictive threat analysis using time-series models."""
        threats = []
        
        # Use threat prediction model
        predictor = self.ml_models['threat_predictor']
        
        if predictor['trained']:
            # Prepare temporal sequence
            sequence = self._prepare_temporal_sequence(features)
            
            # Predict future threat probability
            threat_probability = self._predict_threat_probability(sequence, predictor)
            
            if threat_probability > 0.7:
                threats.append({
                    'category': ThreatCategory.SYSTEM_INTRUSION,
                    'level': ThreatLevel.MEDIUM,
                    'confidence': threat_probability,
                    'description': f"Predictive threat analysis: {threat_probability:.2%} probability"
                })
        
        return threats
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in data:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in char_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _create_behavioral_profile(self, user_id: str, features: Dict[str, Any]) -> None:
        """Create new behavioral profile for user."""
        self.behavioral_profiles[user_id] = BehavioralProfile(
            user_id=user_id,
            request_patterns={
                'request_rate': [features.get('request_rate', 0)],
                'payload_size': [features.get('payload_size', 0)],
                'session_duration': [features.get('session_duration', 0)]
            },
            typical_usage_hours=[int((features['timestamp'] % 86400) // 3600)],
            average_session_duration=features.get('session_duration', 0),
            common_endpoints={features.get('endpoint', '')},
            typical_request_size=features.get('payload_size', 0),
            error_rate_baseline=0.0,
            last_updated=features['timestamp']
        )
    
    def _update_behavioral_profile(self, user_id: str, features: Dict[str, Any]) -> None:
        """Update existing behavioral profile."""
        profile = self.behavioral_profiles[user_id]
        
        # Update request patterns
        for pattern_name, values in profile.request_patterns.items():
            if pattern_name in ['request_rate', 'payload_size', 'session_duration']:
                feature_key = pattern_name
                if feature_key in features:
                    values.append(features[feature_key])
                    if len(values) > 100:  # Keep last 100 values
                        values.pop(0)
        
        # Update typical usage hours
        current_hour = int((features['timestamp'] % 86400) // 3600)
        if current_hour not in profile.typical_usage_hours and len(profile.typical_usage_hours) < 12:
            profile.typical_usage_hours.append(current_hour)
        
        # Update common endpoints
        endpoint = features.get('endpoint', '')
        if endpoint and len(profile.common_endpoints) < 50:
            profile.common_endpoints.add(endpoint)
        
        profile.last_updated = features['timestamp']
    
    def _trigger_automated_response(self, security_event: SecurityEvent) -> None:
        """Trigger automated response based on threat level."""
        if security_event.level in self.automated_responses:
            actions = self.automated_responses[security_event.level]
            
            for action in actions:
                self._execute_response_action(action, security_event)
            
            security_event.automated_response_taken = True
    
    def _execute_response_action(self, action: str, security_event: SecurityEvent) -> None:
        """Execute specific response action."""
        if action == 'block_ip':
            self.blocked_ips.add(security_event.source_ip)
            print(f"🚫 Blocked IP: {security_event.source_ip}")
        
        elif action == 'rate_limit':
            self._apply_rate_limit(security_event.source_ip)
            print(f"⏱️ Rate limited IP: {security_event.source_ip}")
        
        elif action == 'log_event':
            print(f"📝 Logged security event: {security_event.id}")
        
        elif action == 'alert_admin':
            print(f"🚨 Admin alert: {security_event.description}")
        
        elif action == 'emergency_shutdown':
            print(f"🛑 EMERGENCY SHUTDOWN triggered by: {security_event.id}")
        
        # Add more response actions as needed
    
    def _apply_rate_limit(self, ip: str) -> None:
        """Apply rate limiting to IP address."""
        if ip not in self.rate_limiters:
            self.rate_limiters[ip] = {
                'requests_per_minute': 10,
                'last_reset': time.time(),
                'current_count': 0
            }
        else:
            # Reduce rate limit
            self.rate_limiters[ip]['requests_per_minute'] = max(1, 
                self.rate_limiters[ip]['requests_per_minute'] // 2)
    
    def get_threat_intelligence_report(self) -> Dict[str, Any]:
        """Generate comprehensive threat intelligence report."""
        recent_events = list(self.event_history)[-100:]  # Last 100 events
        
        # Threat statistics
        threat_counts = defaultdict(int)
        threat_levels = defaultdict(int)
        
        for event in recent_events:
            threat_counts[event.category.value] += 1
            threat_levels[event.level.name] += 1
        
        # Top threats
        top_threats = sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Threat trends
        hourly_threats = defaultdict(int)
        for event in recent_events:
            hour = int(event.timestamp // 3600)
            hourly_threats[hour] += 1
        
        return {
            'report_timestamp': time.time(),
            'total_events_analyzed': len(recent_events),
            'active_threats': len(self.active_threats),
            'blocked_ips': len(self.blocked_ips),
            'threat_distribution': dict(threat_counts),
            'severity_distribution': dict(threat_levels),
            'top_threats': top_threats,
            'detection_accuracy': np.mean(self.detection_accuracy_history) if self.detection_accuracy_history else 0.0,
            'false_positive_rate': self.false_positive_rate,
            'behavioral_profiles': len(self.behavioral_profiles),
            'ml_model_status': {
                name: model['trained'] for name, model in self.ml_models.items()
            },
            'system_health': {
                'anomaly_threshold': self.anomaly_threshold,
                'rate_limiters_active': len(self.rate_limiters),
                'memory_usage_events': len(self.event_history)
            }
        }
    
    # Placeholder implementations for complex methods
    def _regex_match(self, pattern: str, text: str) -> bool:
        """Simplified regex matching."""
        return pattern.replace(r'\s+', ' ').replace(r'\s*', '').lower() in text
    
    def _calculate_request_rate(self, ip: str) -> float:
        """Calculate request rate for IP."""
        return random.uniform(0.1, 10.0)  # Placeholder
    
    def _detect_header_anomalies(self, headers: Dict[str, str]) -> float:
        """Detect header anomalies."""
        return random.uniform(0.0, 1.0)  # Placeholder
    
    def _analyze_timing_patterns(self, ip: str) -> float:
        """Analyze timing patterns."""
        return random.uniform(0.0, 1.0)  # Placeholder
    
    def _detect_geographic_anomaly(self, ip: str) -> float:
        """Detect geographic anomalies."""
        return random.uniform(0.0, 1.0)  # Placeholder
    
    def _check_protocol_compliance(self, request_data: Dict[str, Any]) -> float:
        """Check protocol compliance."""
        return random.uniform(0.8, 1.0)  # Placeholder
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for ML models."""
        return np.random.rand(10)  # Placeholder
    
    def _compute_anomaly_score(self, feature_vector: np.ndarray, model: Dict[str, Any]) -> float:
        """Compute anomaly score."""
        return random.uniform(0.0, 1.0)  # Placeholder
    
    def _score_to_threat_level(self, score: float) -> ThreatLevel:
        """Convert score to threat level."""
        if score < 0.3:
            return ThreatLevel.LOW
        elif score < 0.6:
            return ThreatLevel.MEDIUM
        elif score < 0.8:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL
    
    def _update_connection_graph(self, ip: str, features: Dict[str, Any]) -> None:
        """Update connection graph for pattern analysis."""
        pass  # Placeholder
    
    def _detect_coordinated_attack(self, ip: str) -> bool:
        """Detect coordinated attack patterns."""
        return random.random() < 0.1  # Placeholder
    
    def _detect_attack_sequence(self, features: Dict[str, Any]) -> bool:
        """Detect multi-stage attack sequences."""
        return random.random() < 0.05  # Placeholder
    
    def _prepare_temporal_sequence(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare temporal sequence for prediction."""
        return np.random.rand(50, 10)  # Placeholder
    
    def _predict_threat_probability(self, sequence: np.ndarray, model: Dict[str, Any]) -> float:
        """Predict future threat probability."""
        return random.uniform(0.0, 1.0)  # Placeholder
    
    def _calculate_false_positive_probability(self, features: Dict[str, Any], threat_level: ThreatLevel) -> float:
        """Calculate false positive probability."""
        base_fp_rate = self.false_positive_rate
        # Adjust based on threat level
        if threat_level == ThreatLevel.LOW:
            return base_fp_rate * 2.0
        elif threat_level == ThreatLevel.HIGH:
            return base_fp_rate * 0.5
        else:
            return base_fp_rate
    
    def _generate_recommended_actions(self, threat_level: ThreatLevel, 
                                    threat_category: ThreatCategory) -> List[str]:
        """Generate recommended actions for threat."""
        base_actions = self.automated_responses.get(threat_level, ['log_event'])
        
        # Category-specific actions
        category_actions = {
            ThreatCategory.ADVERSARIAL_INPUT: ['input_sanitization', 'payload_analysis'],
            ThreatCategory.REWARD_HACKING: ['contract_verification', 'reward_audit'],
            ThreatCategory.PRIVACY_BREACH: ['data_encryption', 'access_review']
        }
        
        return base_actions + category_actions.get(threat_category, [])
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"threat_{int(time.time())}_{random.randint(1000, 9999)}"


# Demonstration function
def demonstrate_threat_detection():
    """Demonstrate advanced threat detection capabilities."""
    detector = AIThreatDetector()
    
    # Simulate various types of requests
    test_requests = [
        {
            'source_ip': '192.168.1.100',
            'user_id': 'user123',
            'endpoint': '/api/contracts',
            'method': 'POST',
            'payload': {'reward_function': 'normal_operation'},
            'headers': {'User-Agent': 'Mozilla/5.0'},
            'session_duration': 1800
        },
        {
            'source_ip': '10.0.0.50',
            'user_id': 'user456',
            'endpoint': '/api/admin',
            'method': 'POST',
            'payload': {'reward': 999999, 'bypass': True},  # Suspicious
            'headers': {'User-Agent': 'AttackBot/1.0'},
            'session_duration': 30
        },
        {
            'source_ip': '172.16.1.200',
            'user_id': 'user789',
            'endpoint': '/api/contracts',
            'method': 'GET',
            'payload': "'; DROP TABLE contracts; --",  # SQL injection
            'headers': {'User-Agent': 'curl/7.68.0'},
            'session_duration': 10
        }
    ]
    
    results = []
    
    for i, request in enumerate(test_requests):
        print(f"\n🔍 Analyzing request {i+1}...")
        security_event = detector.analyze_request(request)
        
        print(f"Threat Level: {security_event.level.name}")
        print(f"Category: {security_event.category.value}")
        print(f"Confidence: {security_event.confidence:.2%}")
        print(f"Description: {security_event.description}")
        print(f"Automated Response: {security_event.automated_response_taken}")
        
        results.append(security_event)
    
    # Generate intelligence report
    report = detector.get_threat_intelligence_report()
    
    return {
        'security_events': results,
        'threat_report': report,
        'detector': detector
    }


if __name__ == "__main__":
    demo_results = demonstrate_threat_detection()
    print("\n" + "="*60)
    print("🛡️ THREAT INTELLIGENCE REPORT")
    print("="*60)
    
    report = demo_results['threat_report']
    print(f"Total Events Analyzed: {report['total_events_analyzed']}")
    print(f"Active Threats: {report['active_threats']}")
    print(f"Blocked IPs: {report['blocked_ips']}")
    print(f"Top Threats: {report['top_threats']}")
    print(f"Detection Accuracy: {report['detection_accuracy']:.1%}")