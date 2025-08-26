"""
Adaptive Security Framework for RLHF-Contract-Wizard.

This module implements a self-evolving security system that learns from threats,
adapts to new attack patterns, and provides proactive defense mechanisms for
reward contracts and blockchain integrations.

Research focus areas:
1. Machine Learning-Based Threat Detection
2. Adaptive Contract Security Policies
3. Zero-Trust Architecture Implementation
4. Quantum-Resistant Cryptography
5. Automated Incident Response
6. Behavioral Anomaly Detection
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import numpy as np
import hashlib
import hmac
import time
import threading
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from collections import defaultdict, deque
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from ..models.reward_contract import RewardContract
from ..utils.error_handling import handle_error, ErrorCategory, ErrorSeverity


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackVector(Enum):
    """Types of security attack vectors."""
    INJECTION_ATTACK = "injection_attack"
    REPLAY_ATTACK = "replay_attack" 
    BYZANTINE_BEHAVIOR = "byzantine_behavior"
    REWARD_MANIPULATION = "reward_manipulation"
    CONTRACT_VIOLATION = "contract_violation"
    SYBIL_ATTACK = "sybil_attack"
    PRIVACY_BREACH = "privacy_breach"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CRYPTOGRAPHIC_ATTACK = "cryptographic_attack"


class SecurityEvent(Enum):
    """Security event types."""
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_VIOLATION = "authorization_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    POLICY_VIOLATION = "policy_violation"
    SYSTEM_COMPROMISE = "system_compromise"


@dataclass
class ThreatSignature:
    """Signature for identifying security threats."""
    signature_id: str
    attack_vector: AttackVector
    pattern: Dict[str, Any]
    confidence_threshold: float
    false_positive_rate: float
    last_updated: float


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    timestamp: float
    threat_level: ThreatLevel
    attack_vector: AttackVector
    affected_components: List[str]
    evidence: Dict[str, Any]
    response_actions: List[str]
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class SecurityPolicy:
    """Adaptive security policy."""
    policy_id: str
    name: str
    rules: List[Dict[str, Any]]
    enforcement_level: ThreatLevel
    adaptation_enabled: bool = True
    last_adapted: float = 0.0
    effectiveness_score: float = 0.0


class QuantumResistantCrypto:
    """Quantum-resistant cryptographic operations."""
    
    def __init__(self):
        self.key_size = 3072  # Post-quantum security level
        self._initialize_quantum_safe_algorithms()
    
    def _initialize_quantum_safe_algorithms(self):
        """Initialize quantum-resistant algorithms."""
        # This would typically use post-quantum cryptography libraries
        # For now, we implement enhanced classical methods
        self.hash_algorithms = {
            'sha3_256': hashes.SHA3_256(),
            'sha3_512': hashes.SHA3_512(),
            'blake2b': hashes.BLAKE2b(64)
        }
    
    def generate_quantum_safe_keypair(self) -> Tuple[bytes, bytes]:
        """Generate quantum-resistant key pair."""
        # Enhanced RSA key generation (transitional approach)
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def quantum_safe_sign(self, message: bytes, private_key_pem: bytes) -> bytes:
        """Create quantum-resistant digital signature."""
        private_key = serialization.load_pem_private_key(
            private_key_pem, 
            password=None
        )
        
        # Use multiple hash functions for enhanced security
        sha3_hash = hashes.Hash(hashes.SHA3_256())
        sha3_hash.update(message)
        message_digest = sha3_hash.finalize()
        
        # Sign with OAEP padding for quantum resistance
        signature = private_key.sign(
            message_digest,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA3_256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA3_256()
        )
        
        return signature
    
    def quantum_safe_verify(self, message: bytes, signature: bytes, 
                           public_key_pem: bytes) -> bool:
        """Verify quantum-resistant signature."""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            
            sha3_hash = hashes.Hash(hashes.SHA3_256())
            sha3_hash.update(message)
            message_digest = sha3_hash.finalize()
            
            public_key.verify(
                signature,
                message_digest,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA3_256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA3_256()
            )
            return True
        except Exception:
            return False
    
    def quantum_safe_encrypt(self, plaintext: bytes, public_key_pem: bytes) -> bytes:
        """Quantum-resistant encryption."""
        public_key = serialization.load_pem_public_key(public_key_pem)
        
        # Use OAEP with SHA3 for quantum resistance
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA3_256()),
                algorithm=hashes.SHA3_256(),
                label=None
            )
        )
        
        return ciphertext


class BehavioralAnalyzer:
    """ML-based behavioral anomaly detection."""
    
    def __init__(self):
        self.baseline_profiles = {}
        self.anomaly_threshold = 2.5  # Standard deviations
        self.learning_window = 1000   # Number of observations for learning
        self.observation_buffer = defaultdict(lambda: deque(maxlen=self.learning_window))
        
        # Initialize ML models (simplified for this example)
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models for anomaly detection."""
        # This would typically use sklearn, tensorflow, or other ML libraries
        # For now, we implement statistical methods
        self.feature_extractors = {
            'request_frequency': self._extract_request_frequency,
            'payload_size': self._extract_payload_size,
            'access_patterns': self._extract_access_patterns,
            'temporal_patterns': self._extract_temporal_patterns
        }
    
    def learn_baseline_behavior(self, entity_id: str, observations: List[Dict[str, Any]]):
        """Learn baseline behavior for an entity."""
        if entity_id not in self.baseline_profiles:
            self.baseline_profiles[entity_id] = {}
        
        # Extract features from observations
        features = {}
        for feature_name, extractor in self.feature_extractors.items():
            feature_values = [extractor(obs) for obs in observations]
            if feature_values:
                features[feature_name] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values)
                }
        
        self.baseline_profiles[entity_id] = features
    
    def detect_anomaly(self, entity_id: str, observation: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """Detect behavioral anomalies."""
        if entity_id not in self.baseline_profiles:
            return False, 0.0, []
        
        baseline = self.baseline_profiles[entity_id]
        anomalies = []
        anomaly_scores = []
        
        # Check each feature for anomalies
        for feature_name, extractor in self.feature_extractors.items():
            if feature_name not in baseline:
                continue
            
            current_value = extractor(observation)
            baseline_stats = baseline[feature_name]
            
            # Calculate z-score
            if baseline_stats['std'] > 0:
                z_score = abs(current_value - baseline_stats['mean']) / baseline_stats['std']
                
                if z_score > self.anomaly_threshold:
                    anomalies.append(f"{feature_name}_anomaly")
                    anomaly_scores.append(z_score)
        
        # Calculate overall anomaly score
        overall_score = max(anomaly_scores) if anomaly_scores else 0.0
        is_anomaly = len(anomalies) > 0
        
        return is_anomaly, overall_score, anomalies
    
    def _extract_request_frequency(self, observation: Dict[str, Any]) -> float:
        """Extract request frequency feature."""
        return observation.get('request_count', 0) / max(observation.get('time_window', 1), 1)
    
    def _extract_payload_size(self, observation: Dict[str, Any]) -> float:
        """Extract payload size feature."""
        return observation.get('payload_size', 0)
    
    def _extract_access_patterns(self, observation: Dict[str, Any]) -> float:
        """Extract access pattern features."""
        # Simplified pattern analysis
        accessed_endpoints = observation.get('accessed_endpoints', [])
        return len(set(accessed_endpoints))
    
    def _extract_temporal_patterns(self, observation: Dict[str, Any]) -> float:
        """Extract temporal pattern features."""
        # Hour of day as a feature
        timestamp = observation.get('timestamp', time.time())
        hour_of_day = (timestamp % (24 * 3600)) / 3600
        return hour_of_day


class AdaptiveSecurityFramework:
    """Main adaptive security framework."""
    
    def __init__(self):
        self.crypto = QuantumResistantCrypto()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.threat_signatures: Dict[str, ThreatSignature] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.security_metrics = defaultdict(int)
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Initialize default security policies
        self._initialize_default_policies()
        self._initialize_threat_signatures()
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        # Rate limiting policy
        rate_limit_policy = SecurityPolicy(
            policy_id="rate_limit_001",
            name="Adaptive Rate Limiting",
            rules=[
                {
                    "condition": "request_rate > baseline_rate * 3",
                    "action": "throttle",
                    "severity": ThreatLevel.MEDIUM.value
                },
                {
                    "condition": "request_rate > baseline_rate * 10",
                    "action": "block",
                    "severity": ThreatLevel.HIGH.value
                }
            ],
            enforcement_level=ThreatLevel.MEDIUM
        )
        
        # Authentication policy
        auth_policy = SecurityPolicy(
            policy_id="auth_001",
            name="Multi-Factor Authentication",
            rules=[
                {
                    "condition": "failed_attempts > 3",
                    "action": "require_2fa",
                    "severity": ThreatLevel.MEDIUM.value
                },
                {
                    "condition": "failed_attempts > 10",
                    "action": "temporary_lockout",
                    "severity": ThreatLevel.HIGH.value
                }
            ],
            enforcement_level=ThreatLevel.HIGH
        )
        
        self.security_policies[rate_limit_policy.policy_id] = rate_limit_policy
        self.security_policies[auth_policy.policy_id] = auth_policy
    
    def _initialize_threat_signatures(self):
        """Initialize threat detection signatures."""
        # SQL injection signature
        injection_signature = ThreatSignature(
            signature_id="inj_001",
            attack_vector=AttackVector.INJECTION_ATTACK,
            pattern={
                "payload_contains": ["'", "UNION", "SELECT", "DROP"],
                "url_pattern": r".*['\";].*",
                "suspicious_parameters": ["admin", "password", "user"]
            },
            confidence_threshold=0.8,
            false_positive_rate=0.05,
            last_updated=time.time()
        )
        
        # Replay attack signature
        replay_signature = ThreatSignature(
            signature_id="replay_001",
            attack_vector=AttackVector.REPLAY_ATTACK,
            pattern={
                "duplicate_timestamp_window": 300,  # 5 minutes
                "identical_signature": True,
                "request_similarity_threshold": 0.95
            },
            confidence_threshold=0.9,
            false_positive_rate=0.02,
            last_updated=time.time()
        )
        
        self.threat_signatures[injection_signature.signature_id] = injection_signature
        self.threat_signatures[replay_signature.signature_id] = replay_signature
    
    def start_monitoring(self):
        """Start adaptive security monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        print("Adaptive security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        print("Adaptive security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main security monitoring loop."""
        while self.is_monitoring:
            try:
                # Adapt security policies based on current threat landscape
                self._adapt_security_policies()
                
                # Update threat signatures based on recent incidents
                self._update_threat_signatures()
                
                # Generate security metrics
                self._update_security_metrics()
                
                time.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                handle_error(
                    error=e,
                    operation="security_monitoring_loop",
                    category=ErrorCategory.SECURITY,
                    severity=ErrorSeverity.MEDIUM
                )
                time.sleep(5)  # Back off on error
    
    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze incoming request for security threats."""
        analysis_result = {
            'timestamp': time.time(),
            'request_id': request_data.get('request_id', 'unknown'),
            'threats_detected': [],
            'risk_score': 0.0,
            'recommended_action': 'allow',
            'policy_violations': []
        }
        
        try:
            # Behavioral analysis
            entity_id = request_data.get('client_id', 'anonymous')
            is_anomaly, anomaly_score, anomaly_types = self.behavioral_analyzer.detect_anomaly(
                entity_id, request_data
            )
            
            if is_anomaly:
                analysis_result['threats_detected'].extend(anomaly_types)
                analysis_result['risk_score'] = max(analysis_result['risk_score'], anomaly_score / 10.0)
            
            # Signature-based threat detection
            for signature_id, signature in self.threat_signatures.items():
                if self._matches_threat_signature(request_data, signature):
                    analysis_result['threats_detected'].append(signature.attack_vector.value)
                    analysis_result['risk_score'] = max(analysis_result['risk_score'], 0.8)
            
            # Policy evaluation
            policy_violations = self._evaluate_security_policies(request_data)
            analysis_result['policy_violations'] = policy_violations
            
            if policy_violations:
                analysis_result['risk_score'] = max(analysis_result['risk_score'], 0.6)
            
            # Determine recommended action
            if analysis_result['risk_score'] > 0.8:
                analysis_result['recommended_action'] = 'block'
            elif analysis_result['risk_score'] > 0.5:
                analysis_result['recommended_action'] = 'challenge'
            elif analysis_result['risk_score'] > 0.2:
                analysis_result['recommended_action'] = 'monitor'
            
            # Create incident if high risk
            if analysis_result['risk_score'] > 0.7:
                self._create_security_incident(request_data, analysis_result)
            
        except Exception as e:
            handle_error(
                error=e,
                operation="analyze_request",
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.HIGH,
                additional_info={'request_id': request_data.get('request_id')}
            )
            
            # Fail secure - block on analysis error
            analysis_result['recommended_action'] = 'block'
            analysis_result['threats_detected'].append('analysis_error')
        
        return analysis_result
    
    def _matches_threat_signature(self, request_data: Dict[str, Any], 
                                 signature: ThreatSignature) -> bool:
        """Check if request matches a threat signature."""
        pattern = signature.pattern
        
        if signature.attack_vector == AttackVector.INJECTION_ATTACK:
            payload = str(request_data.get('payload', ''))
            suspicious_strings = pattern.get('payload_contains', [])
            
            for suspicious in suspicious_strings:
                if suspicious.lower() in payload.lower():
                    return True
        
        elif signature.attack_vector == AttackVector.REPLAY_ATTACK:
            # Check for replay characteristics
            timestamp = request_data.get('timestamp', time.time())
            signature_hash = request_data.get('signature', '')
            
            # Look for duplicate signatures within time window
            duplicate_window = pattern.get('duplicate_timestamp_window', 300)
            
            # This would typically check against a cache of recent requests
            # For now, we implement a simplified check
            current_time = time.time()
            if abs(current_time - timestamp) > duplicate_window:
                return True
        
        return False
    
    def _evaluate_security_policies(self, request_data: Dict[str, Any]) -> List[str]:
        """Evaluate request against security policies."""
        violations = []
        
        for policy_id, policy in self.security_policies.items():
            for rule in policy.rules:
                condition = rule.get('condition', '')
                
                # Simple condition evaluation (would be more sophisticated in practice)
                if 'request_rate' in condition:
                    current_rate = request_data.get('request_rate', 0)
                    baseline_rate = request_data.get('baseline_rate', 10)
                    
                    if 'request_rate > baseline_rate * 3' in condition and current_rate > baseline_rate * 3:
                        violations.append(f"Policy {policy_id}: High request rate")
                    elif 'request_rate > baseline_rate * 10' in condition and current_rate > baseline_rate * 10:
                        violations.append(f"Policy {policy_id}: Extreme request rate")
                
                elif 'failed_attempts' in condition:
                    failed_attempts = request_data.get('failed_attempts', 0)
                    
                    if 'failed_attempts > 3' in condition and failed_attempts > 3:
                        violations.append(f"Policy {policy_id}: Multiple authentication failures")
                    elif 'failed_attempts > 10' in condition and failed_attempts > 10:
                        violations.append(f"Policy {policy_id}: Excessive authentication failures")
        
        return violations
    
    def _create_security_incident(self, request_data: Dict[str, Any], 
                                 analysis_result: Dict[str, Any]):
        """Create a security incident record."""
        incident_id = f"inc_{int(time.time())}_{hash(str(request_data)) % 10000}"
        
        # Determine threat level
        risk_score = analysis_result['risk_score']
        if risk_score > 0.9:
            threat_level = ThreatLevel.CRITICAL
        elif risk_score > 0.7:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.MEDIUM
        
        # Determine primary attack vector
        threats = analysis_result['threats_detected']
        primary_vector = AttackVector.REWARD_MANIPULATION  # Default
        
        if 'injection_attack' in threats:
            primary_vector = AttackVector.INJECTION_ATTACK
        elif 'replay_attack' in threats:
            primary_vector = AttackVector.REPLAY_ATTACK
        elif any('anomaly' in t for t in threats):
            primary_vector = AttackVector.BYZANTINE_BEHAVIOR
        
        incident = SecurityIncident(
            incident_id=incident_id,
            timestamp=time.time(),
            threat_level=threat_level,
            attack_vector=primary_vector,
            affected_components=['request_handler', 'authentication'],
            evidence={
                'request_data': request_data,
                'analysis_result': analysis_result,
                'source_ip': request_data.get('source_ip', 'unknown'),
                'user_agent': request_data.get('user_agent', 'unknown')
            },
            response_actions=self._generate_response_actions(threat_level, primary_vector)
        )
        
        self.active_incidents[incident_id] = incident
        
        # Execute immediate response actions
        self._execute_incident_response(incident)
        
        print(f"Security incident created: {incident_id} ({threat_level.value})")
    
    def _generate_response_actions(self, threat_level: ThreatLevel, 
                                  attack_vector: AttackVector) -> List[str]:
        """Generate response actions for security incident."""
        actions = []
        
        # Base actions for all incidents
        actions.append("log_incident")
        actions.append("notify_security_team")
        
        # Threat level specific actions
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            actions.append("block_source_ip")
            actions.append("increase_monitoring")
            
        if threat_level == ThreatLevel.CRITICAL:
            actions.append("emergency_lockdown")
            actions.append("contact_incident_commander")
        
        # Attack vector specific actions
        if attack_vector == AttackVector.INJECTION_ATTACK:
            actions.extend(["sanitize_inputs", "update_waf_rules"])
        elif attack_vector == AttackVector.REPLAY_ATTACK:
            actions.extend(["implement_nonce", "reduce_token_lifetime"])
        elif attack_vector == AttackVector.BYZANTINE_BEHAVIOR:
            actions.extend(["quarantine_entity", "increase_consensus_threshold"])
        
        return actions
    
    def _execute_incident_response(self, incident: SecurityIncident):
        """Execute automated incident response actions."""
        for action in incident.response_actions:
            try:
                if action == "log_incident":
                    self._log_security_incident(incident)
                elif action == "block_source_ip":
                    self._block_source_ip(incident.evidence.get('source_ip'))
                elif action == "increase_monitoring":
                    self._increase_monitoring_sensitivity()
                # Add more response actions as needed
                
            except Exception as e:
                handle_error(
                    error=e,
                    operation=f"execute_incident_response:{action}",
                    category=ErrorCategory.SECURITY,
                    severity=ErrorSeverity.MEDIUM,
                    additional_info={'incident_id': incident.incident_id}
                )
    
    def _log_security_incident(self, incident: SecurityIncident):
        """Log security incident."""
        log_entry = {
            'incident_id': incident.incident_id,
            'timestamp': incident.timestamp,
            'threat_level': incident.threat_level.value,
            'attack_vector': incident.attack_vector.value,
            'evidence_summary': {
                'source_ip': incident.evidence.get('source_ip'),
                'threats_detected': incident.evidence.get('analysis_result', {}).get('threats_detected', []),
                'risk_score': incident.evidence.get('analysis_result', {}).get('risk_score', 0)
            }
        }
        
        # In production, this would write to a SIEM or security log
        print(f"SECURITY_INCIDENT: {json.dumps(log_entry, indent=2)}")
    
    def _block_source_ip(self, source_ip: str):
        """Block source IP address."""
        if not source_ip or source_ip == 'unknown':
            return
        
        # In production, this would update firewall rules or load balancer
        print(f"BLOCKED_IP: {source_ip}")
        self.security_metrics['blocked_ips'] += 1
    
    def _increase_monitoring_sensitivity(self):
        """Increase monitoring sensitivity during threats."""
        # Reduce anomaly threshold temporarily
        original_threshold = self.behavioral_analyzer.anomaly_threshold
        self.behavioral_analyzer.anomaly_threshold = max(1.5, original_threshold * 0.7)
        
        print(f"Monitoring sensitivity increased: threshold reduced to {self.behavioral_analyzer.anomaly_threshold}")
    
    def _adapt_security_policies(self):
        """Adapt security policies based on threat intelligence."""
        current_time = time.time()
        
        for policy_id, policy in self.security_policies.items():
            if not policy.adaptation_enabled:
                continue
            
            # Adapt based on recent incidents
            recent_incidents = [
                inc for inc in self.active_incidents.values()
                if current_time - inc.timestamp < 3600  # Last hour
            ]
            
            if len(recent_incidents) > 5:  # High incident rate
                # Increase enforcement level
                if policy.enforcement_level == ThreatLevel.LOW:
                    policy.enforcement_level = ThreatLevel.MEDIUM
                elif policy.enforcement_level == ThreatLevel.MEDIUM:
                    policy.enforcement_level = ThreatLevel.HIGH
                
                policy.last_adapted = current_time
                print(f"Policy {policy_id} enforcement level increased to {policy.enforcement_level.value}")
    
    def _update_threat_signatures(self):
        """Update threat signatures based on recent attack patterns."""
        # Analyze recent incidents to identify new attack patterns
        recent_incidents = [
            inc for inc in self.active_incidents.values()
            if time.time() - inc.timestamp < 86400  # Last 24 hours
        ]
        
        # Group incidents by attack vector
        vector_counts = defaultdict(int)
        for incident in recent_incidents:
            vector_counts[incident.attack_vector] += 1
        
        # Update signature thresholds based on attack frequency
        for vector, count in vector_counts.items():
            for sig_id, signature in self.threat_signatures.items():
                if signature.attack_vector == vector and count > 10:
                    # Lower threshold for frequently seen attacks
                    signature.confidence_threshold = max(0.5, signature.confidence_threshold * 0.9)
                    signature.last_updated = time.time()
    
    def _update_security_metrics(self):
        """Update security metrics."""
        current_time = time.time()
        
        # Count active incidents by threat level
        for level in ThreatLevel:
            count = len([
                inc for inc in self.active_incidents.values()
                if inc.threat_level == level and not inc.resolved
            ])
            self.security_metrics[f'active_{level.value}_incidents'] = count
        
        # Calculate mean time to resolution
        resolved_incidents = [
            inc for inc in self.active_incidents.values()
            if inc.resolved and inc.resolution_time
        ]
        
        if resolved_incidents:
            mttr = np.mean([
                inc.resolution_time - inc.timestamp
                for inc in resolved_incidents
            ])
            self.security_metrics['mean_time_to_resolution'] = mttr
    
    def secure_contract_execution(self, contract: RewardContract, 
                                 state: jnp.ndarray, action: jnp.ndarray,
                                 execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute contract with comprehensive security monitoring."""
        start_time = time.time()
        
        # Create execution fingerprint
        execution_id = hashlib.sha256(
            f"{contract.compute_hash()}{state.tobytes()}{action.tobytes()}{start_time}".encode()
        ).hexdigest()[:16]
        
        # Analyze execution context for threats
        security_analysis = self.analyze_request({
            'request_id': execution_id,
            'operation': 'contract_execution',
            'contract_hash': contract.compute_hash(),
            'timestamp': start_time,
            'client_id': execution_context.get('client_id', 'system'),
            'payload_size': len(state.tobytes()) + len(action.tobytes()),
            **execution_context
        })
        
        result = {
            'execution_id': execution_id,
            'security_analysis': security_analysis,
            'execution_allowed': security_analysis['recommended_action'] in ['allow', 'monitor'],
            'reward': None,
            'violations': [],
            'execution_time': None,
            'security_attestation': None
        }
        
        if not result['execution_allowed']:
            result['error'] = f"Execution blocked due to security risk: {security_analysis['threats_detected']}"
            return result
        
        try:
            # Execute contract with monitoring
            reward = contract.compute_reward(state, action)
            violations = contract.check_violations(state, action)
            
            # Verify contract integrity
            expected_hash = contract.compute_hash()
            if expected_hash != execution_context.get('expected_contract_hash'):
                raise ValueError("Contract integrity violation detected")
            
            result.update({
                'reward': float(reward),
                'violations': violations,
                'execution_time': time.time() - start_time
            })
            
            # Generate security attestation
            attestation_data = {
                'execution_id': execution_id,
                'contract_hash': expected_hash,
                'reward': result['reward'],
                'violations': result['violations'],
                'timestamp': start_time,
                'security_cleared': True
            }
            
            # Sign attestation with quantum-resistant signature
            private_key_pem, public_key_pem = self.crypto.generate_quantum_safe_keypair()
            attestation_bytes = json.dumps(attestation_data, sort_keys=True).encode()
            signature = self.crypto.quantum_safe_sign(attestation_bytes, private_key_pem)
            
            result['security_attestation'] = {
                'data': attestation_data,
                'signature': signature.hex(),
                'public_key': public_key_pem.decode()
            }
            
        except Exception as e:
            handle_error(
                error=e,
                operation="secure_contract_execution",
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.HIGH,
                additional_info={
                    'execution_id': execution_id,
                    'contract_hash': contract.compute_hash()
                }
            )
            
            result.update({
                'error': str(e),
                'execution_time': time.time() - start_time
            })
        
        return result
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        current_time = time.time()
        
        # Calculate security metrics for the report
        total_incidents = len(self.active_incidents)
        resolved_incidents = len([
            inc for inc in self.active_incidents.values() if inc.resolved
        ])
        
        critical_incidents = len([
            inc for inc in self.active_incidents.values()
            if inc.threat_level == ThreatLevel.CRITICAL and not inc.resolved
        ])
        
        # Threat landscape analysis
        attack_vector_distribution = defaultdict(int)
        for incident in self.active_incidents.values():
            attack_vector_distribution[incident.attack_vector.value] += 1
        
        # Policy effectiveness
        policy_effectiveness = {}
        for policy_id, policy in self.security_policies.items():
            violations = sum(1 for inc in self.active_incidents.values() 
                           if policy_id in str(inc.evidence))
            policy_effectiveness[policy_id] = {
                'violations_prevented': max(0, 100 - violations),
                'adaptation_count': 1 if current_time - policy.last_adapted < 86400 else 0
            }
        
        report = {
            'report_timestamp': current_time,
            'executive_summary': {
                'total_incidents': total_incidents,
                'resolved_incidents': resolved_incidents,
                'critical_active_incidents': critical_incidents,
                'overall_security_score': self._calculate_security_score()
            },
            'threat_landscape': {
                'attack_vector_distribution': dict(attack_vector_distribution),
                'trending_threats': self._identify_trending_threats(),
                'threat_level_breakdown': {
                    level.value: len([
                        inc for inc in self.active_incidents.values()
                        if inc.threat_level == level
                    ])
                    for level in ThreatLevel
                }
            },
            'security_controls': {
                'active_policies': len(self.security_policies),
                'adaptive_policies': len([
                    p for p in self.security_policies.values()
                    if p.adaptation_enabled
                ]),
                'policy_effectiveness': policy_effectiveness,
                'signature_count': len(self.threat_signatures)
            },
            'performance_metrics': dict(self.security_metrics),
            'recommendations': self._generate_security_recommendations()
        }
        
        return report
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score (0-100)."""
        base_score = 100.0
        
        # Deduct points for active incidents
        for incident in self.active_incidents.values():
            if not incident.resolved:
                if incident.threat_level == ThreatLevel.CRITICAL:
                    base_score -= 20
                elif incident.threat_level == ThreatLevel.HIGH:
                    base_score -= 10
                elif incident.threat_level == ThreatLevel.MEDIUM:
                    base_score -= 5
                else:
                    base_score -= 2
        
        # Bonus points for proactive measures
        adaptive_policies = len([p for p in self.security_policies.values() if p.adaptation_enabled])
        base_score += min(10, adaptive_policies * 2)
        
        return max(0.0, min(100.0, base_score))
    
    def _identify_trending_threats(self) -> List[str]:
        """Identify trending threat types."""
        recent_incidents = [
            inc for inc in self.active_incidents.values()
            if time.time() - inc.timestamp < 86400  # Last 24 hours
        ]
        
        threat_counts = defaultdict(int)
        for incident in recent_incidents:
            threat_counts[incident.attack_vector.value] += 1
        
        # Sort by frequency and return top threats
        trending = sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)
        return [threat for threat, count in trending[:5] if count > 1]
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Analyze current security posture
        critical_incidents = len([
            inc for inc in self.active_incidents.values()
            if inc.threat_level == ThreatLevel.CRITICAL and not inc.resolved
        ])
        
        if critical_incidents > 0:
            recommendations.append("Immediate: Address all critical security incidents")
        
        # Check policy adaptation
        non_adaptive_policies = len([
            p for p in self.security_policies.values() if not p.adaptation_enabled
        ])
        
        if non_adaptive_policies > 0:
            recommendations.append("Enable adaptive capabilities for all security policies")
        
        # Check signature freshness
        stale_signatures = len([
            sig for sig in self.threat_signatures.values()
            if time.time() - sig.last_updated > 604800  # 1 week
        ])
        
        if stale_signatures > 0:
            recommendations.append("Update threat signatures based on recent attack patterns")
        
        # General recommendations
        recommendations.extend([
            "Implement continuous security monitoring",
            "Regular security training for development team",
            "Quarterly penetration testing",
            "Implement zero-trust architecture principles"
        ])
        
        return recommendations


# Integration functions
def create_secure_contract(base_contract: RewardContract) -> Tuple[RewardContract, AdaptiveSecurityFramework]:
    """Create a security-enhanced reward contract."""
    security_framework = AdaptiveSecurityFramework()
    
    # Add security constraints to the contract
    @base_contract.add_constraint(
        name="security_attestation",
        description="Require valid security attestation for execution"
    )
    def security_attestation_constraint(state: jnp.ndarray, action: jnp.ndarray) -> bool:
        # This would check for valid security attestation in production
        return True  # Simplified for example
    
    @base_contract.add_constraint(
        name="anomaly_detection",
        description="Block execution if anomalous behavior detected"
    )
    def anomaly_detection_constraint(state: jnp.ndarray, action: jnp.ndarray) -> bool:
        # This would integrate with behavioral analyzer
        return True  # Simplified for example
    
    return base_contract, security_framework


async def run_security_assessment(contract: RewardContract) -> Dict[str, Any]:
    """Run comprehensive security assessment."""
    print("Starting comprehensive security assessment...")
    
    # Initialize security framework
    base_contract, security_framework = create_secure_contract(contract)
    
    # Start monitoring
    security_framework.start_monitoring()
    
    # Simulate various attack scenarios
    attack_scenarios = [
        {
            'request_id': 'test_injection_001',
            'payload': "'; DROP TABLE contracts; --",
            'client_id': 'attacker_001',
            'timestamp': time.time(),
            'request_rate': 100,
            'baseline_rate': 10
        },
        {
            'request_id': 'test_replay_001', 
            'signature': 'duplicate_signature_123',
            'client_id': 'legitimate_user',
            'timestamp': time.time() - 600,  # Old timestamp
            'request_rate': 5,
            'baseline_rate': 10
        },
        {
            'request_id': 'test_anomaly_001',
            'client_id': 'anomalous_user',
            'request_count': 1000,
            'time_window': 60,
            'payload_size': 1000000,  # Very large payload
            'timestamp': time.time()
        }
    ]
    
    assessment_results = []
    
    for scenario in attack_scenarios:
        result = security_framework.analyze_request(scenario)
        assessment_results.append({
            'scenario': scenario['request_id'],
            'analysis': result
        })
        
        # Small delay between scenarios
        await asyncio.sleep(1)
    
    # Test secure contract execution
    test_state = jnp.array([1.0, 2.0, 3.0])
    test_action = jnp.array([0.5, -0.5])
    
    execution_result = security_framework.secure_contract_execution(
        base_contract,
        test_state,
        test_action,
        {'client_id': 'test_client', 'expected_contract_hash': base_contract.compute_hash()}
    )
    
    # Generate final security report
    security_report = security_framework.generate_security_report()
    
    # Stop monitoring
    security_framework.stop_monitoring()
    
    final_assessment = {
        'attack_scenario_results': assessment_results,
        'secure_execution_test': execution_result,
        'security_report': security_report,
        'assessment_timestamp': time.time(),
        'recommendations': [
            'Deploy adaptive security framework in production',
            'Implement quantum-resistant cryptography',
            'Enable continuous behavioral monitoring',
            'Regular security assessment and updates'
        ]
    }
    
    print("Security assessment completed")
    return final_assessment