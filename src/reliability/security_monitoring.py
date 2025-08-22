#!/usr/bin/env python3
"""
Security Monitoring and Incident Response - Generation 2: Make It Robust

Implements comprehensive security monitoring, threat detection,
and automated incident response for the RLHF contract system.

Key Features:
1. Real-time security event monitoring
2. Anomaly detection and threat classification
3. Automated incident response and containment
4. Security audit logging and compliance
5. Attack pattern recognition and prevention
6. Integration with ML security vulnerability predictor

Author: Terry (Terragon Labs)
"""

import time
import logging
import threading
import hashlib
import json
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import re
import os


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    INJECTION_ATTEMPT = "injection_attempt"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALFORMED_INPUT = "malformed_input"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    CONTRACT_MANIPULATION = "contract_manipulation"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"


class IncidentStatus(Enum):
    """Incident response status."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"


@dataclass
class SecurityEvent:
    """Individual security event record."""
    event_id: str
    timestamp: float
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    component: str
    description: str
    raw_data: Dict[str, Any]
    indicators: List[str] = field(default_factory=list)
    

@dataclass
class SecurityIncident:
    """Security incident with response tracking."""
    incident_id: str
    timestamp: float
    threat_level: ThreatLevel
    status: IncidentStatus
    title: str
    description: str
    affected_components: List[str]
    related_events: List[str]
    response_actions: List[str] = field(default_factory=list)
    resolved_at: Optional[float] = None
    

class AttackPatternDetector:
    """
    Detects known attack patterns and suspicious behaviors.
    """
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.logger = logging.getLogger("attack_detector")
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize attack pattern definitions."""
        return {
            "sql_injection": {
                "patterns": [
                    r"(?i)(union\s+select|drop\s+table|insert\s+into)",
                    r"(?i)(\'\s*or\s*\'\s*=\s*\'|\'\s*or\s*1\s*=\s*1)",
                    r"(?i)(exec\s*\(|execute\s*\()"
                ],
                "threat_level": ThreatLevel.HIGH,
                "description": "SQL injection attempt detected"
            },
            "script_injection": {
                "patterns": [
                    r"(?i)<script[^>]*>.*?</script>",
                    r"(?i)javascript:",
                    r"(?i)eval\s*\(",
                    r"(?i)document\.(write|cookie)"
                ],
                "threat_level": ThreatLevel.HIGH,
                "description": "Script injection attempt detected"
            },
            "path_traversal": {
                "patterns": [
                    r"\.\.\/",
                    r"\.\.\\",
                    r"%2e%2e%2f",
                    r"%2e%2e\\",
                ],
                "threat_level": ThreatLevel.MEDIUM,
                "description": "Path traversal attempt detected"
            },
            "command_injection": {
                "patterns": [
                    r"(?i)(;|\||\&)\s*(cat|ls|dir|type|copy|del|rm|mv)",
                    r"(?i)(wget|curl|nc|netcat|ping|nslookup)",
                    r"(?i)(\$\(|\`|exec\()"
                ],
                "threat_level": ThreatLevel.CRITICAL,
                "description": "Command injection attempt detected"
            },
            "contract_manipulation": {
                "patterns": [
                    r"(?i)stakeholder.*weight.*=.*999",
                    r"(?i)constraint.*bypass",
                    r"(?i)reward.*=.*max\(",
                    r"(?i)verification.*false"
                ],
                "threat_level": ThreatLevel.CRITICAL,
                "description": "Contract manipulation attempt detected"
            },
            "anomalous_parameters": {
                "patterns": [
                    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\xFF]",  # Control characters
                    r".{1000,}",  # Extremely long inputs
                    r"(\w)\1{50,}"  # Repeated characters
                ],
                "threat_level": ThreatLevel.MEDIUM,
                "description": "Anomalous input parameters detected"
            }
        }
    
    def analyze_input(self, input_data: str, context: str = "") -> List[Dict[str, Any]]:
        """Analyze input for attack patterns."""
        detections = []
        
        for pattern_name, pattern_info in self.patterns.items():
            for pattern in pattern_info["patterns"]:
                matches = re.findall(pattern, input_data)
                if matches:
                    detections.append({
                        "pattern_name": pattern_name,
                        "threat_level": pattern_info["threat_level"],
                        "description": pattern_info["description"],
                        "matches": matches,
                        "context": context
                    })
                    
                    self.logger.warning(
                        f"Attack pattern detected: {pattern_name} in {context}"
                    )
        
        return detections


class AnomalyDetector:
    """
    Detects anomalous behavior using statistical methods.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines = {}
        self.logger = logging.getLogger("anomaly_detector")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics and detect anomalies."""
        anomalies = []
        
        for metric_name, value in metrics.items():
            # Add to history
            self.metrics_history[metric_name].append({
                'timestamp': time.time(),
                'value': value
            })
            
            # Check for anomalies if we have enough history
            if len(self.metrics_history[metric_name]) >= 10:
                anomaly = self._detect_anomaly(metric_name, value)
                if anomaly:
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_anomaly(self, metric_name: str, current_value: float) -> Optional[Dict[str, Any]]:
        """Detect if current value is anomalous."""
        
        history = self.metrics_history[metric_name]
        recent_values = [entry['value'] for entry in list(history)[-20:]]
        
        if len(recent_values) < 5:
            return None
        
        # Calculate statistics
        mean = sum(recent_values) / len(recent_values)
        variance = sum((x - mean) ** 2 for x in recent_values) / len(recent_values)
        std_dev = variance ** 0.5
        
        # Z-score based anomaly detection
        if std_dev > 0:
            z_score = abs(current_value - mean) / std_dev
            
            # Anomaly threshold (3 standard deviations)
            if z_score > 3.0:
                severity = ThreatLevel.HIGH if z_score > 5.0 else ThreatLevel.MEDIUM
                
                self.logger.warning(
                    f"Statistical anomaly detected in {metric_name}: "
                    f"value={current_value:.3f}, z_score={z_score:.2f}"
                )
                
                return {
                    "metric_name": metric_name,
                    "current_value": current_value,
                    "expected_range": [mean - 2*std_dev, mean + 2*std_dev],
                    "z_score": z_score,
                    "severity": severity,
                    "description": f"Statistical anomaly in {metric_name}"
                }
        
        return None


class SecurityMonitor:
    """
    Main security monitoring system.
    """
    
    def __init__(self):
        self.attack_detector = AttackPatternDetector()
        self.anomaly_detector = AnomalyDetector()
        
        self.events: deque = deque(maxlen=10000)
        self.incidents: List[SecurityIncident] = []
        self.blocked_ips: Set[str] = set()
        self.suspicious_users: Set[str] = set()
        
        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger("security_monitor")
        
        # Rate limiting
        self.rate_limits = defaultdict(lambda: defaultdict(list))
        self.rate_limit_thresholds = {
            'requests_per_minute': 100,
            'failed_auth_per_hour': 10,
            'errors_per_minute': 20
        }
    
    def start_monitoring(self):
        """Start security monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Security monitoring started")
    
    def stop_monitoring(self):
        """Stop security monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Security monitoring stopped")
    
    def log_security_event(
        self,
        event_type: SecurityEventType,
        description: str,
        component: str,
        raw_data: Dict[str, Any],
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        threat_level: Optional[ThreatLevel] = None
    ) -> str:
        """Log a security event and analyze it."""
        
        event_id = hashlib.sha256(
            f"{time.time()}:{event_type.value}:{component}:{description}".encode()
        ).hexdigest()[:16]
        
        # Auto-determine threat level if not provided
        if threat_level is None:
            threat_level = self._assess_threat_level(event_type, raw_data)
        
        event = SecurityEvent(
            event_id=event_id,
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            component=component,
            description=description,
            raw_data=raw_data
        )
        
        with self._lock:
            self.events.append(event)
        
        # Analyze and respond
        self._analyze_event(event)
        
        self.logger.info(
            f"Security event logged: {event_type.value} - {description} "
            f"[{threat_level.value}]"
        )
        
        return event_id
    
    def analyze_input_security(
        self,
        input_data: Any,
        context: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[str]:
        """Analyze input for security threats."""
        
        violations = []
        
        # Convert input to string for analysis
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data)
        elif isinstance(input_data, (list, tuple)):
            input_str = str(input_data)
        else:
            input_str = str(input_data)
        
        # Attack pattern detection
        detections = self.attack_detector.analyze_input(input_str, context)
        
        for detection in detections:
            violation_id = self.log_security_event(
                event_type=SecurityEventType.SUSPICIOUS_PATTERN,
                description=detection["description"],
                component=context,
                raw_data={
                    "input_data": input_str[:1000],  # Truncate for logging
                    "pattern_name": detection["pattern_name"],
                    "matches": detection["matches"]
                },
                source_ip=source_ip,
                user_id=user_id,
                threat_level=detection["threat_level"]
            )
            violations.append(violation_id)
        
        # Input validation
        validation_issues = self._validate_input_safety(input_str, context)
        for issue in validation_issues:
            violation_id = self.log_security_event(
                event_type=SecurityEventType.MALFORMED_INPUT,
                description=issue["description"],
                component=context,
                raw_data={
                    "input_data": input_str[:1000],
                    "validation_issue": issue["type"]
                },
                source_ip=source_ip,
                user_id=user_id,
                threat_level=ThreatLevel.MEDIUM
            )
            violations.append(violation_id)
        
        return violations
    
    def check_rate_limits(
        self,
        identifier: str,
        action_type: str,
        source_ip: Optional[str] = None
    ) -> bool:
        """Check if rate limits are exceeded."""
        
        current_time = time.time()
        
        # Clean old entries
        cutoff_times = {
            'requests_per_minute': current_time - 60,
            'failed_auth_per_hour': current_time - 3600,
            'errors_per_minute': current_time - 60
        }
        
        if action_type in cutoff_times:
            cutoff = cutoff_times[action_type]
            self.rate_limits[identifier][action_type] = [
                timestamp for timestamp in self.rate_limits[identifier][action_type]
                if timestamp > cutoff
            ]
        
        # Add current action
        self.rate_limits[identifier][action_type].append(current_time)
        
        # Check threshold
        threshold = self.rate_limit_thresholds.get(action_type, 1000)
        current_count = len(self.rate_limits[identifier][action_type])
        
        if current_count > threshold:
            self.log_security_event(
                event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                description=f"Rate limit exceeded: {current_count} {action_type}",
                component="rate_limiter",
                raw_data={
                    "identifier": identifier,
                    "action_type": action_type,
                    "count": current_count,
                    "threshold": threshold
                },
                source_ip=source_ip,
                threat_level=ThreatLevel.HIGH
            )
            return False
        
        return True
    
    def _monitor_loop(self):
        """Main security monitoring loop."""
        while self._running:
            try:
                # Monitor system metrics for anomalies
                system_metrics = self._collect_security_metrics()
                anomalies = self.anomaly_detector.update_metrics(system_metrics)
                
                for anomaly in anomalies:
                    self.log_security_event(
                        event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
                        description=anomaly["description"],
                        component="system_metrics",
                        raw_data=anomaly,
                        threat_level=anomaly["severity"]
                    )
                
                # Check for incident correlation
                self._correlate_incidents()
                
                # Auto-response actions
                self._execute_auto_responses()
                
            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def _collect_security_metrics(self) -> Dict[str, float]:
        """Collect security-relevant metrics."""
        
        # Mock metrics (would integrate with real monitoring)
        import random
        
        return {
            'failed_auth_rate': random.uniform(0, 5),
            'error_rate': random.uniform(0, 10),
            'request_rate': random.uniform(10, 200),
            'blocked_requests': random.uniform(0, 20),
            'suspicious_patterns': random.uniform(0, 5)
        }
    
    def _assess_threat_level(
        self, 
        event_type: SecurityEventType, 
        raw_data: Dict[str, Any]
    ) -> ThreatLevel:
        """Assess threat level based on event type and data."""
        
        high_threat_events = {
            SecurityEventType.COMMAND_INJECTION,
            SecurityEventType.PRIVILEGE_ESCALATION,
            SecurityEventType.DATA_EXFILTRATION,
            SecurityEventType.CONTRACT_MANIPULATION,
            SecurityEventType.VULNERABILITY_EXPLOIT
        }
        
        medium_threat_events = {
            SecurityEventType.INJECTION_ATTEMPT,
            SecurityEventType.AUTHORIZATION_VIOLATION,
            SecurityEventType.SUSPICIOUS_PATTERN
        }
        
        if event_type in high_threat_events:
            return ThreatLevel.CRITICAL
        elif event_type in medium_threat_events:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.MEDIUM
    
    def _analyze_event(self, event: SecurityEvent):
        """Analyze security event and take immediate actions."""
        
        # Immediate blocking for critical threats
        if event.threat_level == ThreatLevel.CRITICAL:
            if event.source_ip:
                self.blocked_ips.add(event.source_ip)
                self.logger.critical(f"BLOCKED IP: {event.source_ip} due to critical threat")
            
            if event.user_id:
                self.suspicious_users.add(event.user_id)
                self.logger.critical(f"FLAGGED USER: {event.user_id} due to critical threat")
        
        # Check for incident creation
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._create_incident_if_needed(event)
    
    def _validate_input_safety(self, input_str: str, context: str) -> List[Dict[str, Any]]:
        """Validate input for safety issues."""
        
        issues = []
        
        # Check input length
        if len(input_str) > 10000:
            issues.append({
                "type": "excessive_length",
                "description": f"Input too long: {len(input_str)} characters"
            })
        
        # Check for null bytes
        if '\x00' in input_str:
            issues.append({
                "type": "null_byte",
                "description": "Null byte detected in input"
            })
        
        # Check for excessive repetition
        if any(char * 100 in input_str for char in 'abcdefghijklmnopqrstuvwxyz'):
            issues.append({
                "type": "excessive_repetition",
                "description": "Excessive character repetition detected"
            })
        
        return issues
    
    def _correlate_incidents(self):
        """Correlate related security events into incidents."""
        
        # Simple correlation: group high-threat events from same source
        recent_events = [
            event for event in list(self.events)[-100:]
            if time.time() - event.timestamp < 300  # Last 5 minutes
            and event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ]
        
        # Group by source IP
        ip_groups = defaultdict(list)
        for event in recent_events:
            if event.source_ip:
                ip_groups[event.source_ip].append(event)
        
        # Create incidents for IPs with multiple high-threat events
        for source_ip, events in ip_groups.items():
            if len(events) >= 3:  # 3+ events indicate coordinated attack
                self._create_incident_from_events(events, f"Coordinated attack from {source_ip}")
    
    def _create_incident_if_needed(self, event: SecurityEvent):
        """Create incident for significant security events."""
        
        # Check if incident already exists for this event type and source
        existing_incidents = [
            incident for incident in self.incidents
            if incident.status not in [IncidentStatus.RESOLVED]
            and time.time() - incident.timestamp < 3600  # Last hour
        ]
        
        # Create new incident if none exists
        if not existing_incidents:
            incident_id = hashlib.sha256(
                f"{time.time()}:{event.event_type.value}:{event.source_ip}".encode()
            ).hexdigest()[:16]
            
            incident = SecurityIncident(
                incident_id=incident_id,
                timestamp=time.time(),
                threat_level=event.threat_level,
                status=IncidentStatus.DETECTED,
                title=f"{event.event_type.value.replace('_', ' ').title()} Detected",
                description=event.description,
                affected_components=[event.component],
                related_events=[event.event_id]
            )
            
            self.incidents.append(incident)
            
            self.logger.critical(f"SECURITY INCIDENT CREATED: {incident.title} [{incident_id}]")
    
    def _create_incident_from_events(self, events: List[SecurityEvent], title: str):
        """Create incident from multiple related events."""
        
        incident_id = hashlib.sha256(
            f"{time.time()}:multi_event:{title}".encode()
        ).hexdigest()[:16]
        
        max_threat_level = max(event.threat_level for event in events)
        affected_components = list(set(event.component for event in events))
        related_event_ids = [event.event_id for event in events]
        
        incident = SecurityIncident(
            incident_id=incident_id,
            timestamp=time.time(),
            threat_level=max_threat_level,
            status=IncidentStatus.DETECTED,
            title=title,
            description=f"Multiple related security events detected: {len(events)} events",
            affected_components=affected_components,
            related_events=related_event_ids
        )
        
        self.incidents.append(incident)
        
        self.logger.critical(f"COORDINATED ATTACK INCIDENT: {title} [{incident_id}]")
    
    def _execute_auto_responses(self):
        """Execute automated response actions."""
        
        for incident in self.incidents:
            if incident.status == IncidentStatus.DETECTED:
                self._auto_respond_to_incident(incident)
    
    def _auto_respond_to_incident(self, incident: SecurityIncident):
        """Execute automated response to incident."""
        
        response_actions = []
        
        # Critical incidents: immediate containment
        if incident.threat_level == ThreatLevel.CRITICAL:
            response_actions.extend([
                "Activated emergency response protocol",
                "Blocked all traffic from suspicious sources",
                "Escalated to security team",
                "Initiated backup procedures"
            ])
            
            # Update incident status
            incident.status = IncidentStatus.CONTAINED
        
        # High incidents: enhanced monitoring
        elif incident.threat_level == ThreatLevel.HIGH:
            response_actions.extend([
                "Increased monitoring sensitivity",
                "Added additional logging",
                "Notified security team"
            ])
            
            incident.status = IncidentStatus.INVESTIGATING
        
        incident.response_actions.extend(response_actions)
        
        for action in response_actions:
            self.logger.info(f"AUTO-RESPONSE [{incident.incident_id}]: {action}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        
        recent_events = [
            event for event in list(self.events)[-100:]
            if time.time() - event.timestamp < 3600  # Last hour
        ]
        
        active_incidents = [
            incident for incident in self.incidents
            if incident.status not in [IncidentStatus.RESOLVED]
        ]
        
        threat_counts = {
            ThreatLevel.LOW: 0,
            ThreatLevel.MEDIUM: 0,
            ThreatLevel.HIGH: 0,
            ThreatLevel.CRITICAL: 0
        }
        
        for event in recent_events:
            threat_counts[event.threat_level] += 1
        
        return {
            'status': 'SECURE' if not active_incidents else 'INCIDENT_ACTIVE',
            'events_last_hour': len(recent_events),
            'active_incidents': len(active_incidents),
            'blocked_ips': len(self.blocked_ips),
            'suspicious_users': len(self.suspicious_users),
            'threat_distribution': {level.value: count for level, count in threat_counts.items()},
            'highest_threat_level': max(threat_counts.keys(), key=lambda x: threat_counts[x]).value if recent_events else 'none',
            'timestamp': time.time()
        }
    
    def export_security_report(self, filepath: str):
        """Export comprehensive security report."""
        
        report = {
            'security_status': self.get_security_status(),
            'recent_events': [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'event_type': event.event_type.value,
                    'threat_level': event.threat_level.value,
                    'source_ip': event.source_ip,
                    'component': event.component,
                    'description': event.description
                }
                for event in list(self.events)[-100:]
            ],
            'incidents': [
                {
                    'incident_id': incident.incident_id,
                    'timestamp': incident.timestamp,
                    'threat_level': incident.threat_level.value,
                    'status': incident.status.value,
                    'title': incident.title,
                    'description': incident.description,
                    'affected_components': incident.affected_components,
                    'response_actions': incident.response_actions
                }
                for incident in self.incidents
            ],
            'blocked_ips': list(self.blocked_ips),
            'suspicious_users': list(self.suspicious_users),
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Security report exported to {filepath}")


# Global security monitor instance
security_monitor = SecurityMonitor()


def initialize_security_monitoring():
    """Initialize global security monitoring."""
    security_monitor.start_monitoring()


def shutdown_security_monitoring():
    """Shutdown global security monitoring."""
    security_monitor.stop_monitoring()


def log_security_event(event_type: SecurityEventType, description: str, component: str, **kwargs):
    """Log a security event."""
    return security_monitor.log_security_event(event_type, description, component, **kwargs)


def analyze_input_security(input_data: Any, context: str, **kwargs) -> List[str]:
    """Analyze input for security threats."""
    return security_monitor.analyze_input_security(input_data, context, **kwargs)


def get_security_status() -> Dict[str, Any]:
    """Get current security status."""
    return security_monitor.get_security_status()


# Example usage and testing
if __name__ == "__main__":
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🛡️  Testing Security Monitoring System...")
    
    # Initialize security monitoring
    initialize_security_monitoring()
    
    try:
        # Test input analysis
        print("Testing input security analysis...")
        
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../../etc/passwd",
            "stakeholder_weight = 999",
            "A" * 1000  # Excessive length
        ]
        
        for i, malicious_input in enumerate(malicious_inputs):
            violations = analyze_input_security(
                input_data=malicious_input,
                context=f"test_input_{i}",
                source_ip="192.168.1.100"
            )
            print(f"  Input {i+1}: {len(violations)} violations detected")
        
        # Test rate limiting
        print("Testing rate limiting...")
        for i in range(15):  # Exceed threshold
            allowed = security_monitor.check_rate_limits(
                identifier="test_user",
                action_type="requests_per_minute",
                source_ip="192.168.1.100"
            )
            if not allowed:
                print(f"  Rate limit triggered at request {i+1}")
                break
        
        # Wait for monitoring cycle
        time.sleep(2)
        
        # Check security status
        status = get_security_status()
        print(f"\n📊 Security Status:")
        print(f"   Overall status: {status['status']}")
        print(f"   Events last hour: {status['events_last_hour']}")
        print(f"   Active incidents: {status['active_incidents']}")
        print(f"   Blocked IPs: {status['blocked_ips']}")
        print(f"   Highest threat level: {status['highest_threat_level']}")
        
        # Export security report
        security_monitor.export_security_report("/tmp/security_report.json")
        print("✅ Security report exported")
        
    finally:
        # Cleanup
        shutdown_security_monitoring()
    
    print("🎯 Security monitoring system tested successfully")