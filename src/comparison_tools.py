"""
Comparison Tools for Mammography Analysis

Provides tools for comparing current and previous mammograms,
detecting changes, and tracking lesions over time.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json


class ScanComparator:
    """Compare current and previous mammography scans."""
    
    def __init__(self):
        """Initialize scan comparator."""
        self.history_file = "data/scan_history.json"
    
    def compare_scans(self, current_scan: Dict, previous_scan: Dict) -> Dict:
        """
        Compare current scan with previous scan.
        
        Args:
            current_scan: Current scan data with BIRADS, findings, etc.
            previous_scan: Previous scan data
            
        Returns:
            Comparison analysis
        """
        comparison = {
            'birads_change': self._compare_birads(
                current_scan.get('birads', 0),
                previous_scan.get('birads', 0)
            ),
            'stability_assessment': self._assess_stability(current_scan, previous_scan),
            'new_findings': self._detect_new_findings(current_scan, previous_scan),
            'interval_change': self._calculate_interval_change(current_scan, previous_scan),
            'recommendations': []
        }
        
        # Generate recommendations based on comparison
        comparison['recommendations'] = self._generate_comparison_recommendations(comparison)
        
        return comparison
    
    def _compare_birads(self, current: int, previous: int) -> Dict:
        """Compare BIRADS scores."""
        change = current - previous
        
        if change == 0:
            status = "Stable"
            interpretation = "No change in BIRADS category"
        elif change > 0:
            status = "Increased"
            interpretation = f"BIRADS increased from {previous} to {current}"
            if change >= 2:
                interpretation += " - Significant change, requires attention"
        else:
            status = "Decreased"
            interpretation = f"BIRADS decreased from {previous} to {current}"
            if abs(change) >= 2:
                interpretation += " - Improvement noted"
        
        return {
            'current_birads': current,
            'previous_birads': previous,
            'change': change,
            'status': status,
            'interpretation': interpretation
        }
    
    def _assess_stability(self, current: Dict, previous: Dict) -> Dict:
        """Assess overall stability of findings."""
        birads_stable = current.get('birads', 0) == previous.get('birads', 0)
        
        # Compare probabilities if available
        current_prob = current.get('probabilities', {})
        previous_prob = previous.get('probabilities', {})
        
        prob_changes = {}
        for category in ['BIRADS 1', 'BIRADS 2', 'BIRADS 3', 'BIRADS 4', 'BIRADS 5']:
            curr = current_prob.get(category, 0)
            prev = previous_prob.get(category, 0)
            prob_changes[category] = curr - prev
        
        max_change = max(abs(v) for v in prob_changes.values())
        
        if birads_stable and max_change < 0.1:
            stability = "Stable"
        elif max_change < 0.2:
            stability = "Minimal Change"
        else:
            stability = "Significant Change"
        
        return {
            'stability': stability,
            'birads_stable': birads_stable,
            'probability_changes': prob_changes,
            'max_probability_change': round(max_change, 3)
        }
    
    def _detect_new_findings(self, current: Dict, previous: Dict) -> List[Dict]:
        """Detect new findings compared to previous scan."""
        new_findings = []
        
        current_birads = current.get('birads', 0)
        previous_birads = previous.get('birads', 0)
        
        if current_birads > previous_birads:
            new_findings.append({
                'type': 'BIRADS Upgrade',
                'description': f'BIRADS category increased from {previous_birads} to {current_birads}',
                'significance': 'high' if current_birads >= 4 else 'moderate'
            })
        
        # Check for new suspicious findings
        if current_birads >= 4 and previous_birads < 4:
            new_findings.append({
                'type': 'New Suspicious Finding',
                'description': 'New suspicious finding detected requiring biopsy',
                'significance': 'high'
            })
        
        return new_findings
    
    def _calculate_interval_change(self, current: Dict, previous: Dict) -> Dict:
        """Calculate interval change metrics."""
        current_date = datetime.fromisoformat(current.get('date', datetime.now().isoformat()))
        previous_date = datetime.fromisoformat(previous.get('date', datetime.now().isoformat()))
        
        interval_days = (current_date - previous_date).days
        interval_months = interval_days / 30.44
        
        return {
            'interval_days': interval_days,
            'interval_months': round(interval_months, 1),
            'current_date': current_date.strftime('%Y-%m-%d'),
            'previous_date': previous_date.strftime('%Y-%m-%d')
        }
    
    def _generate_comparison_recommendations(self, comparison: Dict) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []
        
        birads_change = comparison['birads_change']
        stability = comparison['stability_assessment']
        
        if birads_change['change'] >= 2:
            recommendations.append("Significant BIRADS increase - urgent follow-up recommended")
            recommendations.append("Consider immediate biopsy if BIRADS 4 or 5")
        elif birads_change['change'] > 0:
            recommendations.append("BIRADS increase noted - close monitoring recommended")
            recommendations.append("Consider short-term follow-up in 3-6 months")
        elif stability['stability'] == "Stable":
            recommendations.append("Stable findings - continue routine screening")
        else:
            recommendations.append("Some changes detected - monitor closely")
        
        if comparison['new_findings']:
            recommendations.append(f"{len(comparison['new_findings'])} new finding(s) detected")
        
        return recommendations
    
    def track_lesion(self, lesion_id: str, current_scan: Dict) -> Dict:
        """
        Track a specific lesion across multiple scans.
        
        Args:
            lesion_id: Unique lesion identifier
            current_scan: Current scan data
            
        Returns:
            Lesion tracking information
        """
        history = self.load_history()
        
        if lesion_id not in history:
            history[lesion_id] = []
        
        history[lesion_id].append({
            'date': current_scan.get('date', datetime.now().isoformat()),
            'birads': current_scan.get('birads', 0),
            'location': current_scan.get('location', 'Unknown'),
            'size': current_scan.get('size', 0),
            'probabilities': current_scan.get('probabilities', {})
        })
        
        history[lesion_id].sort(key=lambda x: x['date'])
        self.save_history(history)
        
        return self.get_lesion_history(lesion_id)
    
    def get_lesion_history(self, lesion_id: str) -> Dict:
        """Get complete history for a lesion."""
        history = self.load_history()
        
        if lesion_id not in history:
            return {'error': 'Lesion not found'}
        
        lesion_history = history[lesion_id]
        
        if len(lesion_history) < 2:
            return {
                'lesion_id': lesion_id,
                'total_scans': len(lesion_history),
                'history': lesion_history,
                'trend': 'Insufficient data for trend analysis'
            }
        
        # Calculate trend
        birads_trend = [h['birads'] for h in lesion_history]
        size_trend = [h.get('size', 0) for h in lesion_history]
        
        birads_change = birads_trend[-1] - birads_trend[0]
        size_change = size_trend[-1] - size_trend[0] if all(s > 0 for s in size_trend) else 0
        
        return {
            'lesion_id': lesion_id,
            'total_scans': len(lesion_history),
            'first_scan': lesion_history[0]['date'],
            'last_scan': lesion_history[-1]['date'],
            'birads_trend': 'increasing' if birads_change > 0 else 'decreasing' if birads_change < 0 else 'stable',
            'size_trend': 'increasing' if size_change > 0 else 'decreasing' if size_change < 0 else 'stable',
            'history': lesion_history
        }
    
    def load_history(self) -> Dict:
        """Load scan history."""
        if Path(self.history_file).exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_history(self, history: Dict):
        """Save scan history."""
        Path(self.history_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)

