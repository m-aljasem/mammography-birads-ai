"""
Risk Assessment Calculator for Mammography BIRADS Classification

Implements risk calculation models including Gail model and
density-based risk assessment.
"""

from typing import Dict, Optional
from datetime import datetime
import math


class BreastCancerRiskCalculator:
    """Calculate breast cancer risk using multiple models."""
    
    def __init__(self):
        """Initialize risk calculator."""
        self.gail_model_coefficients = self._load_gail_coefficients()
    
    def calculate_gail_risk(self, age: int, age_at_menarche: int, 
                           age_at_first_birth: Optional[int], 
                           first_degree_relatives: int,
                           previous_biopsies: int,
                           atypical_hyperplasia: bool = False) -> Dict:
        """
        Calculate 5-year and lifetime risk using Gail model.
        
        Args:
            age: Current age
            age_at_menarche: Age at first period
            age_at_first_birth: Age at first live birth (None if nulliparous)
            first_degree_relatives: Number of first-degree relatives with breast cancer
            previous_biopsies: Number of previous breast biopsies
            atypical_hyperplasia: History of atypical hyperplasia
            
        Returns:
            Risk assessment dictionary
        """
        # Simplified Gail model calculation
        # In production, use validated Gail model coefficients
        
        base_risk = self._calculate_base_risk(age)
        
        # Adjust for risk factors
        risk_multiplier = 1.0
        
        # Age at menarche adjustment
        if age_at_menarche < 12:
            risk_multiplier *= 1.2
        elif age_at_menarche >= 14:
            risk_multiplier *= 0.9
        
        # Age at first birth adjustment
        if age_at_first_birth is None:
            risk_multiplier *= 1.3  # Nulliparous
        elif age_at_first_birth >= 30:
            risk_multiplier *= 1.2
        
        # Family history adjustment
        if first_degree_relatives == 1:
            risk_multiplier *= 1.8
        elif first_degree_relatives >= 2:
            risk_multiplier *= 2.5
        
        # Biopsy history adjustment
        if previous_biopsies >= 1:
            risk_multiplier *= 1.3
        if atypical_hyperplasia:
            risk_multiplier *= 1.5
        
        five_year_risk = base_risk * risk_multiplier * 5
        lifetime_risk = base_risk * risk_multiplier * (80 - age)
        
        # Cap risks at reasonable maximums
        five_year_risk = min(five_year_risk, 20.0)
        lifetime_risk = min(lifetime_risk, 50.0)
        
        return {
            'five_year_risk': round(five_year_risk, 2),
            'lifetime_risk': round(lifetime_risk, 2),
            'risk_level': self._classify_risk(five_year_risk),
            'risk_factors': {
                'age': age,
                'family_history': first_degree_relatives > 0,
                'biopsy_history': previous_biopsies > 0,
                'atypical_hyperplasia': atypical_hyperplasia
            }
        }
    
    def calculate_birads_risk(self, birads_score: int, density: str, 
                             age: int) -> Dict:
        """
        Calculate risk based on BIRADS score and density.
        
        Args:
            birads_score: BIRADS category (1-5)
            density: Breast density category
            age: Patient age
            
        Returns:
            Risk assessment
        """
        # BIRADS-based risk
        birads_risks = {
            1: 0.1,  # Negative
            2: 0.5,  # Benign
            3: 2.0,  # Probably benign
            4: 20.0, # Suspicious
            5: 95.0  # Highly suspicious
        }
        
        base_risk = birads_risks.get(birads_score, 0.1)
        
        # Density adjustment
        density_multipliers = {
            'A': 0.8,  # Almost entirely fatty
            'B': 1.0,  # Scattered fibroglandular
            'C': 1.2,  # Heterogeneously dense
            'D': 1.5   # Extremely dense
        }
        density_mult = density_multipliers.get(density.upper(), 1.0)
        
        # Age adjustment (higher risk with age)
        age_mult = 1.0 + (age - 50) * 0.01 if age >= 50 else 1.0
        
        adjusted_risk = base_risk * density_mult * age_mult
        
        return {
            'birads_risk': round(adjusted_risk, 2),
            'birads_category': birads_score,
            'density_category': density,
            'risk_level': self._classify_birads_risk(birads_score),
            'recommendations': self._get_birads_recommendations(birads_score)
        }
    
    def calculate_comprehensive_risk(self, gail_data: Dict, birads_data: Dict) -> Dict:
        """
        Calculate comprehensive risk combining multiple factors.
        
        Args:
            gail_data: Gail model results
            birads_data: BIRADS risk data
            
        Returns:
            Comprehensive risk assessment
        """
        gail_risk = gail_data.get('five_year_risk', 0)
        birads_risk = birads_data.get('birads_risk', 0)
        
        # Weighted combination
        comprehensive_risk = (gail_risk * 0.4 + birads_risk * 0.6)
        
        return {
            'comprehensive_risk': round(comprehensive_risk, 2),
            'gail_risk': gail_risk,
            'birads_risk': birads_risk,
            'overall_risk_level': self._classify_risk(comprehensive_risk),
            'recommendations': self._get_comprehensive_recommendations(comprehensive_risk)
        }
    
    def _calculate_base_risk(self, age: int) -> float:
        """Calculate base risk for age."""
        # Simplified age-based risk curve
        if age < 40:
            return 0.1
        elif age < 50:
            return 0.3
        elif age < 60:
            return 0.5
        elif age < 70:
            return 0.8
        else:
            return 1.2
    
    def _classify_risk(self, risk: float) -> str:
        """Classify risk level."""
        if risk >= 20:
            return "High"
        elif risk >= 5:
            return "Moderate"
        elif risk >= 1.5:
            return "Above Average"
        else:
            return "Average"
    
    def _classify_birads_risk(self, birads: int) -> str:
        """Classify BIRADS risk."""
        if birads >= 4:
            return "High"
        elif birads == 3:
            return "Moderate"
        else:
            return "Low"
    
    def _get_birads_recommendations(self, birads: int) -> list:
        """Get recommendations based on BIRADS score."""
        recommendations = {
            1: ["Routine annual screening"],
            2: ["Routine annual screening"],
            3: ["Short-term follow-up in 6 months", "Consider additional imaging"],
            4: ["Biopsy recommended", "Urgent follow-up"],
            5: ["Biopsy strongly recommended", "Immediate follow-up"]
        }
        return recommendations.get(birads, ["Consult with radiologist"])
    
    def _get_comprehensive_recommendations(self, risk: float) -> list:
        """Get comprehensive recommendations."""
        recommendations = []
        
        if risk >= 20:
            recommendations.append("High risk - consider genetic counseling")
            recommendations.append("Enhanced screening (MRI) recommended")
            recommendations.append("Consider risk-reducing medications")
        elif risk >= 5:
            recommendations.append("Moderate risk - annual mammography")
            recommendations.append("Consider supplemental screening")
        else:
            recommendations.append("Average risk - routine screening")
            recommendations.append("Continue annual mammography")
        
        return recommendations
    
    def _load_gail_coefficients(self) -> Dict:
        """Load Gail model coefficients (simplified)."""
        # In production, use validated coefficients from research
        return {}


class FollowUpRecommendations:
    """Generate follow-up recommendations based on BIRADS and risk."""
    
    @staticmethod
    def get_screening_interval(birads_score: int, risk_level: str, age: int) -> Dict:
        """
        Determine appropriate screening interval.
        
        Args:
            birads_score: BIRADS category
            risk_level: Overall risk level
            age: Patient age
            
        Returns:
            Screening interval recommendation
        """
        if birads_score >= 4:
            interval_months = 3
            urgency = "urgent"
        elif birads_score == 3:
            interval_months = 6
            urgency = "moderate"
        elif risk_level == "High":
            interval_months = 6
            urgency = "moderate"
        elif age >= 50:
            interval_months = 12
            urgency = "routine"
        else:
            interval_months = 12
            urgency = "routine"
        
        return {
            'interval_months': interval_months,
            'next_screening_date': None,  # Would calculate from current date
            'urgency': urgency,
            'recommendation': f"Next screening recommended in {interval_months} months"
        }

