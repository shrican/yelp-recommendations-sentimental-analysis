"""
Agentic AI Recommendation System for Yelp
=========================================

This module implements an agentic AI system that proactively provides
context-aware recommendations without explicit user requests.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from modern_yelp_recommender import ModernYelpRecommender, Config, RecommendationRequest, RecommendationType

class AgentState(Enum):
    """States of the agentic AI system"""
    LEARNING = "learning"
    MONITORING = "monitoring"
    RECOMMENDING = "recommending"
    ADAPTING = "adapting"
    SLEEPING = "sleeping"

class ContextType(Enum):
    """Types of contextual information"""
    TEMPORAL = "temporal"
    LOCATIONAL = "locational"
    SOCIAL = "social"
    WEATHER = "weather"
    CALENDAR = "calendar"
    BEHAVIORAL = "behavioral"

@dataclass
class ContextualSignal:
    """Represents a contextual signal for recommendations"""
    signal_type: ContextType
    value: Any
    confidence: float
    timestamp: datetime
    source: str

@dataclass
class UserProfile:
    """Enhanced user profile for agentic recommendations"""
    user_id: str
    preferences: Dict[str, Any]
    behavior_patterns: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    current_context: Dict[str, Any]
    last_active: datetime
    recommendation_history: List[Dict[str, Any]]

class AgenticRecommendationAgent:
    """
    Agentic AI system that proactively provides recommendations
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.recommender = ModernYelpRecommender(config)
        self.state = AgentState.LEARNING
        self.user_profiles: Dict[str, UserProfile] = {}
        self.context_signals: List[ContextualSignal] = []
        self.recommendation_queue: List[Dict[str, Any]] = []
        
        # Agent parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.proactive_threshold = 0.8
        
        print("🤖 Agentic Recommendation Agent initialized")
    
    async def start_agent_loop(self):
        """
        Main agent loop for continuous operation
        """
        print("🚀 Starting agentic recommendation loop...")
        
        while True:
            try:
                await self._agent_cycle()
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                print(f"❌ Agent error: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _agent_cycle(self):
        """
        Single cycle of the agent's operation
        """
        # Update state based on current conditions
        await self._update_agent_state()
        
        # Collect contextual signals
        await self._collect_contextual_signals()
        
        # Process signals and update user profiles
        await self._process_contextual_signals()
        
        # Generate proactive recommendations
        await self._generate_proactive_recommendations()
        
        # Deliver recommendations to users
        await self._deliver_recommendations()
        
        # Learn from user interactions
        await self._learn_from_interactions()
    
    async def _update_agent_state(self):
        """
        Update agent state based on current conditions
        """
        current_time = datetime.now()
        hour = current_time.hour
        
        # Determine state based on time and activity
        if 6 <= hour <= 22:  # Active hours
            if len(self.context_signals) > 10:
                self.state = AgentState.RECOMMENDING
            else:
                self.state = AgentState.MONITORING
        else:  # Night hours
            self.state = AgentState.SLEEPING
    
    async def _collect_contextual_signals(self):
        """
        Collect various contextual signals
        """
        signals = []
        
        # Temporal signals
        current_time = datetime.now()
        signals.append(ContextualSignal(
            signal_type=ContextType.TEMPORAL,
            value={
                "hour": current_time.hour,
                "day_of_week": current_time.weekday(),
                "is_weekend": current_time.weekday() >= 5,
                "season": self._get_season(current_time)
            },
            confidence=1.0,
            timestamp=current_time,
            source="system_clock"
        ))
        
        # Weather signals (mock implementation)
        weather_data = await self._get_weather_data()
        if weather_data:
            signals.append(ContextualSignal(
                signal_type=ContextType.WEATHER,
                value=weather_data,
                confidence=0.9,
                timestamp=current_time,
                source="weather_api"
            ))
        
        # Social signals (mock implementation)
        social_trends = await self._get_social_trends()
        if social_trends:
            signals.append(ContextualSignal(
                signal_type=ContextType.SOCIAL,
                value=social_trends,
                confidence=0.8,
                timestamp=current_time,
                source="social_media_api"
            ))
        
        # Update signals list
        self.context_signals = signals[-100:]  # Keep last 100 signals
    
    async def _process_contextual_signals(self):
        """
        Process contextual signals and update user profiles
        """
        for signal in self.context_signals:
            # Update user profiles based on signals
            for user_id, profile in self.user_profiles.items():
                await self._update_user_context(user_id, signal)
    
    async def _generate_proactive_recommendations(self):
        """
        Generate proactive recommendations based on context
        """
        if self.state != AgentState.RECOMMENDING:
            return
        
        for user_id, profile in self.user_profiles.items():
            # Check if user is active
            if not self._is_user_active(profile):
                continue
            
            # Generate contextual recommendations
            recommendations = await self._generate_contextual_recommendations(profile)
            
            if recommendations:
                self.recommendation_queue.append({
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "context": profile.current_context,
                    "timestamp": datetime.now(),
                    "priority": self._calculate_priority(profile, recommendations)
                })
    
    async def _deliver_recommendations(self):
        """
        Deliver recommendations to users through various channels
        """
        # Sort by priority
        self.recommendation_queue.sort(key=lambda x: x["priority"], reverse=True)
        
        for rec in self.recommendation_queue[:10]:  # Process top 10
            await self._deliver_to_user(rec)
        
        # Clear processed recommendations
        self.recommendation_queue = []
    
    async def _learn_from_interactions(self):
        """
        Learn from user interactions to improve recommendations
        """
        # This would analyze user interactions and update models
        # For now, we'll implement a simple learning mechanism
        
        for user_id, profile in self.user_profiles.items():
            if profile.interaction_history:
                # Analyze recent interactions
                recent_interactions = profile.interaction_history[-10:]
                
                # Update preferences based on interactions
                await self._update_user_preferences(user_id, recent_interactions)
    
    async def add_user(self, user_id: str, initial_preferences: Dict[str, Any] = None):
        """
        Add a new user to the agentic system
        """
        profile = UserProfile(
            user_id=user_id,
            preferences=initial_preferences or {},
            behavior_patterns={},
            interaction_history=[],
            current_context={},
            last_active=datetime.now(),
            recommendation_history=[]
        )
        
        self.user_profiles[user_id] = profile
        print(f"👤 Added user {user_id} to agentic system")
    
    async def record_interaction(self, user_id: str, interaction: Dict[str, Any]):
        """
        Record user interaction for learning
        """
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.interaction_history.append({
                **interaction,
                "timestamp": datetime.now().isoformat()
            })
            profile.last_active = datetime.now()
            
            # Keep only recent interactions
            profile.interaction_history = profile.interaction_history[-100:]
    
    async def _generate_contextual_recommendations(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on user context
        """
        # Build recommendation request based on context
        request = RecommendationRequest(
            user_id=profile.user_id,
            preferences=self._build_preferences_from_context(profile),
            recommendation_type=RecommendationType.CONTEXTUAL,
            count=5,
            include_explanations=True
        )
        
        try:
            response = await self.recommender.get_recommendations(request)
            return response.recommendations
        except Exception as e:
            print(f"❌ Error generating recommendations for {profile.user_id}: {e}")
            return []
    
    def _build_preferences_from_context(self, profile: UserProfile) -> Dict[str, Any]:
        """
        Build user preferences from current context
        """
        preferences = profile.preferences.copy()
        
        # Adjust preferences based on current context
        current_time = datetime.now()
        hour = current_time.hour
        
        # Time-based preferences
        if 6 <= hour <= 10:  # Breakfast
            preferences["time_preference"] = "breakfast"
        elif 11 <= hour <= 14:  # Lunch
            preferences["time_preference"] = "lunch"
        elif 17 <= hour <= 21:  # Dinner
            preferences["time_preference"] = "dinner"
        
        # Weather-based preferences
        weather_signal = self._get_latest_signal(ContextType.WEATHER)
        if weather_signal:
            weather = weather_signal.value
            if weather.get("temperature", 20) < 10:  # Cold weather
                preferences["cuisine_types"] = preferences.get("cuisine_types", []) + ["hot_soup", "comfort_food"]
            elif weather.get("condition") == "rainy":
                preferences["ambience_preference"] = preferences.get("ambience_preference", []) + ["indoor", "cozy"]
        
        return preferences
    
    def _calculate_priority(self, profile: UserProfile, recommendations: List[Dict[str, Any]]) -> float:
        """
        Calculate priority score for recommendations
        """
        base_priority = 0.5
        
        # Increase priority for active users
        time_since_active = (datetime.now() - profile.last_active).total_seconds()
        if time_since_active < 3600:  # Active in last hour
            base_priority += 0.3
        
        # Increase priority based on recommendation quality
        if recommendations:
            avg_confidence = np.mean([rec.get("confidence_score", 0.5) for rec in recommendations])
            base_priority += avg_confidence * 0.2
        
        # Increase priority for users with high engagement
        if len(profile.interaction_history) > 10:
            base_priority += 0.1
        
        return min(base_priority, 1.0)
    
    def _is_user_active(self, profile: UserProfile) -> bool:
        """
        Check if user is currently active
        """
        time_since_active = (datetime.now() - profile.last_active).total_seconds()
        return time_since_active < 7200  # Active within last 2 hours
    
    async def _update_user_context(self, user_id: str, signal: ContextualSignal):
        """
        Update user context based on signal
        """
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.current_context[signal.signal_type.value] = signal.value
    
    async def _update_user_preferences(self, user_id: str, interactions: List[Dict[str, Any]]):
        """
        Update user preferences based on interactions
        """
        if user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[user_id]
        
        # Analyze interaction patterns
        positive_interactions = [i for i in interactions if i.get("rating", 0) > 3]
        
        if positive_interactions:
            # Extract patterns from positive interactions
            for interaction in positive_interactions:
                business_id = interaction.get("business_id")
                if business_id:
                    # Update preferences based on liked businesses
                    # This would involve analyzing business features
                    pass
    
    def _get_latest_signal(self, signal_type: ContextType) -> Optional[ContextualSignal]:
        """
        Get the latest signal of a specific type
        """
        for signal in reversed(self.context_signals):
            if signal.signal_type == signal_type:
                return signal
        return None
    
    def _get_season(self, date: datetime) -> str:
        """
        Get season based on date
        """
        month = date.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    async def _get_weather_data(self) -> Optional[Dict[str, Any]]:
        """
        Get current weather data (mock implementation)
        """
        # In a real implementation, this would call a weather API
        return {
            "temperature": 22,
            "condition": "sunny",
            "humidity": 60,
            "wind_speed": 10
        }
    
    async def _deliver_to_user(self, recommendation: Dict[str, Any]):
        """
        Deliver recommendation to user (mock implementation)
        """
        user_id = recommendation["user_id"]
        recs = recommendation["recommendations"]
        
        print(f"📱 Delivering {len(recs)} recommendations to user {user_id}")
        
        # In a real implementation, this would:
        # - Send push notification
        # - Send email
        # - Update in-app feed
        # - Send SMS
        
        # Record the delivery
        await self.record_interaction(user_id, {
            "type": "recommendation_delivered",
            "count": len(recs),
            "context": recommendation["context"]
        })
    
    async def _get_social_trends(self) -> Optional[Dict[str, Any]]:
        """
        Get current social media trends (mock implementation)
        """
        # In a real implementation, this would analyze social media trends
        return {
            "trending_cuisines": ["italian", "japanese", "vegan"],
            "popular_hashtags": ["#foodie", "#restaurant", "#delicious"],
            "viral_restaurants": []
        }

# Agentic System Manager
class AgenticSystemManager:
    """
    Manages multiple agentic recommendation agents
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.agents: Dict[str, AgenticRecommendationAgent] = {}
        self.global_context: Dict[str, Any] = {}
    
    async def create_agent(self, agent_id: str) -> AgenticRecommendationAgent:
        """
        Create a new agentic recommendation agent
        """
        agent = AgenticRecommendationAgent(self.config)
        self.agents[agent_id] = agent
        
        # Start agent loop
        asyncio.create_task(agent.start_agent_loop())
        
        print(f"🤖 Created agentic agent: {agent_id}")
        return agent
    
    async def get_agent(self, agent_id: str) -> Optional[AgenticRecommendationAgent]:
        """
        Get an existing agent
        """
        return self.agents.get(agent_id)
    
    async def add_user_to_agent(self, agent_id: str, user_id: str, preferences: Dict[str, Any] = None):
        """
        Add user to a specific agent
        """
        agent = await self.get_agent(agent_id)
        if agent:
            await agent.add_user(user_id, preferences)
        else:
            print(f"❌ Agent {agent_id} not found")
    
    async def record_interaction(self, agent_id: str, user_id: str, interaction: Dict[str, Any]):
        """
        Record interaction for a specific agent
        """
        agent = await self.get_agent(agent_id)
        if agent:
            await agent.record_interaction(user_id, interaction)
        else:
            print(f"❌ Agent {agent_id} not found")

# Example usage
async def main():
    """
    Example of how to use the agentic recommendation system
    """
    config = Config()
    manager = AgenticSystemManager(config)
    
    # Create an agent
    agent = await manager.create_agent("yelp_agent_1")
    
    # Add some users
    await manager.add_user_to_agent("yelp_agent_1", "user_1", {
        "cuisine_types": ["italian", "japanese"],
        "price_range": [2, 4],
        "location_preference": "downtown"
    })
    
    await manager.add_user_to_agent("yelp_agent_1", "user_2", {
        "cuisine_types": ["mexican", "american"],
        "price_range": [1, 3],
        "location_preference": "suburbs"
    })
    
    # Record some interactions
    await manager.record_interaction("yelp_agent_1", "user_1", {
        "type": "restaurant_view",
        "business_id": "restaurant_123",
        "rating": 4.5,
        "timestamp": datetime.now().isoformat()
    })
    
    print("🤖 Agentic recommendation system running...")
    print("The system will now proactively provide recommendations based on context!")
    
    # Keep the system running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Stopping agentic recommendation system...")

if __name__ == "__main__":
    asyncio.run(main())