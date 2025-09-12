"""
Real-Time Learning System for Yelp Recommendations
=================================================

This module implements a real-time learning system that continuously
improves recommendations based on user interactions and feedback.
"""

import asyncio
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging
from pathlib import Path

from modern_yelp_recommender import ModernYelpRecommender, Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InteractionEvent:
    """Represents a user interaction event"""
    user_id: str
    business_id: str
    event_type: str  # view, click, like, dislike, rating, review
    value: Any
    timestamp: datetime
    context: Dict[str, Any]
    session_id: Optional[str] = None

@dataclass
class LearningBatch:
    """Batch of interactions for learning"""
    interactions: List[InteractionEvent]
    batch_id: str
    created_at: datetime
    processed: bool = False

class OnlineLearningModel(nn.Module):
    """
    Neural network model for online learning
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Model architecture
        self.embedding_layer = nn.Embedding(10000, 64)  # For categorical features
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass"""
        return self.linear_layers(x)

class RealTimeLearningSystem:
    """
    Real-time learning system for continuous model improvement
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.recommender = ModernYelpRecommender(config)
        
        # Learning parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.update_frequency = 100  # Update model every N interactions
        self.max_interaction_history = 10000
        
        # Data structures
        self.interaction_queue: deque = deque(maxlen=self.max_interaction_history)
        self.user_embeddings: Dict[str, np.ndarray] = {}
        self.business_embeddings: Dict[str, np.ndarray] = {}
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Models
        self.click_prediction_model = OnlineLearningModel(input_dim=256)
        self.rating_prediction_model = OnlineLearningModel(input_dim=256)
        self.preference_model = OnlineLearningModel(input_dim=256, output_dim=10)
        
        # Optimizers
        self.click_optimizer = torch.optim.Adam(
            self.click_prediction_model.parameters(), 
            lr=self.learning_rate
        )
        self.rating_optimizer = torch.optim.Adam(
            self.rating_prediction_model.parameters(), 
            lr=self.learning_rate
        )
        self.preference_optimizer = torch.optim.Adam(
            self.preference_model.parameters(), 
            lr=self.learning_rate
        )
        
        # Learning statistics
        self.total_interactions = 0
        self.model_updates = 0
        self.last_update_time = datetime.now()
        
        # Performance tracking
        self.performance_metrics = {
            "click_prediction_accuracy": [],
            "rating_prediction_rmse": [],
            "preference_prediction_accuracy": []
        }
        
        logger.info("🚀 Real-time learning system initialized")
    
    async def start_learning_loop(self):
        """
        Start the continuous learning loop
        """
        logger.info("🔄 Starting real-time learning loop...")
        
        while True:
            try:
                await self._learning_cycle()
                await asyncio.sleep(10)  # Run every 10 seconds
            except Exception as e:
                logger.error(f"❌ Learning cycle error: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _learning_cycle(self):
        """
        Single cycle of the learning process
        """
        # Process new interactions
        await self._process_interactions()
        
        # Update models if needed
        if len(self.interaction_queue) >= self.update_frequency:
            await self._update_models()
        
        # Update embeddings
        await self._update_embeddings()
        
        # Evaluate model performance
        await self._evaluate_models()
        
        # Clean up old data
        await self._cleanup_old_data()
    
    async def record_interaction(self, interaction: InteractionEvent):
        """
        Record a new user interaction
        """
        self.interaction_queue.append(interaction)
        self.total_interactions += 1
        
        # Update user preferences in real-time
        await self._update_user_preferences_realtime(interaction)
        
        logger.debug(f"📝 Recorded interaction: {interaction.event_type} for user {interaction.user_id}")
    
    async def _process_interactions(self):
        """
        Process interactions for learning
        """
        if len(self.interaction_queue) < self.batch_size:
            return
        
        # Get recent interactions
        recent_interactions = list(self.interaction_queue)[-self.batch_size:]
        
        # Create learning batches
        batches = await self._create_learning_batches(recent_interactions)
        
        # Process each batch
        for batch in batches:
            await self._process_batch(batch)
    
    async def _create_learning_batches(self, interactions: List[InteractionEvent]) -> List[LearningBatch]:
        """
        Create learning batches from interactions
        """
        batches = []
        
        # Group interactions by type
        click_interactions = [i for i in interactions if i.event_type == "click"]
        rating_interactions = [i for i in interactions if i.event_type == "rating"]
        preference_interactions = [i for i in interactions if i.event_type in ["like", "dislike"]]
        
        # Create batches for different learning tasks
        if click_interactions:
            batches.append(LearningBatch(
                interactions=click_interactions,
                batch_id=f"click_{datetime.now().timestamp()}",
                created_at=datetime.now()
            ))
        
        if rating_interactions:
            batches.append(LearningBatch(
                interactions=rating_interactions,
                batch_id=f"rating_{datetime.now().timestamp()}",
                created_at=datetime.now()
            ))
        
        if preference_interactions:
            batches.append(LearningBatch(
                interactions=preference_interactions,
                batch_id=f"preference_{datetime.now().timestamp()}",
                created_at=datetime.now()
            ))
        
        return batches
    
    async def _process_batch(self, batch: LearningBatch):
        """
        Process a learning batch
        """
        if batch.processed:
            return
        
        try:
            if "click" in batch.batch_id:
                await self._train_click_prediction(batch)
            elif "rating" in batch.batch_id:
                await self._train_rating_prediction(batch)
            elif "preference" in batch.batch_id:
                await self._train_preference_prediction(batch)
            
            batch.processed = True
            logger.info(f"✅ Processed batch: {batch.batch_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing batch {batch.batch_id}: {e}")
    
    async def _train_click_prediction(self, batch: LearningBatch):
        """
        Train click prediction model
        """
        # Prepare training data
        features, labels = await self._prepare_click_training_data(batch.interactions)
        
        if len(features) == 0:
            return
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(labels)
        
        # Training step
        self.click_prediction_model.train()
        self.click_optimizer.zero_grad()
        
        predictions = self.click_prediction_model(X)
        loss = nn.BCELoss()(predictions.squeeze(), y)
        
        loss.backward()
        self.click_optimizer.step()
        
        # Update metrics
        accuracy = self._calculate_accuracy(predictions.squeeze(), y)
        self.performance_metrics["click_prediction_accuracy"].append(accuracy)
        
        logger.info(f"🎯 Click prediction model updated. Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
    
    async def _train_rating_prediction(self, batch: LearningBatch):
        """
        Train rating prediction model
        """
        # Prepare training data
        features, labels = await self._prepare_rating_training_data(batch.interactions)
        
        if len(features) == 0:
            return
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(labels)
        
        # Training step
        self.rating_prediction_model.train()
        self.rating_optimizer.zero_grad()
        
        predictions = self.rating_prediction_model(X)
        loss = nn.MSELoss()(predictions.squeeze(), y)
        
        loss.backward()
        self.rating_optimizer.step()
        
        # Update metrics
        rmse = torch.sqrt(loss).item()
        self.performance_metrics["rating_prediction_rmse"].append(rmse)
        
        logger.info(f"⭐ Rating prediction model updated. Loss: {loss.item():.4f}, RMSE: {rmse:.4f}")
    
    async def _train_preference_prediction(self, batch: LearningBatch):
        """
        Train preference prediction model
        """
        # Prepare training data
        features, labels = await self._prepare_preference_training_data(batch.interactions)
        
        if len(features) == 0:
            return
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(labels)
        
        # Training step
        self.preference_model.train()
        self.preference_optimizer.zero_grad()
        
        predictions = self.preference_model(X)
        loss = nn.BCELoss()(predictions, y)
        
        loss.backward()
        self.preference_optimizer.step()
        
        # Update metrics
        accuracy = self._calculate_preference_accuracy(predictions, y)
        self.performance_metrics["preference_prediction_accuracy"].append(accuracy)
        
        logger.info(f"❤️ Preference prediction model updated. Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
    
    async def _prepare_click_training_data(self, interactions: List[InteractionEvent]) -> Tuple[List[List[float]], List[float]]:
        """
        Prepare training data for click prediction
        """
        features = []
        labels = []
        
        for interaction in interactions:
            # Create feature vector
            feature_vector = await self._create_feature_vector(interaction)
            features.append(feature_vector)
            
            # Label: 1 for click, 0 for no click
            labels.append(1.0 if interaction.event_type == "click" else 0.0)
        
        return features, labels
    
    async def _prepare_rating_training_data(self, interactions: List[InteractionEvent]) -> Tuple[List[List[float]], List[float]]:
        """
        Prepare training data for rating prediction
        """
        features = []
        labels = []
        
        for interaction in interactions:
            if interaction.event_type == "rating":
                # Create feature vector
                feature_vector = await self._create_feature_vector(interaction)
                features.append(feature_vector)
                
                # Normalize rating to 0-1
                rating = float(interaction.value)
                labels.append(rating / 5.0)
        
        return features, labels
    
    async def _prepare_preference_training_data(self, interactions: List[InteractionEvent]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Prepare training data for preference prediction
        """
        features = []
        labels = []
        
        for interaction in interactions:
            if interaction.event_type in ["like", "dislike"]:
                # Create feature vector
                feature_vector = await self._create_feature_vector(interaction)
                features.append(feature_vector)
                
                # Create preference vector (one-hot encoding of preferences)
                preference_vector = await self._create_preference_vector(interaction)
                labels.append(preference_vector)
        
        return features, labels
    
    async def _create_feature_vector(self, interaction: InteractionEvent) -> List[float]:
        """
        Create feature vector from interaction
        """
        features = []
        
        # User features
        user_features = await self._get_user_features(interaction.user_id)
        features.extend(user_features)
        
        # Business features
        business_features = await self._get_business_features(interaction.business_id)
        features.extend(business_features)
        
        # Context features
        context_features = await self._get_context_features(interaction.context)
        features.extend(context_features)
        
        # Temporal features
        temporal_features = await self._get_temporal_features(interaction.timestamp)
        features.extend(temporal_features)
        
        # Pad or truncate to fixed size
        target_size = 256
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features
    
    async def _get_user_features(self, user_id: str) -> List[float]:
        """
        Get user features
        """
        # Get user embedding if available
        if user_id in self.user_embeddings:
            return self.user_embeddings[user_id].tolist()
        
        # Create default user features
        return [0.0] * 64
    
    async def _get_business_features(self, business_id: str) -> List[float]:
        """
        Get business features
        """
        # Get business embedding if available
        if business_id in self.business_embeddings:
            return self.business_embeddings[business_id].tolist()
        
        # Create default business features
        return [0.0] * 64
    
    async def _get_context_features(self, context: Dict[str, Any]) -> List[float]:
        """
        Get context features
        """
        features = []
        
        # Location features
        features.append(context.get("latitude", 0.0))
        features.append(context.get("longitude", 0.0))
        
        # Device features
        features.append(1.0 if context.get("device_type") == "mobile" else 0.0)
        
        # Session features
        features.append(context.get("session_duration", 0.0))
        
        # Pad to fixed size
        while len(features) < 64:
            features.append(0.0)
        
        return features[:64]
    
    async def _get_temporal_features(self, timestamp: datetime) -> List[float]:
        """
        Get temporal features
        """
        features = []
        
        # Hour of day (cyclical encoding)
        hour = timestamp.hour
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        # Day of week (cyclical encoding)
        day_of_week = timestamp.weekday()
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        # Month (cyclical encoding)
        month = timestamp.month
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))
        
        # Pad to fixed size
        while len(features) < 64:
            features.append(0.0)
        
        return features[:64]
    
    async def _create_preference_vector(self, interaction: InteractionEvent) -> List[float]:
        """
        Create preference vector from interaction
        """
        # This would create a vector representing user preferences
        # For now, return a simple binary vector
        preferences = [0.0] * 10
        
        if interaction.event_type == "like":
            preferences[0] = 1.0  # General positive preference
        elif interaction.event_type == "dislike":
            preferences[1] = 1.0  # General negative preference
        
        return preferences
    
    async def _update_models(self):
        """
        Update models based on recent interactions
        """
        logger.info("🔄 Updating models...")
        
        # The actual model updates happen in the training methods
        # This method can be used for additional model maintenance
        
        self.model_updates += 1
        self.last_update_time = datetime.now()
        
        # Save model checkpoints
        await self._save_model_checkpoints()
        
        logger.info(f"✅ Models updated. Total updates: {self.model_updates}")
    
    async def _update_embeddings(self):
        """
        Update user and business embeddings
        """
        # Update user embeddings based on recent interactions
        for user_id in set(i.user_id for i in self.interaction_queue):
            user_interactions = [i for i in self.interaction_queue if i.user_id == user_id]
            if user_interactions:
                await self._update_user_embedding(user_id, user_interactions)
        
        # Update business embeddings based on recent interactions
        for business_id in set(i.business_id for i in self.interaction_queue):
            business_interactions = [i for i in self.interaction_queue if i.business_id == business_id]
            if business_interactions:
                await self._update_business_embedding(business_id, business_interactions)
    
    async def _update_user_embedding(self, user_id: str, interactions: List[InteractionEvent]):
        """
        Update user embedding based on interactions
        """
        # Create embedding from interaction patterns
        interaction_features = []
        
        for interaction in interactions:
            # Extract features from interaction
            features = await self._create_feature_vector(interaction)
            interaction_features.append(features)
        
        if interaction_features:
            # Average the features to create user embedding
            user_embedding = np.mean(interaction_features, axis=0)
            self.user_embeddings[user_id] = user_embedding
    
    async def _update_business_embedding(self, business_id: str, interactions: List[InteractionEvent]):
        """
        Update business embedding based on interactions
        """
        # Create embedding from interaction patterns
        interaction_features = []
        
        for interaction in interactions:
            # Extract features from interaction
            features = await self._create_feature_vector(interaction)
            interaction_features.append(features)
        
        if interaction_features:
            # Average the features to create business embedding
            business_embedding = np.mean(interaction_features, axis=0)
            self.business_embeddings[business_id] = business_embedding
    
    async def _update_user_preferences_realtime(self, interaction: InteractionEvent):
        """
        Update user preferences in real-time
        """
        user_id = interaction.user_id
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        # Update preferences based on interaction type
        if interaction.event_type == "like":
            self.user_preferences[user_id]["positive_interactions"] = \
                self.user_preferences[user_id].get("positive_interactions", 0) + 1
        elif interaction.event_type == "dislike":
            self.user_preferences[user_id]["negative_interactions"] = \
                self.user_preferences[user_id].get("negative_interactions", 0) + 1
        elif interaction.event_type == "rating":
            rating = float(interaction.value)
            if rating >= 4.0:
                self.user_preferences[user_id]["high_ratings"] = \
                    self.user_preferences[user_id].get("high_ratings", 0) + 1
            elif rating <= 2.0:
                self.user_preferences[user_id]["low_ratings"] = \
                    self.user_preferences[user_id].get("low_ratings", 0) + 1
    
    async def _evaluate_models(self):
        """
        Evaluate model performance
        """
        # Calculate average performance metrics
        for metric_name, values in self.performance_metrics.items():
            if values:
                avg_value = np.mean(values[-10:])  # Average of last 10 values
                logger.info(f"📊 {metric_name}: {avg_value:.4f}")
    
    async def _cleanup_old_data(self):
        """
        Clean up old data to prevent memory issues
        """
        # Keep only recent performance metrics
        for metric_name in self.performance_metrics:
            if len(self.performance_metrics[metric_name]) > 100:
                self.performance_metrics[metric_name] = \
                    self.performance_metrics[metric_name][-50:]
    
    async def _save_model_checkpoints(self):
        """
        Save model checkpoints
        """
        checkpoint_dir = Path("model_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        torch.save(self.click_prediction_model.state_dict(), 
                  checkpoint_dir / f"click_model_{timestamp}.pth")
        torch.save(self.rating_prediction_model.state_dict(), 
                  checkpoint_dir / f"rating_model_{timestamp}.pth")
        torch.save(self.preference_model.state_dict(), 
                  checkpoint_dir / f"preference_model_{timestamp}.pth")
        
        # Save embeddings
        embeddings_data = {
            "user_embeddings": {k: v.tolist() for k, v in self.user_embeddings.items()},
            "business_embeddings": {k: v.tolist() for k, v in self.business_embeddings.items()},
            "user_preferences": self.user_preferences,
            "timestamp": timestamp
        }
        
        with open(checkpoint_dir / f"embeddings_{timestamp}.json", "w") as f:
            json.dump(embeddings_data, f)
    
    def _calculate_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculate accuracy for binary classification
        """
        predicted_labels = (predictions > 0.5).float()
        correct = (predicted_labels == labels).float().sum()
        return (correct / len(labels)).item()
    
    def _calculate_preference_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculate accuracy for preference prediction
        """
        predicted_labels = (predictions > 0.5).float()
        correct = (predicted_labels == labels).float().sum()
        total = labels.numel()
        return (correct / total).item()
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get current user preferences
        """
        return self.user_preferences.get(user_id, {})
    
    async def predict_click_probability(self, user_id: str, business_id: str, context: Dict[str, Any]) -> float:
        """
        Predict click probability for a user-business pair
        """
        # Create interaction event
        interaction = InteractionEvent(
            user_id=user_id,
            business_id=business_id,
            event_type="prediction",
            value=None,
            timestamp=datetime.now(),
            context=context
        )
        
        # Create feature vector
        feature_vector = await self._create_feature_vector(interaction)
        
        # Make prediction
        self.click_prediction_model.eval()
        with torch.no_grad():
            X = torch.FloatTensor([feature_vector])
            prediction = self.click_prediction_model(X)
            return prediction.item()
    
    async def predict_rating(self, user_id: str, business_id: str, context: Dict[str, Any]) -> float:
        """
        Predict rating for a user-business pair
        """
        # Create interaction event
        interaction = InteractionEvent(
            user_id=user_id,
            business_id=business_id,
            event_type="prediction",
            value=None,
            timestamp=datetime.now(),
            context=context
        )
        
        # Create feature vector
        feature_vector = await self._create_feature_vector(interaction)
        
        # Make prediction
        self.rating_prediction_model.eval()
        with torch.no_grad():
            X = torch.FloatTensor([feature_vector])
            prediction = self.rating_prediction_model(X)
            return prediction.item() * 5.0  # Convert back to 1-5 scale

# Example usage
async def main():
    """
    Example of how to use the real-time learning system
    """
    config = Config()
    learning_system = RealTimeLearningSystem(config)
    
    # Start learning loop
    asyncio.create_task(learning_system.start_learning_loop())
    
    # Simulate some interactions
    interactions = [
        InteractionEvent(
            user_id="user_1",
            business_id="business_1",
            event_type="click",
            value=1,
            timestamp=datetime.now(),
            context={"device_type": "mobile", "session_duration": 300}
        ),
        InteractionEvent(
            user_id="user_1",
            business_id="business_1",
            event_type="rating",
            value=4.5,
            timestamp=datetime.now(),
            context={"device_type": "mobile", "session_duration": 300}
        ),
        InteractionEvent(
            user_id="user_2",
            business_id="business_2",
            event_type="like",
            value=1,
            timestamp=datetime.now(),
            context={"device_type": "desktop", "session_duration": 600}
        )
    ]
    
    # Record interactions
    for interaction in interactions:
        await learning_system.record_interaction(interaction)
    
    print("🔄 Real-time learning system running...")
    print("The system will continuously learn from user interactions!")
    
    # Keep the system running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Stopping real-time learning system...")

if __name__ == "__main__":
    asyncio.run(main())