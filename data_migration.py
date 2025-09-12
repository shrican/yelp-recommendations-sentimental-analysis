"""
Data Migration Script for Modern Yelp Recommendation System
=========================================================

This script migrates data from the legacy system to the modern AI-powered system,
converting it to formats optimized for transformer embeddings and vector databases.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import asyncio
from datetime import datetime
import os
from pathlib import Path

from modern_yelp_recommender import ModernYelpRecommender, Config

class DataMigrator:
    """
    Migrates legacy Yelp data to modern AI-optimized format
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.recommender = ModernYelpRecommender(config)
        
    async def migrate_all_data(self, legacy_data_path: str = "legacy_data/"):
        """
        Migrate all legacy data to modern format
        """
        print("🚀 Starting data migration to modern AI system...")
        
        # Create output directory
        os.makedirs("modern_data", exist_ok=True)
        
        # Migrate each data type
        await self.migrate_business_data(legacy_data_path)
        await self.migrate_review_data(legacy_data_path)
        await self.migrate_user_data(legacy_data_path)
        
        # Generate embeddings and populate vector database
        await self.generate_embeddings_and_populate_vector_db()
        
        print("✅ Data migration completed successfully!")
    
    async def migrate_business_data(self, legacy_path: str):
        """
        Migrate business data with enhanced features for AI
        """
        print("📊 Migrating business data...")
        
        # Load legacy business data
        business_files = [
            "OH_restaurant_business.json",
            "NC_restaurant_business.json", 
            "ON_restaurant_business.json",
            "NV_restaurant_business.json",
            "AZ_restaurant_business.json"
        ]
        
        all_businesses = []
        
        for file in business_files:
            file_path = os.path.join(legacy_path, file)
            if os.path.exists(file_path):
                print(f"  Processing {file}...")
                
                df = pd.read_json(file_path, orient='records', lines=True)
                
                for _, row in df.iterrows():
                    business = {
                        "id": row.get("business_id", f"legacy_{len(all_businesses)}"),
                        "name": row.get("name", "Unknown Restaurant"),
                        "categories": self._parse_categories(row.get("categories", [])),
                        "location": f"{row.get('city', '')}, {row.get('state', '')}",
                        "rating": float(row.get("stars", 0)),
                        "review_count": int(row.get("review_count", 0)),
                        "price_range": self._extract_price_range(row.get("attributes", {})),
                        "attributes": self._enhance_attributes(row.get("attributes", {})),
                        "coordinates": {
                            "latitude": row.get("latitude", 0),
                            "longitude": row.get("longitude", 0)
                        },
                        "description": self._generate_business_description(row),
                        "cuisine_types": self._extract_cuisine_types(row.get("categories", [])),
                        "features": self._extract_features(row),
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    all_businesses.append(business)
        
        # Save modern business data
        with open("modern_data/businesses_modern.json", "w") as f:
            json.dump(all_businesses, f, indent=2)
        
        print(f"✅ Migrated {len(all_businesses)} businesses")
    
    async def migrate_review_data(self, legacy_path: str):
        """
        Migrate review data with sentiment analysis and embeddings
        """
        print("📝 Migrating review data...")
        
        # Load legacy review data
        review_files = [
            "Filtered_review_10K.json",
            "Filtered_review_1L.json"
        ]
        
        all_reviews = []
        
        for file in review_files:
            file_path = os.path.join(legacy_path, file)
            if os.path.exists(file_path):
                print(f"  Processing {file}...")
                
                df = pd.read_json(file_path, orient='records', lines=True)
                
                for _, row in df.iterrows():
                    review_text = row.get("text", "")
                    
                    # Enhanced review processing
                    review = {
                        "id": row.get("review_id", f"legacy_review_{len(all_reviews)}"),
                        "business_id": row.get("business_id", ""),
                        "user_id": row.get("user_id", ""),
                        "text": review_text,
                        "rating": float(row.get("stars", 0)),
                        "sentiment_score": self._calculate_sentiment_score(review_text),
                        "sentiment_label": self._get_sentiment_label(review_text),
                        "aspects": self._extract_aspects(review_text),
                        "helpfulness_score": self._calculate_helpfulness_score(row),
                        "created_at": row.get("date", datetime.now().isoformat()),
                        "processed_at": datetime.now().isoformat()
                    }
                    all_reviews.append(review)
        
        # Save modern review data
        with open("modern_data/reviews_modern.json", "w") as f:
            json.dump(all_reviews, f, indent=2)
        
        print(f"✅ Migrated {len(all_reviews)} reviews")
    
    async def migrate_user_data(self, legacy_path: str):
        """
        Migrate user data with preference modeling
        """
        print("👥 Migrating user data...")
        
        # Load legacy user data
        user_file = os.path.join(legacy_path, "user.json")
        
        if os.path.exists(user_file):
            df = pd.read_json(user_file, orient='records', lines=True)
            
            all_users = []
            
            for _, row in df.iterrows():
                user = {
                    "id": row.get("user_id", f"legacy_user_{len(all_users)}"),
                    "name": row.get("name", "Anonymous User"),
                    "review_count": int(row.get("review_count", 0)),
                    "average_rating": float(row.get("average_stars", 0)),
                    "yelping_since": row.get("yelping_since", ""),
                    "friends_count": len(row.get("friends", [])),
                    "elite_years": row.get("elite", []),
                    "preferences": self._infer_user_preferences(row),
                    "behavior_patterns": self._analyze_behavior_patterns(row),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                all_users.append(user)
            
            # Save modern user data
            with open("modern_data/users_modern.json", "w") as f:
                json.dump(all_users, f, indent=2)
            
            print(f"✅ Migrated {len(all_users)} users")
        else:
            print("⚠️ User data file not found, skipping user migration")
    
    async def generate_embeddings_and_populate_vector_db(self):
        """
        Generate embeddings for all text data and populate vector database
        """
        print("🧠 Generating embeddings and populating vector database...")
        
        # Load modern data
        with open("modern_data/businesses_modern.json", "r") as f:
            businesses = json.load(f)
        
        with open("modern_data/reviews_modern.json", "r") as f:
            reviews = json.load(f)
        
        # Generate embeddings for businesses
        business_embeddings = []
        business_metadata = []
        
        for business in businesses:
            # Create text representation for embedding
            text = f"{business['name']} {business['description']} {' '.join(business['cuisine_types'])}"
            
            # Generate embedding
            embedding = self.recommender.embedding_model.encode([text])[0]
            
            business_embeddings.append(embedding.tolist())
            business_metadata.append({
                "business_id": business["id"],
                "name": business["name"],
                "categories": business["categories"],
                "rating": business["rating"],
                "location": business["location"],
                "cuisine_types": business["cuisine_types"]
            })
        
        # Generate embeddings for reviews
        review_embeddings = []
        review_metadata = []
        
        for review in reviews[:10000]:  # Limit for demo
            if review["text"]:
                embedding = self.recommender.embedding_model.encode([review["text"]])[0]
                
                review_embeddings.append(embedding.tolist())
                review_metadata.append({
                    "review_id": review["id"],
                    "business_id": review["business_id"],
                    "user_id": review["user_id"],
                    "rating": review["rating"],
                    "sentiment_score": review["sentiment_score"],
                    "text_preview": review["text"][:100] + "..." if len(review["text"]) > 100 else review["text"]
                })
        
        # Populate vector database
        print("  Adding business embeddings to vector database...")
        self.recommender.collection.add(
            embeddings=business_embeddings,
            metadatas=business_metadata,
            ids=[f"business_{i}" for i in range(len(business_embeddings))]
        )
        
        print("  Adding review embeddings to vector database...")
        self.recommender.collection.add(
            embeddings=review_embeddings,
            metadatas=review_metadata,
            ids=[f"review_{i}" for i in range(len(review_embeddings))]
        )
        
        print("✅ Vector database populated successfully!")
    
    def _parse_categories(self, categories: List[str]) -> List[str]:
        """Parse and clean categories"""
        if isinstance(categories, str):
            return [cat.strip() for cat in categories.split(",")]
        return categories or []
    
    def _extract_price_range(self, attributes: Dict) -> int:
        """Extract price range from attributes"""
        if isinstance(attributes, dict):
            price_info = attributes.get("RestaurantsPriceRange2", "1")
            try:
                return int(price_info)
            except:
                return 1
        return 1
    
    def _enhance_attributes(self, attributes: Dict) -> Dict:
        """Enhance business attributes for AI processing"""
        if not isinstance(attributes, dict):
            return {}
        
        enhanced = {}
        for key, value in attributes.items():
            if isinstance(value, dict):
                enhanced[key] = value
            else:
                enhanced[key] = str(value)
        
        return enhanced
    
    def _generate_business_description(self, row: pd.Series) -> str:
        """Generate comprehensive business description"""
        name = row.get("name", "")
        categories = row.get("categories", [])
        city = row.get("city", "")
        state = row.get("state", "")
        
        description = f"{name} is a restaurant"
        
        if categories:
            description += f" specializing in {', '.join(categories[:3])}"
        
        if city and state:
            description += f" located in {city}, {state}"
        
        return description
    
    def _extract_cuisine_types(self, categories: List[str]) -> List[str]:
        """Extract cuisine types from categories"""
        cuisine_keywords = [
            "italian", "chinese", "mexican", "american", "japanese", "thai",
            "indian", "french", "mediterranean", "korean", "vietnamese"
        ]
        
        cuisines = []
        for category in categories:
            category_lower = category.lower()
            for cuisine in cuisine_keywords:
                if cuisine in category_lower:
                    cuisines.append(cuisine.title())
        
        return list(set(cuisines))
    
    def _extract_features(self, row: pd.Series) -> List[str]:
        """Extract key features from business data"""
        features = []
        
        if row.get("is_open") == 1:
            features.append("open")
        
        if row.get("review_count", 0) > 100:
            features.append("popular")
        
        if row.get("stars", 0) >= 4.0:
            features.append("highly_rated")
        
        return features
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score using simple heuristics"""
        if not text:
            return 0.0
        
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _get_sentiment_label(self, text: str) -> str:
        """Get sentiment label from text"""
        score = self._calculate_sentiment_score(text)
        
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _extract_aspects(self, text: str) -> List[str]:
        """Extract aspects mentioned in review"""
        aspects = []
        
        aspect_keywords = {
            "food": ["food", "dish", "meal", "taste", "flavor"],
            "service": ["service", "staff", "waiter", "server"],
            "ambience": ["atmosphere", "ambience", "decor", "environment"],
            "price": ["price", "cost", "expensive", "cheap", "value"],
            "location": ["location", "parking", "access", "convenient"]
        }
        
        text_lower = text.lower()
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                aspects.append(aspect)
        
        return aspects
    
    def _calculate_helpfulness_score(self, row: pd.Series) -> float:
        """Calculate helpfulness score for review"""
        useful = row.get("useful", 0)
        funny = row.get("funny", 0)
        cool = row.get("cool", 0)
        
        total_reactions = useful + funny + cool
        if total_reactions == 0:
            return 0.0
        
        # Weight useful votes more heavily
        return (useful * 2 + funny + cool) / (total_reactions * 2)
    
    def _infer_user_preferences(self, row: pd.Series) -> Dict[str, Any]:
        """Infer user preferences from profile data"""
        return {
            "cuisine_types": [],  # Would be inferred from review history
            "price_range": [1, 4],
            "location_preference": None,
            "dietary_restrictions": [],
            "ambience_preference": [],
            "time_preference": None
        }
    
    def _analyze_behavior_patterns(self, row: pd.Series) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        return {
            "review_frequency": "moderate",  # Would be calculated from review history
            "rating_tendency": "balanced",   # Would be calculated from rating distribution
            "review_length": "medium",        # Would be calculated from review lengths
            "elite_status": len(row.get("elite", [])) > 0
        }

async def main():
    """Main migration function"""
    config = Config()
    migrator = DataMigrator(config)
    
    # Run migration
    await migrator.migrate_all_data("src/scripts/")

if __name__ == "__main__":
    asyncio.run(main())