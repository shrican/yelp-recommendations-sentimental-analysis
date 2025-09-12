"""
Test Script for Modern Yelp Recommendation System
=================================================

This script demonstrates the capabilities of the modern AI-powered
Yelp recommendation system.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any

from modern_yelp_recommender import ModernYelpRecommender, Config, RecommendationRequest, RecommendationType, UserPreference

class ModernSystemTester:
    """
    Test the modern Yelp recommendation system
    """
    
    def __init__(self):
        self.config = Config()
        self.recommender = ModernYelpRecommender(self.config)
        self.test_results = []
    
    async def run_all_tests(self):
        """
        Run all system tests
        """
        print("🧪 Starting Modern Yelp Recommendation System Tests")
        print("=" * 60)
        
        # Test 1: Semantic Similarity Recommendations
        await self.test_semantic_similarity()
        
        # Test 2: Multimodal Recommendations
        await self.test_multimodal_recommendations()
        
        # Test 3: Contextual Recommendations
        await self.test_contextual_recommendations()
        
        # Test 4: Performance Tests
        await self.test_performance()
        
        # Test 5: Explainability Tests
        await self.test_explainability()
        
        # Print results
        self.print_test_results()
    
    async def test_semantic_similarity(self):
        """
        Test semantic similarity recommendations
        """
        print("\n🔍 Testing Semantic Similarity Recommendations")
        print("-" * 50)
        
        test_cases = [
            {
                "query": "best Italian restaurant for a romantic dinner",
                "expected_keywords": ["italian", "romantic", "dinner"]
            },
            {
                "query": "cheap pizza place near me",
                "expected_keywords": ["pizza", "cheap", "near"]
            },
            {
                "query": "healthy breakfast spot with good coffee",
                "expected_keywords": ["healthy", "breakfast", "coffee"]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n  Test {i}: '{test_case['query']}'")
            
            request = RecommendationRequest(
                query_text=test_case["query"],
                recommendation_type=RecommendationType.SEMANTIC_SIMILARITY,
                count=3,
                include_explanations=True
            )
            
            start_time = time.time()
            try:
                response = await self.recommender.get_recommendations(request)
                end_time = time.time()
                
                print(f"    ✅ Success: {len(response.recommendations)} recommendations")
                print(f"    ⏱️  Response time: {(end_time - start_time) * 1000:.2f}ms")
                print(f"    🎯 Confidence scores: {[f'{score:.2f}' for score in response.confidence_scores]}")
                
                # Show first recommendation
                if response.recommendations:
                    first_rec = response.recommendations[0]
                    print(f"    🏆 Top recommendation: {first_rec.get('name', 'Unknown')}")
                    print(f"    📝 Explanation: {response.explanations.get(first_rec.get('business_id', ''), 'No explanation')}")
                
                self.test_results.append({
                    "test": f"Semantic Similarity {i}",
                    "status": "PASS",
                    "response_time_ms": (end_time - start_time) * 1000,
                    "recommendations_count": len(response.recommendations)
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.test_results.append({
                    "test": f"Semantic Similarity {i}",
                    "status": "FAIL",
                    "error": str(e)
                })
    
    async def test_multimodal_recommendations(self):
        """
        Test multimodal recommendations
        """
        print("\n🎯 Testing Multimodal Recommendations")
        print("-" * 50)
        
        test_cases = [
            {
                "user_id": "test_user_1",
                "preferences": UserPreference(
                    cuisine_types=["italian", "japanese"],
                    price_range=(2, 4),
                    dietary_restrictions=["vegetarian"],
                    time_preference="dinner"
                ),
                "location": "San Francisco"
            },
            {
                "user_id": "test_user_2", 
                "preferences": UserPreference(
                    cuisine_types=["mexican", "american"],
                    price_range=(1, 3),
                    ambience_preference=["casual", "family-friendly"]
                ),
                "location": "New York"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n  Test {i}: User {test_case['user_id']}")
            
            request = RecommendationRequest(
                user_id=test_case["user_id"],
                preferences=test_case["preferences"],
                location=test_case["location"],
                recommendation_type=RecommendationType.MULTIMODAL,
                count=5,
                include_explanations=True
            )
            
            start_time = time.time()
            try:
                response = await self.recommender.get_recommendations(request)
                end_time = time.time()
                
                print(f"    ✅ Success: {len(response.recommendations)} recommendations")
                print(f"    ⏱️  Response time: {(end_time - start_time) * 1000:.2f}ms")
                print(f"    🎯 Average confidence: {sum(response.confidence_scores) / len(response.confidence_scores):.2f}")
                
                # Show recommendations
                for j, rec in enumerate(response.recommendations[:3], 1):
                    print(f"    {j}. {rec.get('name', 'Unknown')} (Score: {rec.get('similarity_score', 0):.2f})")
                
                self.test_results.append({
                    "test": f"Multimodal {i}",
                    "status": "PASS",
                    "response_time_ms": (end_time - start_time) * 1000,
                    "recommendations_count": len(response.recommendations)
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.test_results.append({
                    "test": f"Multimodal {i}",
                    "status": "FAIL",
                    "error": str(e)
                })
    
    async def test_contextual_recommendations(self):
        """
        Test contextual recommendations
        """
        print("\n🌍 Testing Contextual Recommendations")
        print("-" * 50)
        
        # Test different contexts
        contexts = [
            {
                "description": "Morning coffee run",
                "time": "08:30",
                "location": "Downtown",
                "weather": "sunny"
            },
            {
                "description": "Late night dinner",
                "time": "21:00", 
                "location": "Suburbs",
                "weather": "rainy"
            },
            {
                "description": "Weekend brunch",
                "time": "11:00",
                "location": "Beach area",
                "weather": "sunny"
            }
        ]
        
        for i, context in enumerate(contexts, 1):
            print(f"\n  Test {i}: {context['description']}")
            
            request = RecommendationRequest(
                query_text=f"restaurants for {context['description']}",
                location=context["location"],
                recommendation_type=RecommendationType.CONTEXTUAL,
                count=3,
                include_explanations=True
            )
            
            start_time = time.time()
            try:
                response = await self.recommender.get_recommendations(request)
                end_time = time.time()
                
                print(f"    ✅ Success: {len(response.recommendations)} recommendations")
                print(f"    ⏱️  Response time: {(end_time - start_time) * 1000:.2f}ms")
                
                # Show context-aware recommendations
                for j, rec in enumerate(response.recommendations, 1):
                    print(f"    {j}. {rec.get('name', 'Unknown')} - {rec.get('reason', 'No reason provided')}")
                
                self.test_results.append({
                    "test": f"Contextual {i}",
                    "status": "PASS",
                    "response_time_ms": (end_time - start_time) * 1000,
                    "recommendations_count": len(response.recommendations)
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.test_results.append({
                    "test": f"Contextual {i}",
                    "status": "FAIL",
                    "error": str(e)
                })
    
    async def test_performance(self):
        """
        Test system performance
        """
        print("\n⚡ Testing Performance")
        print("-" * 50)
        
        # Test concurrent requests
        print("  Testing concurrent requests...")
        
        async def make_request(request_id: int):
            request = RecommendationRequest(
                query_text=f"test query {request_id}",
                recommendation_type=RecommendationType.SEMANTIC_SIMILARITY,
                count=5
            )
            
            start_time = time.time()
            try:
                response = await self.recommender.get_recommendations(request)
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "success": True,
                    "response_time_ms": (end_time - start_time) * 1000,
                    "recommendations_count": len(response.recommendations)
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Make 10 concurrent requests
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            avg_response_time = sum(r["response_time_ms"] for r in successful_requests) / len(successful_requests)
            max_response_time = max(r["response_time_ms"] for r in successful_requests)
            min_response_time = min(r["response_time_ms"] for r in successful_requests)
            
            print(f"    ✅ Successful requests: {len(successful_requests)}/10")
            print(f"    ⏱️  Average response time: {avg_response_time:.2f}ms")
            print(f"    ⏱️  Min response time: {min_response_time:.2f}ms")
            print(f"    ⏱️  Max response time: {max_response_time:.2f}ms")
            
            self.test_results.append({
                "test": "Concurrent Performance",
                "status": "PASS",
                "successful_requests": len(successful_requests),
                "avg_response_time_ms": avg_response_time
            })
        else:
            print(f"    ❌ All requests failed")
            self.test_results.append({
                "test": "Concurrent Performance",
                "status": "FAIL",
                "error": "All requests failed"
            })
        
        if failed_requests:
            print(f"    ⚠️  Failed requests: {len(failed_requests)}")
            for failed in failed_requests[:3]:  # Show first 3 errors
                print(f"      Request {failed['request_id']}: {failed['error']}")
    
    async def test_explainability(self):
        """
        Test explainable AI features
        """
        print("\n🔍 Testing Explainable AI")
        print("-" * 50)
        
        request = RecommendationRequest(
            query_text="best sushi restaurant for a business lunch",
            recommendation_type=RecommendationType.MULTIMODAL,
            count=3,
            include_explanations=True
        )
        
        try:
            response = await self.recommender.get_recommendations(request)
            
            print(f"    ✅ Generated {len(response.recommendations)} recommendations with explanations")
            
            # Check if explanations are provided
            has_explanations = response.explanations is not None and len(response.explanations) > 0
            has_confidence_scores = len(response.confidence_scores) > 0
            has_metadata = response.metadata is not None
            
            print(f"    📝 Explanations provided: {'✅' if has_explanations else '❌'}")
            print(f"    🎯 Confidence scores provided: {'✅' if has_confidence_scores else '❌'}")
            print(f"    📊 Metadata provided: {'✅' if has_metadata else '❌'}")
            
            # Show example explanation
            if has_explanations and response.recommendations:
                first_rec = response.recommendations[0]
                business_id = first_rec.get("business_id", "")
                explanation = response.explanations.get(business_id, "No explanation available")
                print(f"    💡 Example explanation: {explanation}")
            
            # Show metadata
            if has_metadata:
                print(f"    🔧 Model version: {response.metadata.get('model_version', 'Unknown')}")
                print(f"    🧠 AI technologies: {', '.join(response.metadata.get('ai_technologies', []))}")
            
            self.test_results.append({
                "test": "Explainable AI",
                "status": "PASS",
                "explanations_provided": has_explanations,
                "confidence_scores_provided": has_confidence_scores,
                "metadata_provided": has_metadata
            })
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            self.test_results.append({
                "test": "Explainable AI",
                "status": "FAIL",
                "error": str(e)
            })
    
    def print_test_results(self):
        """
        Print comprehensive test results
        """
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASS"])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📈 Overall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests} ✅")
        print(f"  Failed: {failed_tests} ❌")
        print(f"  Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            print(f"  {status_icon} {result['test']}: {result['status']}")
            
            if result["status"] == "PASS":
                if "response_time_ms" in result:
                    print(f"      Response time: {result['response_time_ms']:.2f}ms")
                if "recommendations_count" in result:
                    print(f"      Recommendations: {result['recommendations_count']}")
            else:
                if "error" in result:
                    print(f"      Error: {result['error']}")
        
        # Performance summary
        response_times = [r["response_time_ms"] for r in self.test_results 
                         if r["status"] == "PASS" and "response_time_ms" in r]
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            print(f"\n⚡ Performance Summary:")
            print(f"  Average response time: {avg_response_time:.2f}ms")
            print(f"  Fastest response: {min(response_times):.2f}ms")
            print(f"  Slowest response: {max(response_times):.2f}ms")
        
        print(f"\n🎉 Modern Yelp Recommendation System Testing Complete!")
        print("=" * 60)

async def main():
    """
    Main test function
    """
    tester = ModernSystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())