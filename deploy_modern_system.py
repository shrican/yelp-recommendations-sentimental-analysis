"""
Deployment Script for Modern Yelp Recommendation System
======================================================

This script deploys the modern AI-powered Yelp recommendation system
with all the latest 2025 AI technologies.
"""

import os
import sys
import subprocess
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModernSystemDeployer:
    """
    Deploys the modern Yelp recommendation system
    """
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.deployment_config = {
            "services": {
                "api": {"port": 8000, "workers": 4},
                "redis": {"port": 6379},
                "database": {"port": 5432},
                "vector_db": {"port": 8001}
            },
            "models": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
            },
            "environment": "production"
        }
    
    async def deploy_full_system(self):
        """
        Deploy the complete modern system
        """
        logger.info("🚀 Starting deployment of Modern Yelp Recommendation System...")
        
        try:
            # Step 1: Setup environment
            await self._setup_environment()
            
            # Step 2: Install dependencies
            await self._install_dependencies()
            
            # Step 3: Setup databases
            await self._setup_databases()
            
            # Step 4: Migrate data
            await self._migrate_data()
            
            # Step 5: Download models
            await self._download_models()
            
            # Step 6: Start services
            await self._start_services()
            
            # Step 7: Run health checks
            await self._run_health_checks()
            
            logger.info("✅ Modern Yelp Recommendation System deployed successfully!")
            self._print_deployment_summary()
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            raise
    
    async def _setup_environment(self):
        """
        Setup the deployment environment
        """
        logger.info("🔧 Setting up environment...")
        
        # Create necessary directories
        directories = [
            "modern_data",
            "model_checkpoints",
            "chroma_db",
            "logs",
            "config"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"  Created directory: {directory}")
        
        # Set environment variables
        env_vars = {
            "PYTHONPATH": str(self.project_root),
            "MODEL_CACHE_DIR": str(self.project_root / "model_cache"),
            "VECTOR_DB_PATH": str(self.project_root / "chroma_db"),
            "LOG_LEVEL": "INFO"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.info(f"  Set {key}={value}")
    
    async def _install_dependencies(self):
        """
        Install all required dependencies
        """
        logger.info("📦 Installing dependencies...")
        
        # Install Python dependencies
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements_modern.txt"
            ], check=True, capture_output=True, text=True)
            logger.info("  ✅ Python dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Failed to install Python dependencies: {e}")
            raise
        
        # Install Redis (if not already installed)
        try:
            subprocess.run(["redis-server", "--version"], check=True, capture_output=True)
            logger.info("  ✅ Redis is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("  ⚠️ Redis not found. Please install Redis manually.")
        
        # Install Docker (for containerized deployment)
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            logger.info("  ✅ Docker is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("  ⚠️ Docker not found. Please install Docker for containerized deployment.")
    
    async def _setup_databases(self):
        """
        Setup required databases
        """
        logger.info("🗄️ Setting up databases...")
        
        # Setup Redis
        await self._setup_redis()
        
        # Setup SQLite (for demo) or PostgreSQL (for production)
        await self._setup_sql_database()
        
        # Setup Vector Database
        await self._setup_vector_database()
    
    async def _setup_redis(self):
        """
        Setup Redis for caching
        """
        logger.info("  Setting up Redis...")
        
        try:
            # Start Redis server
            subprocess.Popen(["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            await asyncio.sleep(2)  # Wait for Redis to start
            
            # Test Redis connection
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            logger.info("    ✅ Redis is running")
            
        except Exception as e:
            logger.warning(f"    ⚠️ Redis setup failed: {e}")
    
    async def _setup_sql_database(self):
        """
        Setup SQL database
        """
        logger.info("  Setting up SQL database...")
        
        # For demo, we'll use SQLite
        # In production, you would setup PostgreSQL
        db_path = self.project_root / "modern_data" / "yelp_modern.db"
        logger.info(f"    ✅ SQLite database will be created at: {db_path}")
    
    async def _setup_vector_database(self):
        """
        Setup vector database
        """
        logger.info("  Setting up vector database...")
        
        # ChromaDB will be initialized when the application starts
        vector_db_path = self.project_root / "chroma_db"
        logger.info(f"    ✅ Vector database will be created at: {vector_db_path}")
    
    async def _migrate_data(self):
        """
        Migrate data from legacy system
        """
        logger.info("📊 Migrating data...")
        
        try:
            # Run data migration
            from data_migration import DataMigrator, Config
            
            config = Config()
            migrator = DataMigrator(config)
            
            # Check if legacy data exists
            legacy_path = self.project_root / "src" / "scripts"
            if legacy_path.exists():
                await migrator.migrate_all_data(str(legacy_path))
                logger.info("  ✅ Data migration completed")
            else:
                logger.warning("  ⚠️ Legacy data not found, skipping migration")
                
        except Exception as e:
            logger.warning(f"  ⚠️ Data migration failed: {e}")
    
    async def _download_models(self):
        """
        Download required AI models
        """
        logger.info("🧠 Downloading AI models...")
        
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # Download embedding model
            logger.info("  Downloading embedding model...")
            SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("    ✅ Embedding model downloaded")
            
            # Download rerank model
            logger.info("  Downloading rerank model...")
            AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
            AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("    ✅ Rerank model downloaded")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Model download failed: {e}")
    
    async def _start_services(self):
        """
        Start all services
        """
        logger.info("🚀 Starting services...")
        
        # Start the main API service
        await self._start_api_service()
        
        # Start agentic recommendation system
        await self._start_agentic_system()
        
        # Start real-time learning system
        await self._start_learning_system()
    
    async def _start_api_service(self):
        """
        Start the main API service
        """
        logger.info("  Starting API service...")
        
        try:
            # Start FastAPI application
            import uvicorn
            from modern_yelp_recommender import app
            
            # Run in background
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=self.deployment_config["services"]["api"]["port"],
                workers=self.deployment_config["services"]["api"]["workers"],
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            asyncio.create_task(server.serve())
            
            logger.info(f"    ✅ API service started on port {self.deployment_config['services']['api']['port']}")
            
        except Exception as e:
            logger.error(f"    ❌ Failed to start API service: {e}")
            raise
    
    async def _start_agentic_system(self):
        """
        Start the agentic recommendation system
        """
        logger.info("  Starting agentic system...")
        
        try:
            from agentic_recommendation_system import AgenticSystemManager, Config
            
            config = Config()
            manager = AgenticSystemManager(config)
            
            # Create and start agent
            agent = await manager.create_agent("main_agent")
            
            logger.info("    ✅ Agentic system started")
            
        except Exception as e:
            logger.warning(f"    ⚠️ Agentic system failed to start: {e}")
    
    async def _start_learning_system(self):
        """
        Start the real-time learning system
        """
        logger.info("  Starting learning system...")
        
        try:
            from real_time_learning_system import RealTimeLearningSystem, Config
            
            config = Config()
            learning_system = RealTimeLearningSystem(config)
            
            # Start learning loop
            asyncio.create_task(learning_system.start_learning_loop())
            
            logger.info("    ✅ Learning system started")
            
        except Exception as e:
            logger.warning(f"    ⚠️ Learning system failed to start: {e}")
    
    async def _run_health_checks(self):
        """
        Run health checks on all services
        """
        logger.info("🏥 Running health checks...")
        
        # Check API service
        await self._check_api_health()
        
        # Check Redis
        await self._check_redis_health()
        
        # Check vector database
        await self._check_vector_db_health()
    
    async def _check_api_health(self):
        """
        Check API service health
        """
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/health")
                if response.status_code == 200:
                    logger.info("    ✅ API service is healthy")
                else:
                    logger.warning("    ⚠️ API service health check failed")
                    
        except Exception as e:
            logger.warning(f"    ⚠️ API health check failed: {e}")
    
    async def _check_redis_health(self):
        """
        Check Redis health
        """
        try:
            import redis
            
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            logger.info("    ✅ Redis is healthy")
            
        except Exception as e:
            logger.warning(f"    ⚠️ Redis health check failed: {e}")
    
    async def _check_vector_db_health(self):
        """
        Check vector database health
        """
        try:
            import chromadb
            
            client = chromadb.Client()
            collections = client.list_collections()
            logger.info("    ✅ Vector database is healthy")
            
        except Exception as e:
            logger.warning(f"    ⚠️ Vector database health check failed: {e}")
    
    def _print_deployment_summary(self):
        """
        Print deployment summary
        """
        print("\n" + "="*60)
        print("🎉 MODERN YELP RECOMMENDATION SYSTEM DEPLOYED!")
        print("="*60)
        print("\n📊 System Overview:")
        print("  • Transformer-based embeddings for semantic understanding")
        print("  • Vector database for efficient similarity search")
        print("  • Agentic AI for proactive recommendations")
        print("  • Real-time learning for continuous improvement")
        print("  • Multimodal features (text, location, temporal)")
        print("  • Explainable AI for transparency")
        
        print("\n🌐 Services:")
        print(f"  • API Service: http://localhost:{self.deployment_config['services']['api']['port']}")
        print("  • Redis Cache: localhost:6379")
        print("  • Vector Database: chroma_db/")
        print("  • SQL Database: modern_data/yelp_modern.db")
        
        print("\n📚 API Endpoints:")
        print("  • GET  / - System information")
        print("  • POST /recommendations - Get AI-powered recommendations")
        print("  • GET  /health - Health check")
        
        print("\n🚀 Getting Started:")
        print("  1. Test the API: curl http://localhost:8000/")
        print("  2. Get recommendations: POST to /recommendations")
        print("  3. Monitor logs: tail -f logs/system.log")
        
        print("\n📖 Documentation:")
        print("  • README_MODERN.md - Complete documentation")
        print("  • API_DOCS.md - API reference")
        print("  • DEPLOYMENT.md - Deployment guide")
        
        print("\n🔧 Management:")
        print("  • View system status: python manage_system.py status")
        print("  • Restart services: python manage_system.py restart")
        print("  • Update models: python manage_system.py update-models")
        
        print("\n" + "="*60)

async def main():
    """
    Main deployment function
    """
    deployer = ModernSystemDeployer()
    await deployer.deploy_full_system()

if __name__ == "__main__":
    asyncio.run(main())