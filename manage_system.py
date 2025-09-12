#!/usr/bin/env python3
"""
System Management Script for Modern Yelp Recommendation System
==============================================================

This script provides management commands for the modern AI-powered
Yelp recommendation system.
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemManager:
    """
    Manages the modern Yelp recommendation system
    """
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.config_file = self.project_root / "config" / "system_config.json"
        self.load_config()
    
    def load_config(self):
        """
        Load system configuration
        """
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "services": {
                    "api": {"port": 8000, "status": "unknown"},
                    "redis": {"port": 6379, "status": "unknown"},
                    "vector_db": {"status": "unknown"}
                },
                "models": {
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
                },
                "performance": {
                    "total_requests": 0,
                    "avg_response_time_ms": 0,
                    "success_rate": 0
                }
            }
    
    def save_config(self):
        """
        Save system configuration
        """
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    async def status(self):
        """
        Check system status
        """
        print("🔍 Checking Modern Yelp Recommendation System Status")
        print("=" * 60)
        
        # Check API service
        await self._check_api_status()
        
        # Check Redis
        await self._check_redis_status()
        
        # Check vector database
        await self._check_vector_db_status()
        
        # Check models
        await self._check_models_status()
        
        # Show performance metrics
        self._show_performance_metrics()
        
        print("\n✅ Status check complete!")
    
    async def _check_api_status(self):
        """
        Check API service status
        """
        print("\n🌐 API Service:")
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("  ✅ API service is running")
                    self.config["services"]["api"]["status"] = "running"
                else:
                    print("  ⚠️  API service responded with error")
                    self.config["services"]["api"]["status"] = "error"
        except Exception as e:
            print(f"  ❌ API service is not running: {e}")
            self.config["services"]["api"]["status"] = "stopped"
    
    async def _check_redis_status(self):
        """
        Check Redis status
        """
        print("\n⚡ Redis Cache:")
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            print("  ✅ Redis is running")
            self.config["services"]["redis"]["status"] = "running"
        except Exception as e:
            print(f"  ❌ Redis is not running: {e}")
            self.config["services"]["redis"]["status"] = "stopped"
    
    async def _check_vector_db_status(self):
        """
        Check vector database status
        """
        print("\n🗄️  Vector Database:")
        try:
            import chromadb
            client = chromadb.Client()
            collections = client.list_collections()
            print(f"  ✅ Vector database is running ({len(collections)} collections)")
            self.config["services"]["vector_db"]["status"] = "running"
        except Exception as e:
            print(f"  ❌ Vector database is not running: {e}")
            self.config["services"]["vector_db"]["status"] = "stopped"
    
    async def _check_models_status(self):
        """
        Check AI models status
        """
        print("\n🧠 AI Models:")
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification
            
            # Check embedding model
            model = SentenceTransformer(self.config["models"]["embedding_model"])
            print(f"  ✅ Embedding model loaded: {self.config['models']['embedding_model']}")
            
            # Check rerank model
            rerank_model = AutoModelForSequenceClassification.from_pretrained(
                self.config["models"]["rerank_model"]
            )
            print(f"  ✅ Rerank model loaded: {self.config['models']['rerank_model']}")
            
        except Exception as e:
            print(f"  ❌ Models not loaded: {e}")
    
    def _show_performance_metrics(self):
        """
        Show performance metrics
        """
        print("\n📊 Performance Metrics:")
        perf = self.config["performance"]
        print(f"  Total requests: {perf['total_requests']}")
        print(f"  Average response time: {perf['avg_response_time_ms']:.2f}ms")
        print(f"  Success rate: {perf['success_rate']:.1f}%")
    
    async def start(self):
        """
        Start the system
        """
        print("🚀 Starting Modern Yelp Recommendation System...")
        
        # Start Redis
        await self._start_redis()
        
        # Start API service
        await self._start_api()
        
        print("✅ System started successfully!")
    
    async def _start_redis(self):
        """
        Start Redis service
        """
        print("  Starting Redis...")
        try:
            subprocess.Popen(["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            await asyncio.sleep(2)
            print("    ✅ Redis started")
        except Exception as e:
            print(f"    ❌ Failed to start Redis: {e}")
    
    async def _start_api(self):
        """
        Start API service
        """
        print("  Starting API service...")
        try:
            # Start the FastAPI application
            subprocess.Popen([
                sys.executable, "modern_yelp_recommender.py"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            await asyncio.sleep(3)
            print("    ✅ API service started")
        except Exception as e:
            print(f"    ❌ Failed to start API service: {e}")
    
    async def stop(self):
        """
        Stop the system
        """
        print("🛑 Stopping Modern Yelp Recommendation System...")
        
        # Stop Redis
        try:
            subprocess.run(["redis-cli", "shutdown"], capture_output=True)
            print("  ✅ Redis stopped")
        except Exception as e:
            print(f"  ⚠️  Redis stop failed: {e}")
        
        # Stop API service
        try:
            subprocess.run(["pkill", "-f", "modern_yelp_recommender.py"], capture_output=True)
            print("  ✅ API service stopped")
        except Exception as e:
            print(f"  ⚠️  API service stop failed: {e}")
        
        print("✅ System stopped!")
    
    async def restart(self):
        """
        Restart the system
        """
        print("🔄 Restarting Modern Yelp Recommendation System...")
        await self.stop()
        await asyncio.sleep(2)
        await self.start()
    
    async def test(self):
        """
        Run system tests
        """
        print("🧪 Running system tests...")
        
        try:
            # Run the test script
            result = subprocess.run([
                sys.executable, "test_modern_system.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed!")
                print(result.stdout)
            else:
                print("❌ Some tests failed!")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
    
    async def update_models(self):
        """
        Update AI models
        """
        print("🔄 Updating AI models...")
        
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # Update embedding model
            print("  Updating embedding model...")
            SentenceTransformer(self.config["models"]["embedding_model"])
            print("    ✅ Embedding model updated")
            
            # Update rerank model
            print("  Updating rerank model...")
            AutoModelForSequenceClassification.from_pretrained(
                self.config["models"]["rerank_model"]
            )
            AutoTokenizer.from_pretrained(
                self.config["models"]["rerank_model"]
            )
            print("    ✅ Rerank model updated")
            
            print("✅ Models updated successfully!")
            
        except Exception as e:
            print(f"❌ Model update failed: {e}")
    
    async def migrate_data(self):
        """
        Migrate data from legacy system
        """
        print("📊 Migrating data from legacy system...")
        
        try:
            # Run data migration
            result = subprocess.run([
                sys.executable, "data_migration.py"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Data migration completed!")
                print(result.stdout)
            else:
                print("❌ Data migration failed!")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Data migration execution failed: {e}")
    
    def logs(self, service: str = "all", lines: int = 50):
        """
        Show system logs
        """
        print(f"📋 Showing logs for {service} (last {lines} lines)")
        
        log_files = {
            "api": "logs/api.log",
            "system": "logs/system.log",
            "learning": "logs/learning.log"
        }
        
        if service == "all":
            for service_name, log_file in log_files.items():
                self._show_log_file(service_name, log_file, lines)
        elif service in log_files:
            self._show_log_file(service, log_files[service], lines)
        else:
            print(f"❌ Unknown service: {service}")
    
    def _show_log_file(self, service_name: str, log_file: str, lines: int):
        """
        Show log file contents
        """
        log_path = Path(log_file)
        if log_path.exists():
            print(f"\n📄 {service_name.upper()} LOGS:")
            print("-" * 40)
            try:
                with open(log_path, 'r') as f:
                    log_lines = f.readlines()
                    for line in log_lines[-lines:]:
                        print(line.rstrip())
            except Exception as e:
                print(f"❌ Error reading log file: {e}")
        else:
            print(f"⚠️  Log file not found: {log_file}")
    
    def config_show(self):
        """
        Show current configuration
        """
        print("⚙️  Current System Configuration")
        print("=" * 40)
        print(json.dumps(self.config, indent=2))
    
    def config_update(self, key: str, value: str):
        """
        Update configuration
        """
        print(f"⚙️  Updating configuration: {key} = {value}")
        
        try:
            # Parse the value
            if value.lower() in ['true', 'false']:
                parsed_value = value.lower() == 'true'
            elif value.isdigit():
                parsed_value = int(value)
            else:
                parsed_value = value
            
            # Update the configuration
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = parsed_value
            
            # Save configuration
            self.save_config()
            print("✅ Configuration updated!")
            
        except Exception as e:
            print(f"❌ Configuration update failed: {e}")

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Manage Modern Yelp Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_system.py status
  python manage_system.py start
  python manage_system.py stop
  python manage_system.py restart
  python manage_system.py test
  python manage_system.py logs --service api --lines 100
  python manage_system.py config-show
  python manage_system.py config-update --key "services.api.port" --value "8001"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Check system status')
    
    # Start command
    subparsers.add_parser('start', help='Start the system')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop the system')
    
    # Restart command
    subparsers.add_parser('restart', help='Restart the system')
    
    # Test command
    subparsers.add_parser('test', help='Run system tests')
    
    # Update models command
    subparsers.add_parser('update-models', help='Update AI models')
    
    # Migrate data command
    subparsers.add_parser('migrate-data', help='Migrate data from legacy system')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show system logs')
    logs_parser.add_argument('--service', default='all', 
                           choices=['all', 'api', 'system', 'learning'],
                           help='Service to show logs for')
    logs_parser.add_argument('--lines', type=int, default=50,
                           help='Number of lines to show')
    
    # Config commands
    subparsers.add_parser('config-show', help='Show current configuration')
    
    config_update_parser = subparsers.add_parser('config-update', help='Update configuration')
    config_update_parser.add_argument('--key', required=True, help='Configuration key')
    config_update_parser.add_argument('--value', required=True, help='Configuration value')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create system manager
    manager = SystemManager()
    
    # Execute command
    if args.command == 'status':
        asyncio.run(manager.status())
    elif args.command == 'start':
        asyncio.run(manager.start())
    elif args.command == 'stop':
        asyncio.run(manager.stop())
    elif args.command == 'restart':
        asyncio.run(manager.restart())
    elif args.command == 'test':
        asyncio.run(manager.test())
    elif args.command == 'update-models':
        asyncio.run(manager.update_models())
    elif args.command == 'migrate-data':
        asyncio.run(manager.migrate_data())
    elif args.command == 'logs':
        manager.logs(args.service, args.lines)
    elif args.command == 'config-show':
        manager.config_show()
    elif args.command == 'config-update':
        manager.config_update(args.key, args.value)

if __name__ == "__main__":
    main()