import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger
import json

class SQLDatabaseManager:
    """SQL database manager for handling model versioning and training metrics."""
    
    def __init__(self, db_path: str = 'federated_learning.db'):
        self.db_path = db_path
        self._initialize_database()
        
    def _initialize_database(self) -> None:
        """Initialize database tables for model versioning and metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Model versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    parameters BLOB NOT NULL,
                    accuracy REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Client metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS client_metadata (
                    client_id INTEGER PRIMARY KEY,
                    last_active TIMESTAMP,
                    privacy_budget REAL,
                    data_size INTEGER,
                    performance_metrics TEXT
                )
            """)
            
            # Training metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id INTEGER,
                    round_number INTEGER,
                    loss REAL,
                    accuracy REAL,
                    privacy_budget_used REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (client_id) REFERENCES client_metadata (client_id)
                )
            """)
            
            conn.commit()
            logger.info("Initialized SQL database tables")
    
    def save_model_version(self, model_name: str, parameters: bytes, accuracy: float) -> int:
        """Save a new model version to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO model_versions (model_name, parameters, accuracy) VALUES (?, ?, ?)",
                (model_name, parameters, accuracy)
            )
            version_id = cursor.lastrowid
            conn.commit()
            logger.debug(f"Saved model version {version_id} for {model_name}")
            return version_id
    
    def get_model_version(self, version_id: int) -> Optional[Dict]:
        """Retrieve a specific model version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM model_versions WHERE version_id = ?",
                (version_id,)
            )
            result = cursor.fetchone()
            
            if result:
                return {
                    'version_id': result[0],
                    'model_name': result[1],
                    'parameters': result[2],
                    'accuracy': result[3],
                    'created_at': result[4]
                }
            return None
    
    def update_client_metadata(self, client_id: int, privacy_budget: float,
                             data_size: int, performance_metrics: Dict) -> None:
        """Update client metadata in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO client_metadata
                (client_id, last_active, privacy_budget, data_size, performance_metrics)
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?)
                """,
                (client_id, privacy_budget, data_size, json.dumps(performance_metrics))
            )
            conn.commit()
            logger.debug(f"Updated metadata for client {client_id}")
    
    def log_training_metrics(self, client_id: int, round_number: int,
                           loss: float, accuracy: float, privacy_budget_used: float) -> None:
        """Log training metrics for a specific client and round."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO training_metrics
                (client_id, round_number, loss, accuracy, privacy_budget_used)
                VALUES (?, ?, ?, ?, ?)
                """,
                (client_id, round_number, loss, accuracy, privacy_budget_used)
            )
            conn.commit()
            logger.debug(f"Logged training metrics for client {client_id}, round {round_number}")
    
    def get_client_performance_history(self, client_id: int) -> List[Dict]:
        """Retrieve training history for a specific client."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM training_metrics WHERE client_id = ? ORDER BY timestamp DESC",
                (client_id,)
            )
            results = cursor.fetchall()
            
            return [{
                'metric_id': row[0],
                'round_number': row[2],
                'loss': row[3],
                'accuracy': row[4],
                'privacy_budget_used': row[5],
                'timestamp': row[6]
            } for row in results]