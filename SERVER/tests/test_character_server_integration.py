"""
Integration tests for character server with external dependencies.
Testing framework: pytest
These tests would integrate with real databases, APIs, and external services.
"""

import pytest
import json
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the SERVER directory to the path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from test_character_server import MockCharacterServer


class TestCharacterServerDatabaseIntegration:
    """Integration tests with database systems."""
    
    @pytest.fixture
    def temp_database(self):
        """Create a temporary SQLite database for testing."""
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_path = db_file.name
        db_file.close()
        
        # Initialize database with character table
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                class TEXT NOT NULL,
                level INTEGER NOT NULL,
                data TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    def test_character_persistence_simulation(self, temp_database):
        """Test character data persistence simulation with SQLite."""
        # This simulates what real database integration would look like
        conn = sqlite3.connect(temp_database)
        cursor = conn.cursor()
        
        # Create character data
        char_data = {
            'name': 'PersistentHero',
            'class': 'Warrior',
            'level': 25,
            'hp': 250,
            'mp': 100
        }
        
        # Insert character into database
        cursor.execute(
            'INSERT INTO characters (name, class, level, data) VALUES (?, ?, ?, ?)',
            (char_data['name'], char_data['class'], char_data['level'], json.dumps(char_data))
        )
        conn.commit()
        
        # Retrieve character from database
        cursor.execute('SELECT * FROM characters WHERE name = ?', (char_data['name'],))
        row = cursor.fetchone()
        
        assert row is not None
        assert row[1] == char_data['name']  # name
        assert row[2] == char_data['class']  # class
        assert row[3] == char_data['level']  # level
        
        retrieved_data = json.loads(row[4])  # data
        assert retrieved_data['hp'] == char_data['hp']
        
        conn.close()
    
    def test_character_batch_database_operations(self, temp_database):
        """Test batch database operations for character management."""
        conn = sqlite3.connect(temp_database)
        cursor = conn.cursor()
        
        # Batch insert characters
        characters = []
        for i in range(100):
            char_data = {
                'name': f'BatchHero_{i}',
                'class': 'BatchClass',
                'level': i + 1,
                'batch_id': i // 10
            }
            characters.append((
                char_data['name'],
                char_data['class'], 
                char_data['level'],
                json.dumps(char_data)
            ))
        
        cursor.executemany(
            'INSERT INTO characters (name, class, level, data) VALUES (?, ?, ?, ?)',
            characters
        )
        conn.commit()
        
        # Query batch results
        cursor.execute('SELECT COUNT(*) FROM characters WHERE class = ?', ('BatchClass',))
        count = cursor.fetchone()[0]
        assert count == 100
        
        # Test batch update
        cursor.execute('UPDATE characters SET level = level + 10 WHERE class = ?', ('BatchClass',))
        conn.commit()
        
        # Verify batch update
        cursor.execute('SELECT AVG(level) FROM characters WHERE class = ?', ('BatchClass',))
        avg_level = cursor.fetchone()[0]
        assert avg_level > 50  # Original average was ~50, now should be ~60
        
        conn.close()


class TestCharacterServerAPIIntegration:
    """Integration tests with external APIs and services."""
    
    @patch('requests.get')
    def test_character_external_validation_service(self, mock_get):
        """Test integration with external character validation service."""
        # Mock external validation API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'is_valid': True,
            'validation_score': 95,
            'suggestions': []
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        char_data = {
            'name': 'ExternalValidatedHero',
            'class': 'Warrior',
            'level': 10
        }
        
        # Simulate external validation call
        import requests
        response = requests.get('https://api.example.com/validate-character', 
                              json=char_data)
        
        validation_result = response.json()
        assert validation_result['is_valid'] is True
        assert validation_result['validation_score'] >= 90
    
    @patch('requests.post')
    def test_character_statistics_reporting(self, mock_post):
        """Test integration with statistics reporting service."""
        # Mock statistics API response
        mock_response = Mock()
        mock_response.json.return_value = {'status': 'success', 'report_id': 'RPT123'}
        mock_response.status_code = 201
        mock_post.return_value = mock_response
        
        # Create character server with characters
        server = MockCharacterServer()
        chars = []
        for i in range(10):
            char_data = {
                'name': f'StatsHero_{i}',
                'class': 'Warrior',
                'level': (i + 1) * 10
            }
            chars.append(server.create_character(char_data))
        
        # Generate statistics
        stats = {
            'total_characters': len(chars),
            'average_level': sum(c['level'] for c in chars) / len(chars),
            'class_distribution': {'Warrior': len(chars)}
        }
        
        # Send statistics to external service
        import requests
        response = requests.post('https://api.example.com/character-stats',
                               json=stats)
        
        result = response.json()
        assert result['status'] == 'success'
        assert 'report_id' in result


class TestCharacterServerFileSystemIntegration:
    """Integration tests with file system operations."""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for file operations."""
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_character_data_export_import(self, temp_directory):
        """Test character data export and import functionality."""
        server = MockCharacterServer()
        
        # Create test characters
        test_chars = []
        for i in range(5):
            char_data = {
                'name': f'ExportHero_{i}',
                'class': 'Exporter',
                'level': (i + 1) * 10,
                'export_batch': 'batch_001'
            }
            char = server.create_character(char_data)
            test_chars.append(char)
        
        # Export characters to JSON file
        export_file = os.path.join(temp_directory, 'characters_export.json')
        export_data = {
            'version': '1.0',
            'export_timestamp': '2023-01-01T00:00:00Z',
            'characters': test_chars
        }
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        # Verify export file exists and is valid
        assert os.path.exists(export_file)
        
        # Import characters from file (simulate new server)
        import_server = MockCharacterServer()
        
        with open(export_file, 'r') as f:
            imported_data = json.load(f)
        
        imported_chars = []
        for char_data in imported_data['characters']:
            # Remove ID for import (will get new ID)
            import_char_data = {k: v for k, v in char_data.items() if k != 'id'}
            imported_char = import_server.create_character(import_char_data)
            imported_chars.append(imported_char)
        
        # Verify import success
        assert len(imported_chars) == len(test_chars)
        for original, imported in zip(test_chars, imported_chars):
            assert original['name'] == imported['name']
            assert original['class'] == imported['class']
            assert original['level'] == imported['level']
    
    def test_character_backup_and_recovery(self, temp_directory):
        """Test character data backup and recovery operations."""
        server = MockCharacterServer()
        
        # Create characters to backup
        for i in range(20):
            char_data = {
                'name': f'BackupHero_{i}',
                'class': 'Guardian',
                'level': i + 1
            }
            server.create_character(char_data)
        
        original_chars = server.list_characters()
        
        # Create backup
        backup_file = os.path.join(temp_directory, 'character_backup.json')
        backup_data = {
            'backup_timestamp': '2023-01-01T12:00:00Z',
            'character_count': len(original_chars),
            'characters': original_chars
        }
        
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f)
        
        # Simulate data loss (clear server)
        server.characters.clear()
        server.next_id = 1
        assert len(server.list_characters()) == 0
        
        # Restore from backup
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)
        
        for char_data in backup_data['characters']:
            # Restore character (will get new ID but same data)
            restore_data = {k: v for k, v in char_data.items() if k != 'id'}
            server.create_character(restore_data)
        
        restored_chars = server.list_characters()
        
        # Verify recovery
        assert len(restored_chars) == len(original_chars)
        
        # Verify character data integrity
        original_names = {char['name'] for char in original_chars}
        restored_names = {char['name'] for char in restored_chars}
        assert original_names == restored_names


if __name__ == '__main__':
    # Run integration tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--durations=10'
    ])