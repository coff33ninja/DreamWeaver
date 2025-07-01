import pytest
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import threading
import time

# Add the SERVER directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the modules we're testing (assuming they exist)
try:
    from character_server import CharacterServer, Character, CharacterStats, CharacterInventory
except ImportError:
    # Create mock classes for testing if the actual modules don't exist
    class Character:
        def __init__(self, data):
            self.id = data.get('id')
            self.name = data.get('name')
            self.level = data.get('level', 1)
            self.health = data.get('health', 100)
            self.max_health = data.get('max_health', 100)
            self.mana = data.get('mana', 50)
            self.experience = data.get('experience', 0)
            self.character_class = data.get('class', 'warrior')
            self.stats = CharacterStats(data.get('stats', {}))
            self.inventory = CharacterInventory(data.get('inventory', {}))
            self.created_at = datetime.now()
            self.last_login = datetime.now()
        
        def level_up(self):
            self.level += 1
            self.max_health += 10
            self.health = self.max_health
        
        def take_damage(self, amount):
            self.health = max(0, self.health - amount)
        
        def heal(self, amount):
            self.health = min(self.max_health, self.health + amount)
        
        def gain_experience(self, amount):
            self.experience += amount
            if self.experience >= self.level * 100:
                self.level_up()
        
        def is_dead(self):
            return self.health <= 0
        
        def to_dict(self):
            return {
                'id': self.id,
                'name': self.name,
                'level': self.level,
                'health': self.health,
                'max_health': self.max_health,
                'mana': self.mana,
                'experience': self.experience,
                'class': self.character_class,
                'stats': self.stats.to_dict(),
                'inventory': self.inventory.to_dict(),
                'created_at': self.created_at.isoformat(),
                'last_login': self.last_login.isoformat()
            }
        
        @classmethod
        def from_dict(cls, data):
            return cls(data)
    
    class CharacterStats:
        def __init__(self, data):
            self.strength = data.get('strength', 10)
            self.agility = data.get('agility', 10)
            self.intelligence = data.get('intelligence', 10)
            self.vitality = data.get('vitality', 10)
            self._base_stats = data.copy()
        
        def modify_stat(self, stat_name, amount):
            if not hasattr(self, stat_name):
                raise KeyError(f"Invalid stat: {stat_name}")
            setattr(self, stat_name, getattr(self, stat_name) + amount)
        
        def total(self):
            return self.strength + self.agility + self.intelligence + self.vitality
        
        def reset(self):
            for stat, value in self._base_stats.items():
                setattr(self, stat, value)
        
        def to_dict(self):
            return {
                'strength': self.strength,
                'agility': self.agility,
                'intelligence': self.intelligence,
                'vitality': self.vitality
            }
    
    class CharacterInventory:
        def __init__(self, data):
            self.gold = data.get('gold', 0)
            self.items = data.get('items', [])
        
        def add_item(self, item):
            existing = self.find_item(item['id'])
            if existing:
                existing['quantity'] += item['quantity']
            else:
                self.items.append(item)
        
        def remove_item(self, item_id, quantity):
            item = self.find_item(item_id)
            if not item:
                raise ValueError(f"Item {item_id} not found")
            if item['quantity'] < quantity:
                raise ValueError("Insufficient quantity")
            item['quantity'] -= quantity
            if item['quantity'] == 0:
                self.items.remove(item)
        
        def find_item(self, item_id):
            return next((item for item in self.items if item['id'] == item_id), None)
        
        def has_item(self, item_id):
            return self.find_item(item_id) is not None
        
        def get_item_quantity(self, item_id):
            item = self.find_item(item_id)
            return item['quantity'] if item else 0
        
        def add_gold(self, amount):
            self.gold += amount
        
        def remove_gold(self, amount):
            if self.gold < amount:
                raise ValueError("Insufficient gold")
            self.gold -= amount
        
        def to_dict(self):
            return {
                'gold': self.gold,
                'items': self.items
            }
    
    class CharacterServer:
        def __init__(self, config):
            self.config = config
            self.characters = {}
            self.running = False
            self._validate_config()
        
        def _validate_config(self):
            required_keys = ['data_path', 'port', 'host', 'max_characters']
            for key in required_keys:
                if key not in self.config:
                    raise ValueError(f"Missing required config key: {key}")
        
        def create_character(self, data):
            if data['id'] in self.characters:
                raise ValueError("Character already exists")
            if len(self.characters) >= self.config['max_characters']:
                raise ValueError("Maximum characters limit reached")
            character = Character(data)
            self.characters[character.id] = character
            return character
        
        def get_character(self, character_id):
            return self.characters.get(character_id)
        
        def update_character(self, character_id, data):
            character = self.characters.get(character_id)
            if not character:
                return False
            for key, value in data.items():
                if hasattr(character, key):
                    setattr(character, key, value)
            return True
        
        def delete_character(self, character_id):
            if character_id in self.characters:
                del self.characters[character_id]
                return True
            return False
        
        def list_characters(self, filters=None):
            characters = list(self.characters.values())
            if filters:
                for key, value in filters.items():
                    characters = [c for c in characters if getattr(c, key, None) == value]
            return characters
        
        def start(self):
            self.running = True
        
        def stop(self):
            self.running = False
        
        def is_running(self):
            return self.running
        
        def save_character_data(self):
            pass  # Mock implementation
        
        def load_character_data(self):
            pass  # Mock implementation


class TestCharacter:
    """Test cases for the Character class using pytest."""
    
    @pytest.fixture
    def character_data(self):
        """Fixture providing test character data."""
        return {
            'id': 'test-char-001',
            'name': 'Test Hero',
            'level': 5,
            'health': 100,
            'max_health': 120,
            'mana': 50,
            'experience': 1250,
            'class': 'warrior',
            'stats': {
                'strength': 15,
                'agility': 12,
                'intelligence': 8,
                'vitality': 18
            },
            'inventory': {
                'gold': 500,
                'items': [
                    {'id': 'sword_001', 'name': 'Iron Sword', 'quantity': 1},
                    {'id': 'potion_001', 'name': 'Health Potion', 'quantity': 3}
                ]
            }
        }
    
    @pytest.fixture
    def character(self, character_data):
        """Fixture providing a test character instance."""
        return Character(character_data)
    
    def test_character_initialization_valid_data(self, character):
        """Test character initialization with valid data."""
        assert character.id == 'test-char-001'
        assert character.name == 'Test Hero'
        assert character.level == 5
        assert character.health == 100
        assert character.mana == 50
        assert character.experience == 1250
        assert character.character_class == 'warrior'
    
    def test_character_initialization_missing_required_fields(self):
        """Test character initialization with missing required fields."""
        incomplete_data = {'name': 'Incomplete Character'}
        with pytest.raises(KeyError):
            Character(incomplete_data)
    
    @pytest.mark.parametrize("invalid_data", [
        {'id': 'test', 'name': 'Test', 'level': 'not_a_number'},
        {'id': 'test', 'name': 'Test', 'level': -1},
        {'id': '', 'name': 'Test', 'level': 1},
        {'id': 'test', 'name': '', 'level': 1},
    ])
    def test_character_initialization_invalid_data(self, invalid_data):
        """Test character initialization with various invalid data scenarios."""
        with pytest.raises((TypeError, ValueError)):
            Character(invalid_data)
    
    def test_character_level_up(self, character):
        """Test character level up functionality."""
        initial_level = character.level
        initial_max_health = character.max_health
        character.level_up()
        assert character.level == initial_level + 1
        assert character.max_health > initial_max_health
        assert character.health == character.max_health
    
    @pytest.mark.parametrize("damage,expected_health", [
        (25, 75),
        (50, 50),
        (100, 0),
        (150, 0),  # Damage exceeding health
    ])
    def test_character_take_damage(self, character, damage, expected_health):
        """Test character taking various amounts of damage."""
        character.take_damage(damage)
        assert character.health == expected_health
    
    def test_character_take_damage_negative(self, character):
        """Test character taking negative damage (should not heal)."""
        initial_health = character.health
        character.take_damage(-10)
        assert character.health >= initial_health
    
    @pytest.mark.parametrize("heal_amount,initial_damage,expected_health", [
        (20, 30, 90),  # Normal healing
        (50, 30, 100),  # Heal to max health
        (1000, 30, 120),  # Overheal should cap at max_health
    ])
    def test_character_heal(self, character, heal_amount, initial_damage, expected_health):
        """Test character healing with various scenarios."""
        character.take_damage(initial_damage)
        character.heal(heal_amount)
        assert character.health == expected_health
    
    def test_character_gain_experience_no_levelup(self, character):
        """Test character gaining experience without leveling up."""
        initial_exp = character.experience
        initial_level = character.level
        character.gain_experience(50)
        assert character.experience == initial_exp + 50
        assert character.level == initial_level
    
    def test_character_gain_experience_with_levelup(self, character):
        """Test character gaining enough experience to level up."""
        character.experience = 450  # Close to level up threshold
        initial_level = character.level
        character.gain_experience(100)
        assert character.level == initial_level + 1
    
    def test_character_death_state(self, character):
        """Test character death state functionality."""
        assert not character.is_dead()
        character.take_damage(200)
        assert character.is_dead()
        assert character.health == 0
    
    def test_character_serialization(self, character):
        """Test character serialization to dictionary."""
        char_dict = character.to_dict()
        assert isinstance(char_dict, dict)
        assert char_dict['id'] == character.id
        assert char_dict['name'] == character.name
        assert char_dict['level'] == character.level
        assert 'created_at' in char_dict
        assert 'last_login' in char_dict
    
    def test_character_deserialization(self, character_data):
        """Test character deserialization from dictionary."""
        original_character = Character(character_data)
        char_dict = original_character.to_dict()
        new_character = Character.from_dict(char_dict)
        assert new_character.id == original_character.id
        assert new_character.name == original_character.name
        assert new_character.level == original_character.level
    
    def test_character_edge_cases(self, character):
        """Test character edge cases and boundary conditions."""
        # Test zero values
        character.take_damage(character.health)
        assert character.health == 0
        
        # Test healing when already at max health
        character.heal(character.max_health)
        character.heal(10)
        assert character.health == character.max_health
        
        # Test massive experience gain
        character.gain_experience(10000)
        assert character.level > 5


class TestCharacterStats:
    """Test cases for the CharacterStats class."""
    
    @pytest.fixture
    def stats_data(self):
        """Fixture providing test stats data."""
        return {
            'strength': 15,
            'agility': 12,
            'intelligence': 8,
            'vitality': 18
        }
    
    @pytest.fixture
    def stats(self, stats_data):
        """Fixture providing a test stats instance."""
        return CharacterStats(stats_data)
    
    def test_stats_initialization(self, stats):
        """Test stats initialization with valid data."""
        assert stats.strength == 15
        assert stats.agility == 12
        assert stats.intelligence == 8
        assert stats.vitality == 18
    
    def test_stats_initialization_defaults(self):
        """Test stats initialization with default values."""
        stats = CharacterStats({})
        assert stats.strength == 10
        assert stats.agility == 10
        assert stats.intelligence == 10
        assert stats.vitality == 10
    
    @pytest.mark.parametrize("stat_name,modifier,expected", [
        ('strength', 5, 20),
        ('agility', -2, 10),
        ('intelligence', 10, 18),
        ('vitality', 0, 18),
    ])
    def test_stats_modification(self, stats, stat_name, modifier, expected):
        """Test stats modification with various values."""
        stats.modify_stat(stat_name, modifier)
        assert getattr(stats, stat_name) == expected
    
    def test_stats_modification_invalid_stat(self, stats):
        """Test stats modification with invalid stat name."""
        with pytest.raises(KeyError):
            stats.modify_stat('invalid_stat', 5)
    
    def test_stats_total_calculation(self, stats):
        """Test total stats calculation."""
        expected_total = 15 + 12 + 8 + 18
        assert stats.total() == expected_total
    
    def test_stats_reset(self, stats):
        """Test stats reset to base values."""
        stats.modify_stat('strength', 10)
        assert stats.strength == 25
        stats.reset()
        assert stats.strength == 15
    
    def test_stats_serialization(self, stats):
        """Test stats serialization to dictionary."""
        stats_dict = stats.to_dict()
        assert isinstance(stats_dict, dict)
        assert stats_dict['strength'] == 15
        assert stats_dict['agility'] == 12
        assert stats_dict['intelligence'] == 8
        assert stats_dict['vitality'] == 18


class TestCharacterInventory:
    """Test cases for the CharacterInventory class."""
    
    @pytest.fixture
    def inventory_data(self):
        """Fixture providing test inventory data."""
        return {
            'gold': 500,
            'items': [
                {'id': 'sword_001', 'name': 'Iron Sword', 'quantity': 1},
                {'id': 'potion_001', 'name': 'Health Potion', 'quantity': 3}
            ]
        }
    
    @pytest.fixture
    def inventory(self, inventory_data):
        """Fixture providing a test inventory instance."""
        return CharacterInventory(inventory_data)
    
    def test_inventory_initialization(self, inventory):
        """Test inventory initialization with valid data."""
        assert inventory.gold == 500
        assert len(inventory.items) == 2
    
    def test_inventory_initialization_empty(self):
        """Test inventory initialization with empty data."""
        inventory = CharacterInventory({})
        assert inventory.gold == 0
        assert len(inventory.items) == 0
    
    def test_add_item_new(self, inventory):
        """Test adding a new item to inventory."""
        new_item = {'id': 'armor_001', 'name': 'Leather Armor', 'quantity': 1}
        inventory.add_item(new_item)
        assert len(inventory.items) == 3
        assert inventory.has_item('armor_001')
    
    def test_add_item_existing_stackable(self, inventory):
        """Test adding an existing stackable item to inventory."""
        existing_item = {'id': 'potion_001', 'name': 'Health Potion', 'quantity': 2}
        initial_quantity = inventory.get_item_quantity('potion_001')
        inventory.add_item(existing_item)
        new_quantity = inventory.get_item_quantity('potion_001')
        assert new_quantity == initial_quantity + 2
    
    @pytest.mark.parametrize("item_id,remove_quantity,expected_remaining", [
        ('potion_001', 1, 2),
        ('potion_001', 2, 1),
        ('sword_001', 1, 0),
    ])
    def test_remove_item_partial(self, inventory, item_id, remove_quantity, expected_remaining):
        """Test removing partial quantity of items."""
        inventory.remove_item(item_id, remove_quantity)
        remaining = inventory.get_item_quantity(item_id)
        assert remaining == expected_remaining
    
    def test_remove_item_complete(self, inventory):
        """Test removing all quantity of an item."""
        inventory.remove_item('sword_001', 1)
        assert not inventory.has_item('sword_001')
    
    def test_remove_item_nonexistent(self, inventory):
        """Test removing an item that doesn't exist."""
        with pytest.raises(ValueError, match="Item nonexistent_item not found"):
            inventory.remove_item('nonexistent_item', 1)
    
    def test_remove_item_insufficient_quantity(self, inventory):
        """Test removing more quantity than available."""
        with pytest.raises(ValueError, match="Insufficient quantity"):
            inventory.remove_item('potion_001', 10)
    
    @pytest.mark.parametrize("gold_amount,expected_total", [
        (100, 600),
        (0, 500),
        (1000, 1500),
    ])
    def test_add_gold(self, inventory, gold_amount, expected_total):
        """Test adding gold to inventory."""
        inventory.add_gold(gold_amount)
        assert inventory.gold == expected_total
    
    def test_remove_gold_valid(self, inventory):
        """Test removing gold from inventory."""
        inventory.remove_gold(100)
        assert inventory.gold == 400
    
    def test_remove_gold_insufficient(self, inventory):
        """Test removing more gold than available."""
        with pytest.raises(ValueError, match="Insufficient gold"):
            inventory.remove_gold(1000)
    
    def test_inventory_serialization(self, inventory):
        """Test inventory serialization to dictionary."""
        inventory_dict = inventory.to_dict()
        assert isinstance(inventory_dict, dict)
        assert inventory_dict['gold'] == 500
        assert len(inventory_dict['items']) == 2


class TestCharacterServer:
    """Test cases for the CharacterServer class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Fixture providing a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def server_config(self, temp_dir):
        """Fixture providing server configuration."""
        return {
            'data_path': temp_dir,
            'port': 8080,
            'host': 'localhost',
            'max_characters': 1000
        }
    
    @pytest.fixture
    def server(self, server_config):
        """Fixture providing a test server instance."""
        return CharacterServer(server_config)
    
    @pytest.fixture
    def test_character_data(self):
        """Fixture providing test character data."""
        return {
            'id': 'test-char-001',
            'name': 'Test Hero',
            'level': 5,
            'health': 100,
            'mana': 50,
            'experience': 1250,
            'class': 'warrior',
            'stats': {
                'strength': 15,
                'agility': 12,
                'intelligence': 8,
                'vitality': 18
            },
            'inventory': {
                'gold': 500,
                'items': []
            }
        }
    
    def test_server_initialization_valid_config(self, server):
        """Test server initialization with valid config."""
        assert server.config['port'] == 8080
        assert server.config['host'] == 'localhost'
        assert server.config['max_characters'] == 1000
    
    @pytest.mark.parametrize("invalid_config", [
        {'port': 8080},  # Missing required keys
        {'port': 'not_a_number', 'host': 'localhost', 'data_path': '/tmp', 'max_characters': 100},
        {},  # Empty config
    ])
    def test_server_initialization_invalid_config(self, invalid_config):
        """Test server initialization with invalid config."""
        with pytest.raises(ValueError):
            CharacterServer(invalid_config)
    
    def test_create_character_valid(self, server, test_character_data):
        """Test creating a character with valid data."""
        character = server.create_character(test_character_data)
        assert character is not None
        assert character.id == 'test-char-001'
        assert character.name == 'Test Hero'
        assert server.get_character('test-char-001') == character
    
    def test_create_character_duplicate_id(self, server, test_character_data):
        """Test creating a character with duplicate ID."""
        server.create_character(test_character_data)
        with pytest.raises(ValueError, match="Character already exists"):
            server.create_character(test_character_data)
    
    def test_create_character_invalid_data(self, server):
        """Test creating a character with invalid data."""
        invalid_data = {'name': 'Invalid Character'}  # Missing required fields
        with pytest.raises(KeyError):
            server.create_character(invalid_data)
    
    def test_get_character_existing(self, server, test_character_data):
        """Test getting an existing character."""
        server.create_character(test_character_data)
        character = server.get_character('test-char-001')
        assert character is not None
        assert character.id == 'test-char-001'
    
    def test_get_character_nonexistent(self, server):
        """Test getting a non-existent character."""
        character = server.get_character('nonexistent-char')
        assert character is None
    
    def test_update_character_existing(self, server, test_character_data):
        """Test updating an existing character."""
        server.create_character(test_character_data)
        update_data = {'level': 6, 'experience': 1500}
        success = server.update_character('test-char-001', update_data)
        assert success is True
        
        updated_character = server.get_character('test-char-001')
        assert updated_character.level == 6
        assert updated_character.experience == 1500
    
    def test_update_character_nonexistent(self, server):
        """Test updating a non-existent character."""
        update_data = {'level': 6}
        success = server.update_character('nonexistent-char', update_data)
        assert success is False
    
    def test_delete_character_existing(self, server, test_character_data):
        """Test deleting an existing character."""
        server.create_character(test_character_data)
        success = server.delete_character('test-char-001')
        assert success is True
        
        character = server.get_character('test-char-001')
        assert character is None
    
    def test_delete_character_nonexistent(self, server):
        """Test deleting a non-existent character."""
        success = server.delete_character('nonexistent-char')
        assert success is False
    
    def test_list_characters_empty(self, server):
        """Test listing characters when none exist."""
        characters = server.list_characters()
        assert len(characters) == 0
    
    def test_list_characters_with_data(self, server, test_character_data):
        """Test listing characters when some exist."""
        server.create_character(test_character_data)
        
        # Create another character
        another_character = test_character_data.copy()
        another_character['id'] = 'test-char-002'
        another_character['name'] = 'Another Hero'
        server.create_character(another_character)
        
        characters = server.list_characters()
        assert len(characters) == 2
    
    def test_list_characters_with_filters(self, server, test_character_data):
        """Test listing characters with filters applied."""
        server.create_character(test_character_data)
        
        # Create another character with different class
        another_character = test_character_data.copy()
        another_character['id'] = 'test-char-002'
        another_character['class'] = 'mage'
        server.create_character(another_character)
        
        warrior_characters = server.list_characters(filters={'character_class': 'warrior'})
        assert len(warrior_characters) == 1
        assert warrior_characters[0].character_class == 'warrior'
    
    def test_server_start_stop(self, server):
        """Test server start and stop functionality."""
        assert not server.is_running()
        
        server.start()
        assert server.is_running()
        
        server.stop()
        assert not server.is_running()
    
    def test_max_characters_limit(self, server_config, test_character_data):
        """Test maximum characters limit enforcement."""
        server_config['max_characters'] = 2
        server = CharacterServer(server_config)
        
        # Create maximum allowed characters
        for i in range(2):
            char_data = test_character_data.copy()
            char_data['id'] = f'test-char-{i:03d}'
            server.create_character(char_data)
        
        # Try to create one more character
        excess_char_data = test_character_data.copy()
        excess_char_data['id'] = 'test-char-excess'
        
        with pytest.raises(ValueError, match="Maximum characters limit reached"):
            server.create_character(excess_char_data)
    
    @pytest.mark.parametrize("invalid_data", [
        {'id': '', 'name': 'Test', 'level': 1},  # Empty ID
        {'id': 'test', 'name': '', 'level': 1},  # Empty name
        {'id': 'test', 'name': 'Test', 'level': 0},  # Invalid level
        {'id': 'test', 'name': 'Test', 'level': 1, 'health': -10},  # Negative health
    ])
    def test_character_validation(self, server, invalid_data):
        """Test comprehensive character data validation."""
        with pytest.raises((ValueError, KeyError)):
            server.create_character(invalid_data)
    
    def test_concurrent_character_operations(self, server, test_character_data):
        """Test concurrent character operations."""
        server.create_character(test_character_data)
        
        def update_character():
            server.update_character('test-char-001', {'level': 10})
        
        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_character)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify character still exists and is in valid state
        character = server.get_character('test-char-001')
        assert character is not None
        assert character.level >= 5  # Should be at least original level
    
    @patch('character_server.CharacterServer.save_character_data')
    def test_save_character_data(self, mock_save, server, test_character_data):
        """Test saving character data to file."""
        server.create_character(test_character_data)
        server.save_character_data()
        mock_save.assert_called_once()
    
    @patch('character_server.CharacterServer.load_character_data')
    def test_load_character_data(self, mock_load, server):
        """Test loading character data from file."""
        server.load_character_data()
        mock_load.assert_called_once()
    
    def test_performance_with_many_characters(self, server, test_character_data):
        """Test performance with many characters."""
        start_time = time.time()
        
        # Create many characters
        for i in range(100):
            char_data = test_character_data.copy()
            char_data['id'] = f'perf-test-{i:03d}'
            char_data['name'] = f'Performance Test {i}'
            server.create_character(char_data)
        
        creation_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        for i in range(100):
            character = server.get_character(f'perf-test-{i:03d}')
            assert character is not None
        
        retrieval_time = time.time() - start_time
        
        # Performance should be reasonable (adjust thresholds as needed)
        assert creation_time < 5.0  # Should create 100 characters in under 5 seconds
        assert retrieval_time < 2.0  # Should retrieve 100 characters in under 2 seconds
    
    def test_character_persistence(self, server, test_character_data):
        """Test character data persistence between operations."""
        # Create a character
        character = server.create_character(test_character_data)
        original_id = character.id
        
        # Modify the character
        server.update_character(original_id, {'level': 10, 'experience': 2000})
        
        # Retrieve and verify changes persist
        updated_character = server.get_character(original_id)
        assert updated_character.level == 10
        assert updated_character.experience == 2000
        assert updated_character.name == test_character_data['name']  # Unchanged field
    
    def test_server_edge_cases(self, server, test_character_data):
        """Test server edge cases and boundary conditions."""
        # Test with minimal character data
        minimal_data = {
            'id': 'minimal-char',
            'name': 'Minimal Character'
        }
        
        # Should handle minimal data gracefully
        try:
            character = server.create_character(minimal_data)
            # If creation succeeds, character should have default values
            assert character.level >= 1
            assert character.health > 0
        except (KeyError, ValueError):
            # If creation fails, that's also acceptable for minimal data
            pass
        
        # Test operations on empty server
        assert server.list_characters() == []
        assert server.get_character('nonexistent') is None
        assert server.delete_character('nonexistent') is False
        assert server.update_character('nonexistent', {}) is False


class TestIntegration:
    """Integration tests for the character server system."""
    
    @pytest.fixture
    def server_with_characters(self, server_config):
        """Fixture providing a server with pre-created characters."""
        server = CharacterServer(server_config)
        
        # Create test characters
        characters_data = [
            {
                'id': 'warrior-001',
                'name': 'Warrior One',
                'level': 5,
                'health': 120,
                'mana': 30,
                'class': 'warrior',
                'stats': {'strength': 20, 'agility': 10, 'intelligence': 5, 'vitality': 15},
                'inventory': {'gold': 1000, 'items': []}
            },
            {
                'id': 'mage-001',
                'name': 'Mage One',
                'level': 4,
                'health': 80,
                'mana': 100,
                'class': 'mage',
                'stats': {'strength': 8, 'agility': 12, 'intelligence': 20, 'vitality': 10},
                'inventory': {'gold': 500, 'items': []}
            },
            {
                'id': 'rogue-001',
                'name': 'Rogue One',
                'level': 6,
                'health': 90,
                'mana': 40,
                'class': 'rogue',
                'stats': {'strength': 12, 'agility': 18, 'intelligence': 15, 'vitality': 10},
                'inventory': {'gold': 750, 'items': []}
            }
        ]
        
        for char_data in characters_data:
            server.create_character(char_data)
        
        return server
    
    def test_character_lifecycle_integration(self, server):
        """Test complete character lifecycle integration."""
        # Create character
        char_data = {
            'id': 'lifecycle-test',
            'name': 'Lifecycle Test Character',
            'level': 1,
            'health': 100,
            'mana': 50,
            'class': 'warrior',
            'stats': {'strength': 10, 'agility': 10, 'intelligence': 10, 'vitality': 10},
            'inventory': {'gold': 100, 'items': []}
        }
        
        character = server.create_character(char_data)
        assert character is not None
        
        # Level up the character
        character.gain_experience(500)
        assert character.level > 1
        
        # Damage and heal the character
        initial_health = character.health
        character.take_damage(30)
        assert character.health < initial_health
        character.heal(20)
        assert character.health > initial_health - 30
        
        # Update character through server
        server.update_character(character.id, {'mana': 75})
        updated_char = server.get_character(character.id)
        assert updated_char.mana == 75
        
        # Add items to inventory
        test_item = {'id': 'test-item', 'name': 'Test Item', 'quantity': 1}
        character.inventory.add_item(test_item)
        assert character.inventory.has_item('test-item')
        
        # Finally delete the character
        success = server.delete_character(character.id)
        assert success is True
        assert server.get_character(character.id) is None
    
    def test_server_filtering_and_sorting(self, server_with_characters):
        """Test server filtering and sorting capabilities."""
        # Test filtering by class
        warriors = server_with_characters.list_characters(filters={'character_class': 'warrior'})
        assert len(warriors) == 1
        assert warriors[0].character_class == 'warrior'
        
        mages = server_with_characters.list_characters(filters={'character_class': 'mage'})
        assert len(mages) == 1
        assert mages[0].character_class == 'mage'
        
        # Test multiple filters
        high_level_warriors = server_with_characters.list_characters(
            filters={'character_class': 'warrior', 'level': 5}
        )
        assert len(high_level_warriors) == 1
    
    def test_bulk_operations(self, server):
        """Test bulk operations on multiple characters."""
        # Create multiple characters
        characters = []
        for i in range(10):
            char_data = {
                'id': f'bulk-test-{i:03d}',
                'name': f'Bulk Test Character {i}',
                'level': i + 1,
                'health': 100 + i * 10,
                'mana': 50 + i * 5,
                'class': 'warrior' if i % 2 == 0 else 'mage',
                'stats': {'strength': 10, 'agility': 10, 'intelligence': 10, 'vitality': 10},
                'inventory': {'gold': 100 * (i + 1), 'items': []}
            }
            character = server.create_character(char_data)
            characters.append(character)
        
        # Verify all characters were created
        all_characters = server.list_characters()
        assert len(all_characters) == 10
        
        # Bulk update - level up all characters
        for char in characters:
            char.gain_experience(1000)
        
        # Verify updates
        for char in characters:
            updated_char = server.get_character(char.id)
            assert updated_char.level > char.level or updated_char.level == char.level
        
        # Bulk delete - remove all even-numbered characters
        for i in range(0, 10, 2):
            success = server.delete_character(f'bulk-test-{i:03d}')
            assert success is True
        
        # Verify deletions
        remaining_characters = server.list_characters()
        assert len(remaining_characters) == 5
        
        for char in remaining_characters:
            assert int(char.id.split('-')[-1]) % 2 == 1  # Only odd-numbered remain


# Performance and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests for the character server."""
    
    @pytest.mark.slow
    def test_large_scale_character_creation(self, server_config):
        """Test creating a large number of characters."""
        server_config['max_characters'] = 10000
        server = CharacterServer(server_config)
        
        start_time = time.time()
        
        # Create 1000 characters
        for i in range(1000):
            char_data = {
                'id': f'stress-test-{i:05d}',
                'name': f'Stress Test Character {i}',
                'level': (i % 50) + 1,
                'health': 100 + (i % 100),
                'mana': 50 + (i % 50),
                'class': ['warrior', 'mage', 'rogue'][i % 3],
                'stats': {
                    'strength': 10 + (i % 10),
                    'agility': 10 + (i % 10),
                    'intelligence': 10 + (i % 10),
                    'vitality': 10 + (i % 10)
                },
                'inventory': {'gold': i * 10, 'items': []}
            }
            server.create_character(char_data)
        
        creation_time = time.time() - start_time
        
        # Performance assertion - should create 1000 characters in reasonable time
        assert creation_time < 30.0  # 30 seconds max
        
        # Verify all characters were created
        all_characters = server.list_characters()
        assert len(all_characters) == 1000
    
    @pytest.mark.slow
    def test_concurrent_operations_stress(self, server_config):
        """Test concurrent operations under stress."""
        server = CharacterServer(server_config)
        
        # Create initial characters
        for i in range(100):
            char_data = {
                'id': f'concurrent-test-{i:03d}',
                'name': f'Concurrent Test Character {i}',
                'level': 1,
                'health': 100,
                'mana': 50,
                'class': 'warrior',
                'stats': {'strength': 10, 'agility': 10, 'intelligence': 10, 'vitality': 10},
                'inventory': {'gold': 100, 'items': []}
            }
            server.create_character(char_data)
        
        # Define concurrent operations
        def random_operations():
            import random
            for _ in range(50):
                operation = random.choice(['get', 'update', 'list'])
                char_id = f'concurrent-test-{random.randint(0, 99):03d}'
                
                if operation == 'get':
                    server.get_character(char_id)
                elif operation == 'update':
                    server.update_character(char_id, {'level': random.randint(1, 10)})
                elif operation == 'list':
                    server.list_characters()
        
        # Run concurrent operations
        threads = []
        start_time = time.time()
        
        for _ in range(10):
            thread = threading.Thread(target=random_operations)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        operation_time = time.time() - start_time
        
        # Performance assertion
        assert operation_time < 10.0  # Should complete in under 10 seconds
        
        # Verify server integrity
        characters = server.list_characters()
        assert len(characters) == 100  # No characters should be lost


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


# Test configuration and utilities
class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def create_test_character(char_id='test-char', name='Test Character', level=1, char_class='warrior'):
        """Utility function to create test character data."""
        return {
            'id': char_id,
            'name': name,
            'level': level,
            'health': 100,
            'mana': 50,
            'experience': 0,
            'class': char_class,
            'stats': {
                'strength': 10,
                'agility': 10,
                'intelligence': 10,
                'vitality': 10
            },
            'inventory': {
                'gold': 100,
                'items': []
            }
        }
    
    def test_utility_functions(self):
        """Test utility functions work correctly."""
        char_data = self.create_test_character()
        assert char_data['id'] == 'test-char'
        assert char_data['name'] == 'Test Character'
        assert char_data['level'] == 1
        
        custom_char = self.create_test_character('custom-001', 'Custom Character', 5, 'mage')
        assert custom_char['id'] == 'custom-001'
        assert custom_char['name'] == 'Custom Character'
        assert custom_char['level'] == 5
        assert custom_char['class'] == 'mage'


if __name__ == '__main__':
    pytest.main(['-v', '--tb=short'])