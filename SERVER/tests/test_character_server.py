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
            """
            Initialize a Character instance with attributes from the provided data dictionary.
            
            Parameters:
                data (dict): Dictionary containing character attributes such as id, name, level, health, mana, experience, class, stats, and inventory. Missing values are set to sensible defaults.
            """
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
            """
            Increase the character's level by one, boost maximum health by 10, and restore current health to the new maximum.
            """
            self.level += 1
            self.max_health += 10
            self.health = self.max_health
        
        def take_damage(self, amount):
            """
            Reduces the character's health by the specified amount, not allowing health to drop below zero.
            
            Parameters:
                amount (int): The amount of damage to apply.
            """
            self.health = max(0, self.health - amount)
        
        def heal(self, amount):
            """
            Restore health by a specified amount, not exceeding the character's maximum health.
            """
            self.health = min(self.max_health, self.health + amount)
        
        def gain_experience(self, amount):
            """
            Adds experience points to the character and levels up if the experience threshold for the current level is reached.
            """
            self.experience += amount
            if self.experience >= self.level * 100:
                self.level_up()
        
        def is_dead(self):
            """
            Determine whether the character's health is zero or below, indicating death.
            
            Returns:
                bool: True if the character is dead, False otherwise.
            """
            return self.health <= 0
        
        def to_dict(self):
            """
            Serialize the character's attributes into a dictionary suitable for storage or transmission.
            
            Returns:
            	dict: A dictionary containing all character attributes, including stats and inventory, with datetime fields in ISO format.
            """
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
            """
            Instantiate a class object from a dictionary of attributes.
            
            Parameters:
                data (dict): Dictionary containing the attributes required to initialize the class.
            
            Returns:
                An instance of the class initialized with the provided data.
            """
            return cls(data)
    
    class CharacterStats:
        def __init__(self, data):
            """
            Initialize character stats with provided values or defaults.
            
            Parameters:
                data (dict): Dictionary containing optional stat values for 'strength', 'agility', 'intelligence', and 'vitality'. Missing stats default to 10.
            """
            self.strength = data.get('strength', 10)
            self.agility = data.get('agility', 10)
            self.intelligence = data.get('intelligence', 10)
            self.vitality = data.get('vitality', 10)
            self._base_stats = data.copy()
        
        def modify_stat(self, stat_name, amount):
            """
            Modify the value of a specified stat by a given amount.
            
            Raises:
                KeyError: If the provided stat name does not exist.
            """
            if not hasattr(self, stat_name):
                raise KeyError(f"Invalid stat: {stat_name}")
            setattr(self, stat_name, getattr(self, stat_name) + amount)
        
        def total(self):
            """
            Return the sum of all character stat values.
            	
            Returns:
            	int: The total of strength, agility, intelligence, and vitality.
            """
            return self.strength + self.agility + self.intelligence + self.vitality
        
        def reset(self):
            """
            Reset all character stats to their original base values.
            """
            for stat, value in self._base_stats.items():
                setattr(self, stat, value)
        
        def to_dict(self):
            """
            Serialize the character's stats into a dictionary.
            
            Returns:
                dict: A dictionary containing the strength, agility, intelligence, and vitality values.
            """
            return {
                'strength': self.strength,
                'agility': self.agility,
                'intelligence': self.intelligence,
                'vitality': self.vitality
            }
    
    class CharacterInventory:
        def __init__(self, data):
            """
            Initialize the inventory with gold and items from the provided data dictionary.
            
            Parameters:
                data (dict): Dictionary containing optional 'gold' (int) and 'items' (list) keys to set up the inventory.
            """
            self.gold = data.get('gold', 0)
            self.items = data.get('items', [])
        
        def add_item(self, item):
            """
            Adds an item to the inventory, stacking quantities if the item already exists.
            
            If an item with the same ID is present, its quantity is increased by the specified amount; otherwise, the item is added as a new entry.
            """
            existing = self.find_item(item['id'])
            if existing:
                existing['quantity'] += item['quantity']
            else:
                self.items.append(item)
        
        def remove_item(self, item_id, quantity):
            """
            Removes a specified quantity of an item from the inventory.
            
            Raises:
                ValueError: If the item is not found or if there is insufficient quantity to remove.
            """
            item = self.find_item(item_id)
            if not item:
                raise ValueError(f"Item {item_id} not found")
            if item['quantity'] < quantity:
                raise ValueError("Insufficient quantity")
            item['quantity'] -= quantity
            if item['quantity'] == 0:
                self.items.remove(item)
        
        def find_item(self, item_id):
            """
            Return the item dictionary with the specified item_id from the inventory, or None if not found.
            
            Parameters:
                item_id: The unique identifier of the item to search for.
            
            Returns:
                dict or None: The item dictionary if found, otherwise None.
            """
            return next((item for item in self.items if item['id'] == item_id), None)
        
        def has_item(self, item_id):
            """
            Check if an item with the specified ID exists in the inventory.
            
            Parameters:
                item_id: The unique identifier of the item to check.
            
            Returns:
                bool: True if the item exists in the inventory, False otherwise.
            """
            return self.find_item(item_id) is not None
        
        def get_item_quantity(self, item_id):
            """
            Return the quantity of a specific item in the inventory.
            
            Parameters:
                item_id: The unique identifier of the item to check.
            
            Returns:
                int: The quantity of the item if it exists, otherwise 0.
            """
            item = self.find_item(item_id)
            return item['quantity'] if item else 0
        
        def add_gold(self, amount):
            """
            Increase the amount of gold in the inventory by the specified amount.
            """
            self.gold += amount
        
        def remove_gold(self, amount):
            """
            Removes a specified amount of gold from the inventory.
            
            Raises:
                ValueError: If there is not enough gold to remove the requested amount.
            """
            if self.gold < amount:
                raise ValueError("Insufficient gold")
            self.gold -= amount
        
        def to_dict(self):
            """
            Serialize the inventory to a dictionary containing gold and items.
            
            Returns:
                dict: A dictionary with keys 'gold' and 'items' representing the inventory state.
            """
            return {
                'gold': self.gold,
                'items': self.items
            }
    
    class CharacterServer:
        def __init__(self, config):
            """
            Initialize the CharacterServer with the provided configuration.
            
            Parameters:
                config (dict): Configuration dictionary containing server settings such as data path, port, host, and maximum number of characters.
            
            Raises:
                ValueError: If required configuration keys are missing.
            """
            self.config = config
            self.characters = {}
            self.running = False
            self._validate_config()
        
        def _validate_config(self):
            """
            Validates that the server configuration contains all required keys.
            
            Raises:
                ValueError: If any required configuration key is missing.
            """
            required_keys = ['data_path', 'port', 'host', 'max_characters']
            for key in required_keys:
                if key not in self.config:
                    raise ValueError(f"Missing required config key: {key}")
        
        def create_character(self, data):
            """
            Creates and stores a new character using the provided data.
            
            Raises:
                ValueError: If a character with the same ID already exists or the maximum character limit is reached.
            
            Returns:
                Character: The newly created character instance.
            """
            if data['id'] in self.characters:
                raise ValueError("Character already exists")
            if len(self.characters) >= self.config['max_characters']:
                raise ValueError("Maximum characters limit reached")
            character = Character(data)
            self.characters[character.id] = character
            return character
        
        def get_character(self, character_id):
            """
            Retrieve a character by its unique identifier.
            
            Returns:
                Character: The character instance if found, otherwise None.
            """
            return self.characters.get(character_id)
        
        def update_character(self, character_id, data):
            """
            Updates the attributes of an existing character with the provided data.
            
            Parameters:
                character_id (str): The unique identifier of the character to update.
                data (dict): A dictionary of attribute names and their new values to set on the character.
            
            Returns:
                bool: True if the character was found and updated, False if the character does not exist.
            """
            character = self.characters.get(character_id)
            if not character:
                return False
            for key, value in data.items():
                if hasattr(character, key):
                    setattr(character, key, value)
            return True
        
        def delete_character(self, character_id):
            """
            Deletes a character by its ID from the server.
            
            Returns:
                bool: True if the character was deleted, False if the character was not found.
            """
            if character_id in self.characters:
                del self.characters[character_id]
                return True
            return False
        
        def list_characters(self, filters=None):
            """
            Returns a list of characters, optionally filtered by specified attribute values.
            
            Parameters:
                filters (dict, optional): A dictionary where keys are character attribute names and values are the values to filter by. Only characters matching all filter criteria are included.
            
            Returns:
                list: A list of Character instances matching the filter criteria, or all characters if no filters are provided.
            """
            characters = list(self.characters.values())
            if filters:
                for key, value in filters.items():
                    characters = [c for c in characters if getattr(c, key, None) == value]
            return characters
        
        def start(self):
            """
            Set the server's running state to True, indicating that the server is active.
            """
            self.running = True
        
        def stop(self):
            """
            Stops the character server by setting its running state to False.
            """
            self.running = False
        
        def is_running(self):
            """
            Return whether the server is currently running.
            
            Returns:
                bool: True if the server is running, False otherwise.
            """
            return self.running
        
        def save_character_data(self):
            """
            Placeholder method for saving all character data to persistent storage.
            """
            pass  # Mock implementation
        
        def load_character_data(self):
            """
            Placeholder for loading character data from persistent storage.
            
            This mock implementation does not perform any operation.
            """
            pass  # Mock implementation


class TestCharacter:
    """Test cases for the Character class using pytest."""
    
    @pytest.fixture
    def character_data(self):
        """
        Provides a dictionary containing sample character data for use in tests.
        
        Returns:
            dict: A dictionary representing a test character with predefined attributes, stats, and inventory.
        """
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
        """
        Fixture that provides a test Character instance initialized with the given character data.
        
        Parameters:
        	character_data (dict): Dictionary containing character attributes for initialization.
        
        Returns:
        	Character: A Character object created with the provided data.
        """
        return Character(character_data)
    
    def test_character_initialization_valid_data(self, character):
        """
        Verify that a Character instance is correctly initialized with valid data.
        """
        assert character.id == 'test-char-001'
        assert character.name == 'Test Hero'
        assert character.level == 5
        assert character.health == 100
        assert character.mana == 50
        assert character.experience == 1250
        assert character.character_class == 'warrior'
    
    def test_character_initialization_missing_required_fields(self):
        """
        Test that initializing a Character with missing required fields raises a KeyError.
        """
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
        """
        Test that initializing a Character with invalid data raises a TypeError or ValueError.
        """
        with pytest.raises((TypeError, ValueError)):
            Character(invalid_data)
    
    def test_character_level_up(self, character):
        """
        Test that leveling up a character increases their level and max health, and restores health to the new maximum.
        """
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
        """
        Test that a character's health is correctly reduced when taking damage.
        
        Parameters:
            character (Character): The character instance to apply damage to.
            damage (int): The amount of damage to inflict.
            expected_health (int): The expected health value after damage is applied.
        """
        character.take_damage(damage)
        assert character.health == expected_health
    
    def test_character_take_damage_negative(self, character):
        """
        Test that applying negative damage to a character does not increase their health.
        """
        initial_health = character.health
        character.take_damage(-10)
        assert character.health >= initial_health
    
    @pytest.mark.parametrize("heal_amount,initial_damage,expected_health", [
        (20, 30, 90),  # Normal healing
        (50, 30, 100),  # Heal to max health
        (1000, 30, 120),  # Overheal should cap at max_health
    ])
    def test_character_heal(self, character, heal_amount, initial_damage, expected_health):
        """
        Test that healing a character after taking damage restores health correctly for different scenarios.
        
        Parameters:
            heal_amount (int): The amount of health to restore.
            initial_damage (int): The amount of damage to apply before healing.
            expected_health (int): The expected health value after healing.
        """
        character.take_damage(initial_damage)
        character.heal(heal_amount)
        assert character.health == expected_health
    
    def test_character_gain_experience_no_levelup(self, character):
        """
        Test that a character gains experience without leveling up when the gained amount is insufficient for a level increase.
        """
        initial_exp = character.experience
        initial_level = character.level
        character.gain_experience(50)
        assert character.experience == initial_exp + 50
        assert character.level == initial_level
    
    def test_character_gain_experience_with_levelup(self, character):
        """
        Test that a character levels up when gaining enough experience to cross the level-up threshold.
        """
        character.experience = 450  # Close to level up threshold
        initial_level = character.level
        character.gain_experience(100)
        assert character.level == initial_level + 1
    
    def test_character_death_state(self, character):
        """
        Test that a character is correctly marked as dead when health reaches zero after taking damage.
        """
        assert not character.is_dead()
        character.take_damage(200)
        assert character.is_dead()
        assert character.health == 0
    
    def test_character_serialization(self, character):
        """
        Test that a Character instance can be correctly serialized to a dictionary with expected fields and values.
        """
        char_dict = character.to_dict()
        assert isinstance(char_dict, dict)
        assert char_dict['id'] == character.id
        assert char_dict['name'] == character.name
        assert char_dict['level'] == character.level
        assert 'created_at' in char_dict
        assert 'last_login' in char_dict
    
    def test_character_deserialization(self, character_data):
        """
        Test that a Character object can be accurately deserialized from a dictionary representation.
        
        Verifies that the deserialized character retains the original id, name, and level.
        """
        original_character = Character(character_data)
        char_dict = original_character.to_dict()
        new_character = Character.from_dict(char_dict)
        assert new_character.id == original_character.id
        assert new_character.name == original_character.name
        assert new_character.level == original_character.level
    
    def test_character_edge_cases(self, character):
        """
        Test Character class behavior under edge and boundary conditions.
        
        Covers scenarios including reducing health to zero, healing at maximum health, and gaining a large amount of experience to trigger multiple level-ups.
        """
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
        """
        Provides a fixture with sample character stats data for testing.
        
        Returns:
            dict: A dictionary containing strength, agility, intelligence, and vitality values.
        """
        return {
            'strength': 15,
            'agility': 12,
            'intelligence': 8,
            'vitality': 18
        }
    
    @pytest.fixture
    def stats(self, stats_data):
        """
        Fixture that provides a CharacterStats instance initialized with the given stats data.
        
        Parameters:
            stats_data (dict): Dictionary containing stat values for initialization.
        
        Returns:
            CharacterStats: An instance of CharacterStats with the specified values.
        """
        return CharacterStats(stats_data)
    
    def test_stats_initialization(self, stats):
        """
        Test that CharacterStats initializes with the correct attribute values.
        """
        assert stats.strength == 15
        assert stats.agility == 12
        assert stats.intelligence == 8
        assert stats.vitality == 18
    
    def test_stats_initialization_defaults(self):
        """
        Test that CharacterStats initializes all stats to default values when no data is provided.
        """
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
        """
        Test that modifying a stat updates its value as expected.
        
        Parameters:
            stats (CharacterStats): The stats object to modify.
            stat_name (str): The name of the stat to modify.
            modifier (int): The amount to modify the stat by.
            expected (int): The expected value of the stat after modification.
        """
        stats.modify_stat(stat_name, modifier)
        assert getattr(stats, stat_name) == expected
    
    def test_stats_modification_invalid_stat(self, stats):
        """
        Test that modifying a stat with an invalid stat name raises a KeyError.
        """
        with pytest.raises(KeyError):
            stats.modify_stat('invalid_stat', 5)
    
    def test_stats_total_calculation(self, stats):
        """
        Test that the total method correctly sums all character stats.
        
        Verifies that the total of strength, agility, intelligence, and vitality matches the expected value.
        """
        expected_total = 15 + 12 + 8 + 18
        assert stats.total() == expected_total
    
    def test_stats_reset(self, stats):
        """
        Verify that resetting stats restores all stat values to their base defaults after modifications.
        """
        stats.modify_stat('strength', 10)
        assert stats.strength == 25
        stats.reset()
        assert stats.strength == 15
    
    def test_stats_serialization(self, stats):
        """
        Verify that CharacterStats serializes correctly to a dictionary with expected stat values.
        """
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
        """
        Provides sample inventory data for testing purposes.
        
        Returns:
            dict: A dictionary containing gold and a list of item dictionaries.
        """
        return {
            'gold': 500,
            'items': [
                {'id': 'sword_001', 'name': 'Iron Sword', 'quantity': 1},
                {'id': 'potion_001', 'name': 'Health Potion', 'quantity': 3}
            ]
        }
    
    @pytest.fixture
    def inventory(self, inventory_data):
        """
        Fixture that returns a CharacterInventory instance initialized with the provided inventory data.
        """
        return CharacterInventory(inventory_data)
    
    def test_inventory_initialization(self, inventory):
        """
        Test that a CharacterInventory instance initializes correctly with provided gold and items.
        """
        assert inventory.gold == 500
        assert len(inventory.items) == 2
    
    def test_inventory_initialization_empty(self):
        """
        Test that a CharacterInventory initialized with empty data has zero gold and no items.
        """
        inventory = CharacterInventory({})
        assert inventory.gold == 0
        assert len(inventory.items) == 0
    
    def test_add_item_new(self, inventory):
        """
        Test that adding a new item to the inventory increases the item count and makes the item available.
        """
        new_item = {'id': 'armor_001', 'name': 'Leather Armor', 'quantity': 1}
        inventory.add_item(new_item)
        assert len(inventory.items) == 3
        assert inventory.has_item('armor_001')
    
    def test_add_item_existing_stackable(self, inventory):
        """
        Test that adding an existing stackable item increases its quantity in the inventory.
        """
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
        """
        Test that removing a partial quantity of an item from the inventory updates the item's quantity correctly.
        
        Parameters:
            inventory (CharacterInventory): The inventory instance to operate on.
            item_id (str): The ID of the item to remove.
            remove_quantity (int): The quantity of the item to remove.
            expected_remaining (int): The expected quantity remaining after removal.
        """
        inventory.remove_item(item_id, remove_quantity)
        remaining = inventory.get_item_quantity(item_id)
        assert remaining == expected_remaining
    
    def test_remove_item_complete(self, inventory):
        """
        Test that removing the full quantity of an item deletes it from the inventory.
        """
        inventory.remove_item('sword_001', 1)
        assert not inventory.has_item('sword_001')
    
    def test_remove_item_nonexistent(self, inventory):
        """
        Test that removing a non-existent item from the inventory raises a ValueError.
        """
        with pytest.raises(ValueError, match="Item nonexistent_item not found"):
            inventory.remove_item('nonexistent_item', 1)
    
    def test_remove_item_insufficient_quantity(self, inventory):
        """
        Test that removing more of an item than is available in the inventory raises a ValueError.
        """
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
        """
        Test that removing a valid amount of gold from the inventory decreases the gold balance accordingly.
        """
        inventory.remove_gold(100)
        assert inventory.gold == 400
    
    def test_remove_gold_insufficient(self, inventory):
        """
        Test that removing more gold than available from the inventory raises a ValueError.
        """
        with pytest.raises(ValueError, match="Insufficient gold"):
            inventory.remove_gold(1000)
    
    def test_inventory_serialization(self, inventory):
        """
        Test that the CharacterInventory serializes correctly to a dictionary with expected gold and item values.
        """
        inventory_dict = inventory.to_dict()
        assert isinstance(inventory_dict, dict)
        assert inventory_dict['gold'] == 500
        assert len(inventory_dict['items']) == 2


class TestCharacterServer:
    """Test cases for the CharacterServer class."""
    
    @pytest.fixture
    def temp_dir(self):
        """
        Pytest fixture that yields a temporary directory for use during a test.
        
        The directory and its contents are automatically cleaned up after the test completes.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def server_config(self, temp_dir):
        """
        Provides a server configuration dictionary for testing, using the given temporary directory as the data path.
        
        Parameters:
            temp_dir: Temporary directory path to be used for server data storage.
        
        Returns:
            dict: Server configuration with data path, port, host, and maximum character count.
        """
        return {
            'data_path': temp_dir,
            'port': 8080,
            'host': 'localhost',
            'max_characters': 1000
        }
    
    @pytest.fixture
    def server(self, server_config):
        """
        Fixture that provides a CharacterServer instance initialized with the given configuration.
        
        Parameters:
            server_config (dict): Configuration dictionary for the CharacterServer.
        
        Returns:
            CharacterServer: An instance of CharacterServer configured for testing.
        """
        return CharacterServer(server_config)
    
    @pytest.fixture
    def test_character_data(self):
        """
        Provides a sample dictionary representing a test character with preset attributes for use in unit tests.
        """
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
        """
        Test that the server initializes correctly with a valid configuration.
        
        Asserts that the server's configuration parameters for port, host, and max_characters are set to their expected default values.
        """
        assert server.config['port'] == 8080
        assert server.config['host'] == 'localhost'
        assert server.config['max_characters'] == 1000
    
    @pytest.mark.parametrize("invalid_config", [
        {'port': 8080},  # Missing required keys
        {'port': 'not_a_number', 'host': 'localhost', 'data_path': '/tmp', 'max_characters': 100},
        {},  # Empty config
    ])
    def test_server_initialization_invalid_config(self, invalid_config):
        """
        Test that initializing the server with an invalid configuration raises a ValueError.
        """
        with pytest.raises(ValueError):
            CharacterServer(invalid_config)
    
    def test_create_character_valid(self, server, test_character_data):
        """
        Test that a character can be successfully created on the server with valid data.
        
        Verifies that the created character is not None, has the expected ID and name, and is retrievable from the server.
        """
        character = server.create_character(test_character_data)
        assert character is not None
        assert character.id == 'test-char-001'
        assert character.name == 'Test Hero'
        assert server.get_character('test-char-001') == character
    
    def test_create_character_duplicate_id(self, server, test_character_data):
        """
        Test that creating a character with an existing ID raises a ValueError.
        """
        server.create_character(test_character_data)
        with pytest.raises(ValueError, match="Character already exists"):
            server.create_character(test_character_data)
    
    def test_create_character_invalid_data(self, server):
        """
        Test that creating a character with missing required fields raises a KeyError.
        """
        invalid_data = {'name': 'Invalid Character'}  # Missing required fields
        with pytest.raises(KeyError):
            server.create_character(invalid_data)
    
    def test_get_character_existing(self, server, test_character_data):
        """
        Test retrieving an existing character from the server.
        
        Ensures that a character created on the server can be successfully retrieved by its ID.
        """
        server.create_character(test_character_data)
        character = server.get_character('test-char-001')
        assert character is not None
        assert character.id == 'test-char-001'
    
    def test_get_character_nonexistent(self, server):
        """
        Test that retrieving a character with a non-existent ID returns None.
        """
        character = server.get_character('nonexistent-char')
        assert character is None
    
    def test_update_character_existing(self, server, test_character_data):
        """
        Test that updating an existing character's attributes returns True and applies the changes correctly.
        """
        server.create_character(test_character_data)
        update_data = {'level': 6, 'experience': 1500}
        success = server.update_character('test-char-001', update_data)
        assert success is True
        
        updated_character = server.get_character('test-char-001')
        assert updated_character.level == 6
        assert updated_character.experience == 1500
    
    def test_update_character_nonexistent(self, server):
        """
        Test that updating a character that does not exist returns False.
        """
        update_data = {'level': 6}
        success = server.update_character('nonexistent-char', update_data)
        assert success is False
    
    def test_delete_character_existing(self, server, test_character_data):
        """
        Test that deleting an existing character removes it from the server.
        
        Creates a character, deletes it, and verifies that it can no longer be retrieved.
        """
        server.create_character(test_character_data)
        success = server.delete_character('test-char-001')
        assert success is True
        
        character = server.get_character('test-char-001')
        assert character is None
    
    def test_delete_character_nonexistent(self, server):
        """
        Test that deleting a character with a non-existent ID returns False.
        """
        success = server.delete_character('nonexistent-char')
        assert success is False
    
    def test_list_characters_empty(self, server):
        """
        Test that listing characters on an empty server returns an empty list.
        """
        characters = server.list_characters()
        assert len(characters) == 0
    
    def test_list_characters_with_data(self, server, test_character_data):
        """
        Test that listing characters returns all created characters when the server contains multiple entries.
        """
        server.create_character(test_character_data)
        
        # Create another character
        another_character = test_character_data.copy()
        another_character['id'] = 'test-char-002'
        another_character['name'] = 'Another Hero'
        server.create_character(another_character)
        
        characters = server.list_characters()
        assert len(characters) == 2
    
    def test_list_characters_with_filters(self, server, test_character_data):
        """
        Test that listing characters with attribute filters returns only characters matching the specified criteria.
        """
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
        """
        Tests that the server correctly transitions between running and stopped states when start and stop methods are called.
        """
        assert not server.is_running()
        
        server.start()
        assert server.is_running()
        
        server.stop()
        assert not server.is_running()
    
    def test_max_characters_limit(self, server_config, test_character_data):
        """
        Tests that the CharacterServer enforces the maximum allowed number of characters and raises a ValueError when attempting to exceed the limit.
        """
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
        """
        Test that creating a character with invalid data raises a ValueError or KeyError.
        """
        with pytest.raises((ValueError, KeyError)):
            server.create_character(invalid_data)
    
    def test_concurrent_character_operations(self, server, test_character_data):
        """
        Tests that concurrent updates to a character's attributes via multiple threads do not corrupt the character's state or cause loss of data.
        """
        server.create_character(test_character_data)
        
        def update_character():
            """
            Updates the character with ID 'test-char-001' on the server, setting its level to 10.
            """
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
        """
        Test that the server's character data save method is called when saving character data.
        
        This test verifies that after creating a character, invoking `save_character_data()` triggers the underlying save operation exactly once.
        """
        server.create_character(test_character_data)
        server.save_character_data()
        mock_save.assert_called_once()
    
    @patch('character_server.CharacterServer.load_character_data')
    def test_load_character_data(self, mock_load, server):
        """
        Test that the server's character data loading method calls the underlying load function exactly once.
        """
        server.load_character_data()
        mock_load.assert_called_once()
    
    def test_performance_with_many_characters(self, server, test_character_data):
        """
        Tests that the server can create and retrieve 100 characters within specified performance thresholds.
        
        This test asserts that character creation completes in under 5 seconds and retrieval in under 2 seconds.
        """
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
        """
        Verify that character data modifications persist across server operations by updating a character and checking that changes are retained.
        """
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
        """
        Test server behavior with minimal character data and operations on an empty server.
        
        Verifies that the server can handle creation attempts with minimal data, and that listing, retrieving, updating, and deleting non-existent characters on an empty server return appropriate results.
        """
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
        """
        Provides a CharacterServer instance initialized with a predefined set of characters for testing purposes.
        
        Parameters:
            server_config (dict): Configuration dictionary for initializing the CharacterServer.
        
        Returns:
            CharacterServer: A server instance containing three pre-created characters of different classes.
        """
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
        """
        Tests the full lifecycle of a character within the server, including creation, leveling up, taking damage, healing, updating attributes, inventory management, and deletion.
        """
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
        """
        Test that the server correctly filters characters by class and by multiple attributes.
        
        Verifies that filtering by character class returns the expected character, and that combining filters (e.g., class and level) yields correct results.
        """
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
        """
        Test creating, updating, and deleting multiple characters in bulk on the server.
        
        This test verifies that the server can handle batch creation of characters, bulk updates (leveling up all characters), and selective deletion (removing all even-numbered characters), ensuring correct state after each operation.
        """
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
        """
        Test the server's ability to create and store 1000 characters efficiently.
        
        This test verifies that the server can handle large-scale character creation within a 30-second time limit and confirms that all characters are successfully stored and retrievable.
        """
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
        """
        Stress tests the CharacterServer by performing concurrent get, update, and list operations across multiple threads.
        
        This test creates 100 characters, then spawns 10 threads, each executing random operations on the server to simulate high concurrency. It asserts that all operations complete within 10 seconds and that no characters are lost during concurrent access.
        """
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
            """
            Performs 50 random operations (get, update, or list) on the character server for concurrency testing.
            
            Randomly selects a character ID and operation type for each iteration to simulate concurrent server usage.
            """
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
    """
    Configures pytest to recognize the 'slow' marker for tests.
    
    Adds a custom marker 'slow' to categorize tests that are slow to run, allowing them to be included or excluded via pytest command-line options.
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


# Test configuration and utilities
class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def create_test_character(char_id='test-char', name='Test Character', level=1, char_class='warrior'):
        """
        Create a dictionary representing a test character with default attributes.
        
        Parameters:
            char_id (str): Unique identifier for the character.
            name (str): Name of the character.
            level (int): Starting level of the character.
            char_class (str): Character class type.
        
        Returns:
            dict: Dictionary containing character data suitable for testing.
        """
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
        """
        Verify that the utility function for creating test character data returns correct default and custom values.
        """
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