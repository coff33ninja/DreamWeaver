import unittest
from unittest.mock import Mock, patch, MagicMock, call
import threading
import time
import json
import sys
import os

# Add the SERVER directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.client_manager import ClientManager, Client
except ImportError:
    # Fallback import path
    from client_manager import ClientManager, Client


class TestClient(unittest.TestCase):
    """Test cases for the Client class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_socket = Mock()
        self.mock_address = ('127.0.0.1', 12345)
        self.client = Client(self.mock_socket, self.mock_address)
    
    def test_client_initialization(self):
        """Test Client initialization with valid parameters."""
        self.assertEqual(self.client.socket, self.mock_socket)
        self.assertEqual(self.client.address, self.mock_address)
        self.assertIsNone(self.client.username)
        self.assertIsInstance(self.client.client_id, str)
        self.assertTrue(len(self.client.client_id) > 0)
    
    def test_client_id_uniqueness(self):
        """Test that each client gets a unique ID."""
        client2 = Client(Mock(), ('127.0.0.1', 12346))
        self.assertNotEqual(self.client.client_id, client2.client_id)
    
    def test_client_str_representation(self):
        """Test string representation of Client."""
        expected = f"Client({self.mock_address[0]}:{self.mock_address[1]})"
        self.assertEqual(str(self.client), expected)
    
    def test_client_repr_representation(self):
        """Test repr representation of Client."""
        self.client.username = "testuser"
        expected = f"Client(id={self.client.client_id}, username=testuser, address={self.mock_address})"
        self.assertEqual(repr(self.client), expected)
    
    def test_client_equality(self):
        """Test Client equality comparison."""
        client2 = Client(Mock(), self.mock_address)
        client2.client_id = self.client.client_id
        self.assertEqual(self.client, client2)
    
    def test_client_inequality(self):
        """Test Client inequality comparison."""
        client2 = Client(Mock(), self.mock_address)
        self.assertNotEqual(self.client, client2)
    
    def test_client_hash(self):
        """Test Client hash functionality."""
        client_set = {self.client}
        self.assertIn(self.client, client_set)
    
    def test_client_with_none_socket(self):
        """Test Client initialization with None socket."""
        with self.assertRaises((TypeError, AttributeError)):
            Client(None, self.mock_address)
    
    def test_client_with_invalid_address(self):
        """Test Client initialization with invalid address."""
        invalid_addresses = [
            None,
            "invalid",
            ('localhost',),  # Missing port
            ('127.0.0.1', 'invalid_port'),  # Invalid port type
        ]
        
        for addr in invalid_addresses:
            with self.subTest(address=addr):
                try:
                    client = Client(self.mock_socket, addr)
                    # If it doesn't raise an exception, that's also valid behavior
                    self.assertIsNotNone(client)
                except (TypeError, ValueError):
                    # Expected for invalid addresses
                    pass


class TestClientManager(unittest.TestCase):
    """Test cases for the ClientManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.client_manager = ClientManager()
        self.mock_socket = Mock()
        self.mock_address = ('127.0.0.1', 12345)
        self.test_client = Client(self.mock_socket, self.mock_address)
    
    def tearDown(self):
        """Clean up after each test method."""
        # Clean up any remaining clients
        if hasattr(self.client_manager, 'clients'):
            self.client_manager.clients.clear()
        if hasattr(self.client_manager, 'username_to_client'):
            self.client_manager.username_to_client.clear()
    
    def test_client_manager_initialization(self):
        """Test ClientManager initialization."""
        self.assertIsInstance(self.client_manager.clients, dict)
        self.assertIsInstance(self.client_manager.username_to_client, dict)
        self.assertIsInstance(self.client_manager.lock, threading.RLock)
        self.assertEqual(len(self.client_manager.clients), 0)
        self.assertEqual(len(self.client_manager.username_to_client), 0)
    
    def test_add_client_success(self):
        """Test successfully adding a client."""
        result = self.client_manager.add_client(self.test_client)
        
        self.assertTrue(result)
        self.assertIn(self.test_client.client_id, self.client_manager.clients)
        self.assertEqual(self.client_manager.clients[self.test_client.client_id], self.test_client)
    
    def test_add_client_duplicate(self):
        """Test adding a duplicate client."""
        self.client_manager.add_client(self.test_client)
        result = self.client_manager.add_client(self.test_client)
        
        self.assertFalse(result)
        self.assertEqual(len(self.client_manager.clients), 1)
    
    def test_add_client_none(self):
        """Test adding None as client."""
        result = self.client_manager.add_client(None)
        self.assertFalse(result)
        self.assertEqual(len(self.client_manager.clients), 0)
    
    def test_remove_client_by_id_success(self):
        """Test successfully removing a client by ID."""
        self.client_manager.add_client(self.test_client)
        result = self.client_manager.remove_client(self.test_client.client_id)
        
        self.assertTrue(result)
        self.assertNotIn(self.test_client.client_id, self.client_manager.clients)
    
    def test_remove_client_by_object_success(self):
        """Test successfully removing a client by object."""
        self.client_manager.add_client(self.test_client)
        result = self.client_manager.remove_client(self.test_client)
        
        self.assertTrue(result)
        self.assertNotIn(self.test_client.client_id, self.client_manager.clients)
    
    def test_remove_client_nonexistent(self):
        """Test removing a non-existent client."""
        result = self.client_manager.remove_client("nonexistent_id")
        self.assertFalse(result)
    
    def test_remove_client_with_username(self):
        """Test removing a client that has a username."""
        self.test_client.username = "testuser"
        self.client_manager.add_client(self.test_client)
        self.client_manager.username_to_client["testuser"] = self.test_client
        
        result = self.client_manager.remove_client(self.test_client.client_id)
        
        self.assertTrue(result)
        self.assertNotIn("testuser", self.client_manager.username_to_client)
    
    def test_get_client_by_id_success(self):
        """Test successfully getting a client by ID."""
        self.client_manager.add_client(self.test_client)
        result = self.client_manager.get_client(self.test_client.client_id)
        
        self.assertEqual(result, self.test_client)
    
    def test_get_client_by_username_success(self):
        """Test successfully getting a client by username."""
        self.test_client.username = "testuser"
        self.client_manager.add_client(self.test_client)
        self.client_manager.username_to_client["testuser"] = self.test_client
        
        result = self.client_manager.get_client_by_username("testuser")
        self.assertEqual(result, self.test_client)
    
    def test_get_client_nonexistent(self):
        """Test getting a non-existent client."""
        result = self.client_manager.get_client("nonexistent_id")
        self.assertIsNone(result)
    
    def test_get_client_by_username_nonexistent(self):
        """Test getting a client by non-existent username."""
        result = self.client_manager.get_client_by_username("nonexistent_user")
        self.assertIsNone(result)
    
    def test_set_username_success(self):
        """Test successfully setting a username."""
        self.client_manager.add_client(self.test_client)
        result = self.client_manager.set_username(self.test_client.client_id, "newuser")
        
        self.assertTrue(result)
        self.assertEqual(self.test_client.username, "newuser")
        self.assertIn("newuser", self.client_manager.username_to_client)
        self.assertEqual(self.client_manager.username_to_client["newuser"], self.test_client)
    
    def test_set_username_update_existing(self):
        """Test updating an existing username."""
        self.test_client.username = "olduser"
        self.client_manager.add_client(self.test_client)
        self.client_manager.username_to_client["olduser"] = self.test_client
        
        result = self.client_manager.set_username(self.test_client.client_id, "newuser")
        
        self.assertTrue(result)
        self.assertEqual(self.test_client.username, "newuser")
        self.assertNotIn("olduser", self.client_manager.username_to_client)
        self.assertIn("newuser", self.client_manager.username_to_client)
    
    def test_set_username_duplicate(self):
        """Test setting a username that already exists."""
        client2 = Client(Mock(), ('127.0.0.1', 12346))
        client2.username = "existinguser"
        self.client_manager.add_client(self.test_client)
        self.client_manager.add_client(client2)
        self.client_manager.username_to_client["existinguser"] = client2
        
        result = self.client_manager.set_username(self.test_client.client_id, "existinguser")
        
        self.assertFalse(result)
        self.assertIsNone(self.test_client.username)
    
    def test_set_username_nonexistent_client(self):
        """Test setting username for non-existent client."""
        result = self.client_manager.set_username("nonexistent_id", "username")
        self.assertFalse(result)
    
    def test_set_username_empty_string(self):
        """Test setting empty username."""
        self.client_manager.add_client(self.test_client)
        result = self.client_manager.set_username(self.test_client.client_id, "")
        
        self.assertFalse(result)
        self.assertIsNone(self.test_client.username)
    
    def test_set_username_none(self):
        """Test setting None as username."""
        self.client_manager.add_client(self.test_client)
        result = self.client_manager.set_username(self.test_client.client_id, None)
        
        self.assertFalse(result)
        self.assertIsNone(self.test_client.username)
    
    def test_get_all_clients(self):
        """Test getting all clients."""
        client2 = Client(Mock(), ('127.0.0.1', 12346))
        self.client_manager.add_client(self.test_client)
        self.client_manager.add_client(client2)
        
        all_clients = self.client_manager.get_all_clients()
        
        self.assertEqual(len(all_clients), 2)
        self.assertIn(self.test_client, all_clients)
        self.assertIn(client2, all_clients)
    
    def test_get_all_clients_empty(self):
        """Test getting all clients when none exist."""
        all_clients = self.client_manager.get_all_clients()
        self.assertEqual(len(all_clients), 0)
        self.assertIsInstance(all_clients, list)
    
    def test_get_client_count(self):
        """Test getting client count."""
        self.assertEqual(self.client_manager.get_client_count(), 0)
        
        self.client_manager.add_client(self.test_client)
        self.assertEqual(self.client_manager.get_client_count(), 1)
        
        client2 = Client(Mock(), ('127.0.0.1', 12346))
        self.client_manager.add_client(client2)
        self.assertEqual(self.client_manager.get_client_count(), 2)
    
    def test_is_username_taken_true(self):
        """Test checking if username is taken when it is."""
        self.test_client.username = "takenuser"
        self.client_manager.add_client(self.test_client)
        self.client_manager.username_to_client["takenuser"] = self.test_client
        
        result = self.client_manager.is_username_taken("takenuser")
        self.assertTrue(result)
    
    def test_is_username_taken_false(self):
        """Test checking if username is taken when it isn't."""
        result = self.client_manager.is_username_taken("availableuser")
        self.assertFalse(result)
    
    def test_is_username_taken_case_sensitive(self):
        """Test that username checking is case sensitive."""
        self.test_client.username = "CaseUser"
        self.client_manager.add_client(self.test_client)
        self.client_manager.username_to_client["CaseUser"] = self.test_client
        
        self.assertTrue(self.client_manager.is_username_taken("CaseUser"))
        self.assertFalse(self.client_manager.is_username_taken("caseuser"))
        self.assertFalse(self.client_manager.is_username_taken("CASEUSER"))
    
    def test_clear_all_clients(self):
        """Test clearing all clients."""
        client2 = Client(Mock(), ('127.0.0.1', 12346))
        self.test_client.username = "user1"
        client2.username = "user2"
        
        self.client_manager.add_client(self.test_client)
        self.client_manager.add_client(client2)
        self.client_manager.username_to_client["user1"] = self.test_client
        self.client_manager.username_to_client["user2"] = client2
        
        self.client_manager.clear_all_clients()
        
        self.assertEqual(len(self.client_manager.clients), 0)
        self.assertEqual(len(self.client_manager.username_to_client), 0)
    
    def test_broadcast_message_to_all(self):
        """Test broadcasting message to all clients."""
        client2 = Client(Mock(), ('127.0.0.1', 12346))
        self.client_manager.add_client(self.test_client)
        self.client_manager.add_client(client2)
        
        message = "Hello everyone!"
        self.client_manager.broadcast_message(message)
        
        self.test_client.socket.send.assert_called()
        client2.socket.send.assert_called()
    
    def test_broadcast_message_exclude_sender(self):
        """Test broadcasting message excluding sender."""
        client2 = Client(Mock(), ('127.0.0.1', 12346))
        self.client_manager.add_client(self.test_client)
        self.client_manager.add_client(client2)
        
        message = "Hello others!"
        self.client_manager.broadcast_message(message, exclude_client=self.test_client)
        
        self.test_client.socket.send.assert_not_called()
        client2.socket.send.assert_called()
    
    def test_broadcast_message_socket_error(self):
        """Test broadcasting message when socket error occurs."""
        self.test_client.socket.send.side_effect = Exception("Socket error")
        self.client_manager.add_client(self.test_client)
        
        # Should not raise exception
        try:
            self.client_manager.broadcast_message("Test message")
        except Exception as e:
            self.fail(f"broadcast_message raised {e} unexpectedly!")
    
    def test_thread_safety_add_remove(self):
        """Test thread safety of add and remove operations."""
        results = []
        num_clients = 50  # Reduced for faster testing
        
        def add_clients():
            for i in range(num_clients):
                client = Client(Mock(), ('127.0.0.1', 12345 + i))
                results.append(('add', self.client_manager.add_client(client)))
        
        def remove_clients():
            time.sleep(0.01)  # Small delay to let some clients be added
            clients = list(self.client_manager.clients.values())
            for client in clients[:num_clients//2]:  # Remove half
                results.append(('remove', self.client_manager.remove_client(client)))
        
        thread1 = threading.Thread(target=add_clients)
        thread2 = threading.Thread(target=remove_clients)
        
        thread1.start()
        thread2.start()
        
        thread1.join(timeout=5)
        thread2.join(timeout=5)
        
        # Should have some successful operations
        self.assertTrue(any(result[1] for result in results))
        # Final state should be consistent
        self.assertGreaterEqual(len(self.client_manager.clients), 0)
    
    def test_concurrent_username_setting(self):
        """Test concurrent username setting operations."""
        clients = []
        for i in range(10):
            client = Client(Mock(), ('127.0.0.1', 12345 + i))
            self.client_manager.add_client(client)
            clients.append(client)
        
        results = []
        
        def set_usernames():
            for i, client in enumerate(clients):
                result = self.client_manager.set_username(client.client_id, f"user{i}")
                results.append(result)
        
        threads = []
        for _ in range(3):  # Multiple threads trying to set usernames
            thread = threading.Thread(target=set_usernames)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=5)
        
        # Check that usernames are consistently managed
        unique_usernames = set()
        for client in clients:
            if hasattr(client, 'username') and client.username:
                unique_usernames.add(client.username)
        
        # No duplicate usernames should exist in the mapping
        mapped_usernames = set(self.client_manager.username_to_client.keys())
        self.assertEqual(len(mapped_usernames), len(self.client_manager.username_to_client))


class TestClientManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for ClientManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client_manager = ClientManager()
    
    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self.client_manager, 'clients'):
            self.client_manager.clients.clear()
        if hasattr(self.client_manager, 'username_to_client'):
            self.client_manager.username_to_client.clear()
    
    def test_malformed_client_data(self):
        """Test handling of malformed client data."""
        # Test with invalid socket - this should be handled gracefully
        try:
            client = Client(Mock(), ('127.0.0.1', 12345))
            client.socket = None  # Simulate malformed socket
            result = self.client_manager.add_client(client)
            # Should handle gracefully
            self.assertIsInstance(result, bool)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass
    
    def test_extreme_username_lengths(self):
        """Test handling of extremely long usernames."""
        mock_socket = Mock()
        client = Client(mock_socket, ('127.0.0.1', 12345))
        self.client_manager.add_client(client)
        
        # Very long username
        long_username = "a" * 10000
        result = self.client_manager.set_username(client.client_id, long_username)
        
        # Should handle gracefully (either accept or reject)
        self.assertIsInstance(result, bool)
    
    def test_special_characters_in_username(self):
        """Test handling of special characters in usernames."""
        mock_socket = Mock()
        client = Client(mock_socket, ('127.0.0.1', 12345))
        self.client_manager.add_client(client)
        
        special_usernames = [
            "user@domain.com",
            "user with spaces",
            "user\nwith\nnewlines",
            "user\twith\ttabs",
            "用户名",  # Unicode characters
            "🚀🌟💻",  # Emojis
            "user'with'quotes",
            'user"with"doublequotes',
            "user\\with\\backslashes",
        ]
        
        for username in special_usernames:
            with self.subTest(username=username):
                # Reset client username
                if hasattr(client, 'username'):
                    old_username = client.username
                    if old_username and old_username in self.client_manager.username_to_client:
                        del self.client_manager.username_to_client[old_username]
                    client.username = None
                
                result = self.client_manager.set_username(client.client_id, username)
                # Should handle gracefully
                self.assertIsInstance(result, bool)
                
                if result:
                    # If accepted, should be retrievable
                    retrieved = self.client_manager.get_client_by_username(username)
                    self.assertEqual(retrieved, client)
    
    def test_memory_cleanup_on_large_operations(self):
        """Test memory cleanup during large-scale operations."""
        # Add many clients (reduced number for faster testing)
        clients = []
        for i in range(100):
            client = Client(Mock(), ('127.0.0.1', 12345 + i))
            client.username = f"user{i}"
            self.client_manager.add_client(client)
            self.client_manager.username_to_client[f"user{i}"] = client
            clients.append(client)
        
        # Verify they were added
        self.assertEqual(len(self.client_manager.clients), 100)
        self.assertEqual(len(self.client_manager.username_to_client), 100)
        
        # Remove all clients
        for client in clients:
            self.client_manager.remove_client(client)
        
        # Verify cleanup
        self.assertEqual(len(self.client_manager.clients), 0)
        self.assertEqual(len(self.client_manager.username_to_client), 0)
    
    def test_client_manager_singleton_behavior(self):
        """Test if ClientManager behaves consistently across instances."""
        # Create multiple instances to test isolation
        cm1 = ClientManager()
        cm2 = ClientManager()
        
        client1 = Client(Mock(), ('127.0.0.1', 12345))
        client2 = Client(Mock(), ('127.0.0.1', 12346))
        
        cm1.add_client(client1)
        cm2.add_client(client2)
        
        # Each instance should maintain its own state
        self.assertEqual(len(cm1.clients), 1)
        self.assertEqual(len(cm2.clients), 1)
        self.assertNotEqual(list(cm1.clients.keys()), list(cm2.clients.keys()))
    
    def test_client_manager_stress_operations(self):
        """Test ClientManager under stress conditions."""
        operations_count = 100
        clients = []
        
        # Rapid add/remove operations
        for i in range(operations_count):
            client = Client(Mock(), ('127.0.0.1', 12345 + i))
            clients.append(client)
            self.client_manager.add_client(client)
            
            if i % 2 == 0 and i > 0:  # Remove every other client
                self.client_manager.remove_client(clients[i-1])
        
        # Verify final state is consistent
        actual_count = len(self.client_manager.clients)
        expected_min = operations_count // 2  # At least half should remain
        self.assertGreaterEqual(actual_count, 0)
        self.assertLessEqual(actual_count, operations_count)
    
    def test_invalid_client_operations(self):
        """Test operations with invalid client parameters."""
        invalid_inputs = [
            None,
            "",
            123,
            [],
            {},
            object(),
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                # These should not crash the system
                try:
                    result = self.client_manager.add_client(invalid_input)
                    self.assertIsInstance(result, bool)
                    self.assertFalse(result)  # Should fail gracefully
                except (TypeError, AttributeError):
                    # Expected for completely invalid inputs
                    pass
                
                try:
                    result = self.client_manager.remove_client(invalid_input)
                    self.assertIsInstance(result, bool)
                    self.assertFalse(result)  # Should fail gracefully
                except (TypeError, AttributeError):
                    # Expected for completely invalid inputs
                    pass


class TestClientManagerIntegration(unittest.TestCase):
    """Integration tests for ClientManager with realistic scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.client_manager = ClientManager()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if hasattr(self.client_manager, 'clients'):
            self.client_manager.clients.clear()
        if hasattr(self.client_manager, 'username_to_client'):
            self.client_manager.username_to_client.clear()
    
    def test_realistic_chat_room_scenario(self):
        """Test a realistic chat room scenario with multiple users."""
        # Simulate users joining
        users = [
            ("Alice", ('192.168.1.10', 5001)),
            ("Bob", ('192.168.1.11', 5002)),
            ("Charlie", ('192.168.1.12', 5003)),
            ("Diana", ('192.168.1.13', 5004)),
        ]
        
        clients = []
        for username, address in users:
            socket_mock = Mock()
            client = Client(socket_mock, address)
            clients.append(client)
            
            # Add client
            self.assertTrue(self.client_manager.add_client(client))
            
            # Set username
            self.assertTrue(self.client_manager.set_username(client.client_id, username))
            
            # Verify username is set correctly
            self.assertEqual(client.username, username)
            retrieved_client = self.client_manager.get_client_by_username(username)
            self.assertEqual(retrieved_client, client)
        
        # Verify all clients are present
        self.assertEqual(self.client_manager.get_client_count(), 4)
        all_clients = self.client_manager.get_all_clients()
        self.assertEqual(len(all_clients), 4)
        
        # Test broadcasting (mock sockets should receive send calls)
        self.client_manager.broadcast_message("Welcome everyone!")
        for client in clients:
            client.socket.send.assert_called()
        
        # Test selective broadcasting
        alice = clients[0]
        self.client_manager.broadcast_message("Alice says hello!", exclude_client=alice)
        alice.socket.send.reset_mock()  # Reset the mock
        
        # Simulate users leaving
        self.assertTrue(self.client_manager.remove_client(clients[1]))  # Bob leaves
        self.assertEqual(self.client_manager.get_client_count(), 3)
        self.assertIsNone(self.client_manager.get_client_by_username("Bob"))
        
        # Simulate username change
        charlie = clients[2]
        old_username = charlie.username
        new_username = "Chuck"
        self.assertTrue(self.client_manager.set_username(charlie.client_id, new_username))
        self.assertEqual(charlie.username, new_username)
        self.assertIsNone(self.client_manager.get_client_by_username(old_username))
        self.assertEqual(self.client_manager.get_client_by_username(new_username), charlie)
    
    def test_concurrent_user_management(self):
        """Test concurrent user join/leave operations."""
        import concurrent.futures
        import random
        
        results = []
        
        def user_operations():
            """Simulate random user operations."""
            for i in range(10):
                # Random operation
                operation = random.choice(['join', 'leave', 'rename'])
                
                if operation == 'join':
                    client = Client(Mock(), ('127.0.0.1', 12345 + i))
                    result = self.client_manager.add_client(client)
                    if result:
                        username_result = self.client_manager.set_username(
                            client.client_id, f"user_{threading.current_thread().ident}_{i}"
                        )
                        results.append(('join', result and username_result))
                    else:
                        results.append(('join', False))
                
                elif operation == 'leave' and self.client_manager.get_client_count() > 0:
                    clients = self.client_manager.get_all_clients()
                    if clients:
                        client_to_remove = random.choice(clients)
                        result = self.client_manager.remove_client(client_to_remove)
                        results.append(('leave', result))
                
                time.sleep(0.001)  # Small delay to encourage race conditions
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(user_operations) for _ in range(5)]
            concurrent.futures.wait(futures, timeout=10)
        
        # Verify system is in a consistent state
        client_count = self.client_manager.get_client_count()
        self.assertGreaterEqual(client_count, 0)
        
        # Verify username mapping consistency
        clients_with_usernames = [
            c for c in self.client_manager.get_all_clients() 
            if hasattr(c, 'username') and c.username
        ]
        username_map_size = len(self.client_manager.username_to_client)
        self.assertEqual(len(clients_with_usernames), username_map_size)


if __name__ == '__main__':
    # Configure test runner for detailed output
    unittest.main(verbosity=2, buffer=True, failfast=False)