import psutil
import socket
import logging

logger = logging.getLogger("dreamweaver_server")

def get_adapter_ip_addresses() -> dict[str, str]:
    """
    Returns a dictionary of network adapter names and their associated IPv4 addresses.
    Excludes loopback addresses.
    """
    adapters = {}
    try:
        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                # Check for IPv4 and exclude loopback addresses
                if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                    adapters[iface] = addr.address
                    logger.debug(f"Found adapter: {iface} with IP: {addr.address}")
    except ImportError:
        logger.error("psutil is not installed. Cannot get network adapter IPs. Please run 'pip install psutil'.")
        return {} # Return empty if psutil is not available
    except Exception as e:
        logger.error(f"An error occurred while trying to get network adapter IPs: {e}", exc_info=True)
        return {} # Return empty on other errors

    if not adapters:
        logger.warning("No suitable network adapters found or psutil failed to retrieve them.")
    return adapters

if __name__ == '__main__':
    # Example usage when run directly
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Attempting to retrieve network adapter IP addresses...")
    ips = get_adapter_ip_addresses()
    if ips:
        logger.info("Available network adapter IPs:")
        for iface, ip in ips.items():
            logger.info(f"  {iface}: {ip}")
    else:
        logger.warning("No network adapter IPs found or psutil is not installed.")
