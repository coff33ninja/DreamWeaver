from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from .database import Database # Assuming Database class is in database.py
from .config import DB_PATH
import psutil
from datetime import datetime, timezone  # Removed unused timedelta

router = APIRouter()

# Dependency to get a database instance
def get_db():
    """
    Yields a database connection for use in FastAPI dependencies, ensuring the connection is closed after use.
    """
    db = Database(DB_PATH)
    try:
        yield db
    finally:
        db.close() # Important to close the connection

@router.get("/dashboard/status", tags=["dashboard"], summary="Get System and Client Status")
async def get_system_status(db: Database = Depends(get_db)):
    """
    Retrieve current server performance metrics and a detailed list of client statuses.
    
    Returns:
        dict: A dictionary containing:
            - server_performance: CPU usage percentage, memory usage percentage, used and total memory in GB.
            - client_statuses: List of clients with actor ID, combined IP address and port, last seen timestamp (UTC), and status. Status values include: 'Registered', 'Offline', 'Online_Heartbeat', 'Online_Responsive', 'Error_API', 'Error_Unreachable', and 'Deactivated'.
    """
    # Server Performance
    cpu_usage = psutil.cpu_percent(interval=0.1) # Non-blocking
    memory_info = psutil.virtual_memory()

    # Client Statuses from database
    # get_all_client_statuses() should now return Actor_id, ip_address, client_port, last_seen, status
    raw_clients = db.get_all_client_statuses()

    processed_clients = []
    if raw_clients: # Check if raw_clients is not None and not empty
        for client_data in raw_clients:
            # Convert Row object to dict if necessary, or access by attribute/key
            # Assuming client_data is already a dict-like object from db.get_all_client_statuses()
            Actor_id = client_data.get("Actor_id", "Unknown Actor")
            ip_addr = client_data.get("ip_address", "N/A")
            port = client_data.get("client_port", "N/A")
            last_seen_iso = client_data.get("last_seen")
            current_status = client_data.get("status", "Unknown") # Get status from DB

            # Optional: Format last_seen timestamp for better readability
            last_seen_display = "Never"
            if last_seen_iso:
                try:
                    last_seen_dt = datetime.fromisoformat(last_seen_iso).astimezone(timezone.utc)
                    # Format for display, e.g., "YYYY-MM-DD HH:MM:SS UTC"
                    last_seen_display = last_seen_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                except (ValueError, TypeError):
                    last_seen_display = last_seen_iso # Show raw if parsing fails

            processed_clients.append({
                "Actor_id": Actor_id,
                "ip_address": f"{ip_addr}:{port}" if ip_addr != "N/A" and port != "N/A" else ip_addr,
                "last_seen": last_seen_display,
                "status": current_status # Directly use the status from DB
            })

    return {
        "server_performance": {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_info.percent,
            "memory_used_gb": round(memory_info.used / (1024**3), 2),
            "memory_total_gb": round(memory_info.total / (1024**3), 2),
        },
        "client_statuses": processed_clients
    }

@router.get("/dashboard", response_class=HTMLResponse, tags=["dashboard"], summary="View Hive Dashboard")
async def get_dashboard_page():
    """
    Serves the HTML page for the Hive monitoring dashboard, providing the frontend interface for viewing server performance and client statuses.
    
    The page includes embedded JavaScript that periodically fetches status data from the backend and updates the dashboard in real time.
    Returns:
        HTMLResponse: The rendered dashboard page.
    """
    # Status CSS classes will map to the new DB statuses
    # e.g., status-online-responsive, status-online-heartbeat, status-error-api, etc.
    html_content = r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dream Weaver - Hive Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #1a1a2e; color: #e0e0e0; display: flex; justify-content: center; align-items: flex-start; min-height: 100vh; padding-top: 40px; box-sizing: border-box; }
            .container { width: 90%; max-width: 1400px; background: #24243e; padding: 25px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            h1 { color: #90a0ff; text-align: center; margin-bottom: 20px; border-bottom: 1px solid #4a4a6a; padding-bottom: 15px; }
            h2 { color: #82aaff; margin-top: 30px; margin-bottom: 15px; border-bottom: 1px solid #4a4a6a; padding-bottom: 10px;}
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { padding: 12px 15px; border: 1px solid #3a3a52; text-align: left; font-size: 0.95em; }
            th { background-color: #31314c; color: #b0c4ff; }
            td { background-color: #2c2c44; }
            tr:nth-child(even) td { background-color: #2f2f47; }
            .status-registered { color: #a0a0a0; }
            .status-offline { color: #ff7b7b; }
            .status-online_heartbeat { color: #ffd700; } /* Gold/Yellow for heartbeat only */
            .status-online_responsive { color: #76ff7b; font-weight: bold; } /* Bright Green for fully responsive */
            .status-error_api { color: #ff9a00; } /* Orange for API errors */
            .status-error_unreachable { color: #ff5555; font-weight: bold; } /* Red for unreachable */
            .status-deactivated { color: #606060; font-style: italic; }
            .status-unknown { color: #cccccc; }
            .perf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px; }
            .perf-card { background: #2c2c44; padding: 18px; border-radius: 6px; border-left: 5px solid #82aaff; }
            .perf-card strong { color: #b0c4ff; display: block; margin-bottom: 5px; font-size: 0.9em; text-transform: uppercase;}
            .loader, .error-message { text-align: center; padding: 20px; font-size: 1.1em; color: #82aaff;}
            .error-message { color: #ff7b7b; background-color: #3a3a52; border-radius: 5px; margin-top: 10px;}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dream Weaver - Hive Dashboard</h1>
            <h2>Server Performance</h2>
            <div id="perf-grid" class="perf-grid"><div class="loader">Loading server performance...</div></div>
            <div id="perf-error" class="error-message" style="display:none;"></div>

            <h2>Client Status</h2>
            <table id="client-status-table">
                <thead>
                    <tr>
                        <th>Client Actor ID</th>
                        <th>Status</th>
                        <th>IP Address : Port</th>
                        <th>Last Seen (UTC)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td colspan="4" class="loader">Loading client statuses...</td></tr>
                </tbody>
            </table>
            <div id="client-error" class="error-message" style="display:none;"></div>
        </div>

        <script>
            function formatStatusClass(statusString) {
                if (!statusString) return 'status-unknown';
                return 'status-' + statusString.toLowerCase().replace(/ /g, '_');
            }

            function updateDashboard() {
                fetch('/dashboard/status')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`Network error: ${response.status} ${response.statusText}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        document.getElementById('perf-error').style.display = 'none';
                        document.getElementById('client-error').style.display = 'none';

                        const perf = data.server_performance;
                        const perfGrid = document.getElementById('perf-grid');
                        perfGrid.innerHTML = `
                            <div class="perf-card"><strong>CPU Usage</strong> ${perf.cpu_usage_percent.toFixed(1)}%</div>
                            <div class="perf-card"><strong>Memory Usage</strong> ${perf.memory_usage_percent.toFixed(1)}% (${perf.memory_used_gb} GB / ${perf.memory_total_gb} GB)</div>
                        `;

                        const tableBody = document.querySelector("#client-status-table tbody");
                        if (data.client_statuses && data.client_statuses.length > 0) {
                            tableBody.innerHTML = '';
                            data.client_statuses.forEach(client => {
                                const statusClass = formatStatusClass(client.status);
                                const ipPort = client.ip_address || 'N/A'; // Already combined or N/A from server
                                const row = `<tr>
                                    <td>${client.Actor_id || 'N/A'}</td>
                                    <td class="${statusClass}">${client.status || 'Unknown'}</td>
                                    <td>${ipPort}</td>
                                    <td>${client.last_seen || 'Never'}</td>
                                </tr>`;
                                tableBody.innerHTML += row;
                            });
                        } else {
                            tableBody.innerHTML = '<tr><td colspan="4" class="loader">No clients registered or found.</td></tr>';
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching dashboard data:', error);
                        document.getElementById('perf-error').textContent = 'Could not load server performance. ' + error.message;
                        document.getElementById('perf-error').style.display = 'block';
                        document.getElementById('client-error').textContent = 'Could not load client statuses. ' + error.message;
                        document.getElementById('client-error').style.display = 'block';
                        document.getElementById('perf-grid').innerHTML = '';
                        document.querySelector("#client-status-table tbody").innerHTML = '<tr><td colspan="4" class="error-message">Error loading client data.</td></tr>';
                    });
            }
            document.addEventListener('DOMContentLoaded', () => {
                updateDashboard();
                setInterval(updateDashboard, 6000); // Refresh every 6 seconds
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
