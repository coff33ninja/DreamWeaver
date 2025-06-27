from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from .database import Database
from .config import DB_PATH # Import from config
import psutil
from datetime import datetime, timezone

router = APIRouter()

# Dependency to get a database instance
def get_db():
    # Use DB_PATH from config
    db = Database(DB_PATH)
    try:
        yield db
    finally:
        # If your Database class has a close() method, call it here
        # db.close()
        pass

@router.get("/dashboard/status", tags=["dashboard"])
async def get_system_status(db: Database = Depends(get_db)):
    """
    API endpoint to provide server performance and client status data.
    """
    # Server Performance
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()

    # Client Statuses
    clients = db.get_all_client_statuses() # This method should exist in your Database class
    now_utc = datetime.now(timezone.utc)

    processed_clients = []
    for client in clients:
        is_online = False
        last_seen_str = client.get("last_seen", None) # Use .get for safety
        if last_seen_str:
            try:
                # Ensure last_seen_str is in the correct ISO format, including timezone info if stored
                # If it's a naive datetime string from SQLite, assume UTC or local and make it aware
                last_seen_dt = datetime.fromisoformat(last_seen_str)
                if last_seen_dt.tzinfo is None: # If naive, assume UTC as per your previous logic
                    last_seen_dt = last_seen_dt.replace(tzinfo=timezone.utc)

                # A client is considered online if seen in the last ~75 seconds (heartbeat is 60s + buffer)
                if (now_utc - last_seen_dt).total_seconds() < 75:
                    is_online = True
            except ValueError as e:
                print(f"Error parsing last_seen timestamp '{last_seen_str}' for client {client.get('pc')}: {e}")
                # Keep is_online as False

        processed_clients.append({
            "pc": client.get("pc", "Unknown"),
            "ip_address": client.get("ip_address", "N/A"),
            "last_seen": last_seen_str if last_seen_str else "Never",
            "online_status": "Online" if is_online else "Offline"
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

@router.get("/dashboard", response_class=HTMLResponse, tags=["dashboard"])
async def get_dashboard_page():
    """
    Serves the main HTML page for the monitoring dashboard.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dream Weaver - Hive Dashboard</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #1e1e2f; color: #e0e0e0; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; box-sizing: border-box; }
            .container { width: 100%; max-width: 1200px; margin: auto; background: #2a2a3e; padding: 30px; border-radius: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.2); }
            h1, h2 { color: #82aaff; border-bottom: 2px solid #82aaff; padding-bottom: 10px; margin-top: 0;}
            h1 { text-align: center; margin-bottom: 30px;}
            table { width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
            th, td { padding: 15px; border: 1px solid #3a3a52; text-align: left; }
            th { background-color: #31314c; color: #b0c4ff; font-weight: bold; }
            td { background-color: #2c2c44; }
            tr:nth-child(even) td { background-color: #2f2f47; }
            .status-online { color: #50fa7b; font-weight: bold; }
            .status-offline { color: #ff5555; font-weight: bold; }
            .perf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-top: 20px; margin-bottom: 30px; }
            .perf-card { background: #2c2c44; padding: 20px; border-radius: 8px; border: 1px solid #3a3a52; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.2s ease-in-out; }
            .perf-card:hover { transform: translateY(-5px); }
            .perf-card strong { color: #82aaff; display: block; margin-bottom: 8px; }
            .loader { text-align: center; padding: 20px; font-size: 1.2em; color: #82aaff;}
            .error-message { color: #ff5555; text-align: center; padding: 10px; background-color: #3a3a52; border-radius: 5px; margin-top: 10px;}
            @media (max-width: 768px) {
                body { margin: 20px; padding: 10px; }
                .container { padding: 20px; }
                h1 { font-size: 1.8em; }
                h2 { font-size: 1.5em; }
                th, td { padding: 10px; }
                .perf-grid { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dream Weaver - Hive Dashboard</h1>

            <h2>Server Performance</h2>
            <div id="perf-grid" class="perf-grid">
                 <div class="loader">Loading server performance...</div>
            </div>
             <div id="perf-error" class="error-message" style="display:none;"></div>

            <h2>Client Status</h2>
            <table id="client-status-table">
                <thead>
                    <tr>
                        <th>Client PC</th>
                        <th>Status</th>
                        <th>IP Address</th>
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
            function displayError(elementId, message) {
                const errorEl = document.getElementById(elementId);
                if (errorEl) {
                    errorEl.textContent = message;
                    errorEl.style.display = 'block';
                }
            }

            function clearError(elementId) {
                const errorEl = document.getElementById(elementId);
                if (errorEl) {
                    errorEl.style.display = 'none';
                }
            }

            function updateDashboard() {
                fetch('/dashboard/status')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`Network response was not ok: ${response.statusText}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        clearError('perf-error');
                        clearError('client-error');

                        // Update Server Performance
                        const perf = data.server_performance;
                        const perfGrid = document.getElementById('perf-grid');
                        if (perfGrid) {
                            perfGrid.innerHTML = \`
                                <div class="perf-card"><strong>CPU Usage:</strong> \${perf.cpu_usage_percent.toFixed(1)}%</div>
                                <div class="perf-card"><strong>Memory Usage:</strong> \${perf.memory_usage_percent.toFixed(1)}% (\${perf.memory_used_gb} / \${perf.memory_total_gb} GB)</div>
                            \`;
                        }

                        // Update Client Status Table
                        const tableBody = document.querySelector("#client-status-table tbody");
                        if (tableBody) {
                            if (data.client_statuses && data.client_statuses.length > 0) {
                                tableBody.innerHTML = ''; // Clear existing rows
                                data.client_statuses.forEach(client => {
                                    const row = \`<tr>
                                        <td>\${client.pc || 'N/A'}</td>
                                        <td class="status-\${client.online_status ? client.online_status.toLowerCase() : 'offline'}">\${client.online_status || 'Offline'}</td>
                                        <td>\${client.ip_address || 'N/A'}</td>
                                        <td>\${client.last_seen || 'Never'}</td>
                                    </tr>\`;
                                    tableBody.innerHTML += row;
                                });
                            } else {
                                tableBody.innerHTML = '<tr><td colspan="4" class="loader">No clients found or connected.</td></tr>';
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching dashboard data:', error);
                        displayError('perf-error', 'Could not load server performance data.');
                        displayError('client-error', 'Could not load client status data.');
                        const perfGrid = document.getElementById('perf-grid');
                        if (perfGrid) perfGrid.innerHTML = ''; // Clear loader
                        const tableBody = document.querySelector("#client-status-table tbody");
                        if (tableBody) tableBody.innerHTML = '<tr><td colspan="4" class="error-message">Error loading client data.</td></tr>';
                    });
            }

            document.addEventListener('DOMContentLoaded', () => {
                updateDashboard();
                setInterval(updateDashboard, 7000); // Refresh every 7 seconds
            });
        </script>
    </body>
    </html>
    """;
    return HTMLResponse(content=html_content)
