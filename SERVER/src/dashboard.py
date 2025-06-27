from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from .database import Database
import psutil
from datetime import datetime, timezone

router = APIRouter()

# Dependency to get a database instance
def get_db():
    db = Database("E:/DreamWeaver/data/dream_weaver.db")
    try:
        yield db
    finally:
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
    clients = db.get_all_client_statuses()
    now_utc = datetime.now(timezone.utc)

    for client in clients:
        is_online = False
        if client["last_seen"]:
            # Convert SQLite string timestamp to datetime object
            last_seen_dt = datetime.fromisoformat(client["last_seen"]).replace(tzinfo=timezone.utc)
            # A client is considered online if seen in the last ~70 seconds (heartbeat is 60s)
            if (now_utc - last_seen_dt).total_seconds() < 75:
                is_online = True
        client["online_status"] = "Online" if is_online else "Offline"

    return {
        "server_performance": {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_info.percent,
            "memory_used_gb": round(memory_info.used / (1024**3), 2),
            "memory_total_gb": round(memory_info.total / (1024**3), 2),
        },
        "client_statuses": clients
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
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; background-color: #f7f7f7; color: #333; }
            .container { max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1, h2 { color: #555; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 12px; border: 1px solid #ddd; text-align: left; }
            th { background-color: #f2f2f2; }
            .status-online { color: #28a745; font-weight: bold; }
            .status-offline { color: #dc3545; font-weight: bold; }
            .perf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px; }
            .perf-card { background: #f9f9f9; padding: 15px; border-radius: 5px; border: 1px solid #eee; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dream Weaver - Hive Dashboard</h1>

            <h2>Server Performance</h2>
            <div id="perf-grid" class="perf-grid"></div>

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
                    <!-- Client data will be injected here -->
                </tbody>
            </table>
        </div>

        <script>
            function updateDashboard() {
                fetch('/dashboard/status')
                    .then(response => response.json())
                    .then(data => {
                        // Update Server Performance
                        const perf = data.server_performance;
                        document.getElementById('perf-grid').innerHTML = `
                            <div class="perf-card"><strong>CPU Usage:</strong> ${perf.cpu_usage_percent.toFixed(1)}%</div>
                            <div class="perf-card"><strong>Memory Usage:</strong> ${perf.memory_usage_percent.toFixed(1)}% (${perf.memory_used_gb} / ${perf.memory_total_gb} GB)</div>
                        `;

                        // Update Client Status Table
                        const tableBody = document.querySelector("#client-status-table tbody");
                        tableBody.innerHTML = ''; // Clear existing rows
                        data.client_statuses.forEach(client => {
                            const row = `<tr>
                                <td>${client.pc}</td>
                                <td class="status-${client.online_status.toLowerCase()}">${client.online_status}</td>
                                <td>${client.ip_address || 'N/A'}</td>
                                <td>${client.last_seen || 'Never'}</td>
                            </tr>`;
                            tableBody.innerHTML += row;
                        });
                    })
                    .catch(error => console.error('Error fetching dashboard data:', error));
            }

            // Initial load and periodic refresh
            document.addEventListener('DOMContentLoaded', () => {
                updateDashboard();
                setInterval(updateDashboard, 5000); // Refresh every 5 seconds
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
