"""
Real-time performance telemetry dashboard.
Visualizes routing decisions, latencies, throughput, and system metrics.
"""

import json
import time
import os
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, request
import logging
from monitoring.telemetry import get_default_collector, TelemetryCollector
from model.latency_probe import get_default_prober

logger = logging.getLogger(__name__)

app = Flask(__name__)

# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>KAI Performance Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #0d7377;
            padding-bottom: 15px;
        }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #0d7377, #14b8a6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .header .timestamp {
            color: #888;
            font-size: 0.9em;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(45, 45, 45, 0.9);
            border: 1px solid #0d7377;
            border-radius: 8px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .stat-card h3 {
            color: #14b8a6;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
            opacity: 0.8;
        }
        .stat-value {
            font-size: 2.2em;
            font-weight: bold;
            color: #fff;
            margin: 10px 0;
        }
        .stat-detail {
            font-size: 0.85em;
            color: #888;
            margin: 5px 0;
        }
        .chart-container {
            background: rgba(45, 45, 45, 0.9);
            border: 1px solid #0d7377;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        .chart-title {
            color: #14b8a6;
            font-size: 1.2em;
            margin-bottom: 15px;
            text-transform: uppercase;
            font-weight: 600;
        }
        .chart-box {
            position: relative;
            height: 300px;
            margin-bottom: 20px;
        }
        .table-container {
            background: rgba(45, 45, 45, 0.9);
            border: 1px solid #0d7377;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        th {
            background: rgba(13, 115, 119, 0.3);
            color: #14b8a6;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #0d7377;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(13, 115, 119, 0.2);
        }
        tr:hover {
            background: rgba(20, 184, 166, 0.1);
        }
        .alert {
            background: rgba(239, 68, 68, 0.1);
            border-left: 3px solid #ef4444;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .success {
            background: rgba(34, 197, 94, 0.1);
            border-left: 3px solid #22c55e;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
        }
        .badge-success { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
        .badge-warning { background: rgba(234, 179, 8, 0.2); color: #eab308; }
        .badge-danger { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
        .refresh-button {
            margin: 20px 0;
            text-align: center;
        }
        button {
            background: linear-gradient(90deg, #0d7377, #14b8a6);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
        }
        button:hover {
            opacity: 0.9;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-item {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #14b8a6;
        }
        .metric-label {
            color: #888;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #14b8a6;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 KAI Performance Dashboard</h1>
        <div class="timestamp">Last updated: <span id="timestamp"></span></div>
    </div>
    
    <div class="container">
        <div class="refresh-button">
            <button onclick="location.reload()">🔄 Refresh Dashboard</button>
            <button onclick="downloadReport()">📥 Download Report</button>
        </div>
        
        <div id="alerts"></div>
        
        <div class="grid" id="stats-grid">
            <!-- Populated by JavaScript -->
        </div>
        
        <div class="chart-container">
            <div class="chart-title">📊 Routing Performance</div>
            <div class="chart-box">
                <canvas id="routingChart"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">⚡ Inference Throughput (tokens/sec)</div>
            <div class="chart-box">
                <canvas id="throughputChart"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">🌐 Network Latency by Host</div>
            <div class="chart-box">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
        
        <div class="table-container">
            <div class="chart-title">📋 Recent Inferences</div>
            <table border="1" id="inferencesTable">
                <thead>
                    <tr>
                        <th>Inference ID</th>
                        <th>Model</th>
                        <th>Duration (ms)</th>
                        <th>Throughput (tok/s)</th>
                        <th>Tokens</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="inferencesBody">
                </tbody>
            </table>
        </div>
        
        <div class="table-container">
            <div class="chart-title">🛣️ Recent Routing Decisions</div>
            <table id="routingTable">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Chunk</th>
                        <th>Selected Host</th>
                        <th>Candidates</th>
                        <th>Decision Time (ms)</th>
                        <th>Method</th>
                    </tr>
                </thead>
                <tbody id="routingBody">
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        let routingChart, throughputChart, latencyChart;
        
        async function updateDashboard() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();
                
                // Update timestamp
                document.getElementById('timestamp').textContent = new Date().toLocaleTimeString();
                
                // Update stats cards
                updateStatsCards(data);
                
                // Update charts
                updateCharts(data);
                
                // Update tables
                updateTables(data);
                
            } catch (error) {
                console.error('Failed to fetch metrics:', error);
                showAlert('Failed to fetch metrics. Is the backend running?', 'danger');
            }
        }
        
        function updateStatsCards(data) {
            const grid = document.getElementById('stats-grid');
            grid.innerHTML = '';
            
            const cards = [
                {
                    title: 'Total Routings',
                    value: data.routing.total_decisions || 0,
                    detail: 'Last 5 mins'
                },
                {
                    title: 'Avg Decision Latency',
                    value: (data.routing.avg_decision_latency_ms || 0).toFixed(2) + ' ms',
                    detail: 'Lower is better'
                },
                {
                    title: 'Total Inferences',
                    value: data.inference.total_inferences || 0,
                    detail: 'Success rate: ' + (data.inference.success_rate_pct || 0).toFixed(1) + '%'
                },
                {
                    title: 'Avg Throughput',
                    value: (data.throughput.avg_tokens_per_second || 0).toFixed(1),
                    detail: 'tokens/sec'
                },
                {
                    title: 'Avg Inference Time',
                    value: (data.inference.avg_duration_ms || 0).toFixed(0) + ' ms',
                    detail: 'Min: ' + (data.inference.min_duration_ms || 0).toFixed(0)
                },
                {
                    title: 'Uptime',
                    value: formatUptime(data.uptime_seconds),
                    detail: 'System uptime'
                }
            ];
            
            cards.forEach(card => {
                const html = `
                    <div class="stat-card">
                        <h3>${card.title}</h3>
                        <div class="stat-value">${card.value}</div>
                        <div class="stat-detail">${card.detail}</div>
                    </div>
                `;
                grid.innerHTML += html;
            });
        }
        
        function updateCharts(data) {
            const routing = data.routing;
            const hosts = Object.keys(routing.hosts || {});
            
            // Routing chart
            if (routingChart) routingChart.destroy();
            const routingCtx = document.getElementById('routingChart').getContext('2d');
            routingChart = new Chart(routingCtx, {
                type: 'doughnut',
                data: {
                    labels: hosts,
                    datasets: [{
                        data: hosts.map(h => routing.hosts[h].selection_count || 0),
                        backgroundColor: [
                            'rgba(13, 115, 119, 0.8)',
                            'rgba(20, 184, 166, 0.8)',
                            'rgba(13, 148, 136, 0.8)',
                            'rgba(12, 74, 110, 0.8)',
                        ],
                        borderColor: '#1e1e1e',
                        borderWidth: 2,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#e0e0e0' } } }
                }
            });
            
            // Throughput chart
            if (throughputChart) throughputChart.destroy();
            const throughputCtx = document.getElementById('throughputChart').getContext('2d');
            throughputChart = new Chart(throughputCtx, {
                type: 'line',
                data: {
                    labels: ['Min', 'Avg', 'Max'],
                    datasets: [{
                        label: 'Tokens/sec',
                        data: [
                            data.throughput.min_tokens_per_second || 0,
                            data.throughput.avg_tokens_per_second || 0,
                            data.throughput.max_tokens_per_second || 0
                        ],
                        borderColor: '#14b8a6',
                        backgroundColor: 'rgba(20, 184, 166, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#e0e0e0' } } },
                    scales: {
                        y: { ticks: { color: '#888' }, grid: { color: 'rgba(13, 115, 119, 0.1)' } },
                        x: { ticks: { color: '#888' }, grid: { color: 'rgba(13, 115, 119, 0.1)' } }
                    }
                }
            });
            
            // Latency by host
            if (latencyChart) latencyChart.destroy();
            const latencyCtx = document.getElementById('latencyChart').getContext('2d');
            const latencyHosts = Object.keys(data.latency_by_host || {});
            latencyChart = new Chart(latencyCtx, {
                type: 'bar',
                data: {
                    labels: latencyHosts,
                    datasets: [{
                        label: 'Avg Latency (ms)',
                        data: latencyHosts.map(h => data.latency_by_host[h].avg_latency_ms || 0),
                        backgroundColor: 'rgba(13, 115, 119, 0.8)',
                        borderColor: '#14b8a6',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#e0e0e0' } } },
                    scales: {
                        y: { ticks: { color: '#888' }, grid: { color: 'rgba(13, 115, 119, 0.1)' } },
                        x: { ticks: { color: '#888' }, grid: { color: 'rgba(13, 115, 119, 0.1)' } }
                    }
                }
            });
        }
        
        function updateTables(data) {
            // Inferences table
            const infBody = document.getElementById('inferencesBody');
            infBody.innerHTML = '';
            (data.recent_inferences || []).slice(-10).reverse().forEach(inf => {
                const status = inf.errors && inf.errors.length > 0 ? 'error' : 'success';
                const html = `
                    <tr>
                        <td>${inf.inference_id}</td>
                        <td>${inf.model_name}</td>
                        <td>${(inf.total_duration_ms || 0).toFixed(1)}</td>
                        <td>${(inf.tokens_per_second || 0).toFixed(2)}</td>
                        <td>${inf.total_tokens}</td>
                        <td><span class="badge badge-${status}">${status.toUpperCase()}</span></td>
                    </tr>
                `;
                infBody.innerHTML += html;
            });
            
            // Routing table
            const routingBody = document.getElementById('routingBody');
            routingBody.innerHTML = '';
            (data.recent_routing || []).slice(-20).reverse().forEach(route => {
                const html = `
                    <tr>
                        <td>${new Date(route.timestamp * 1000).toLocaleTimeString()}</td>
                        <td>${route.chunk_index}</td>
                        <td><strong>${route.selected_host}</strong></td>
                        <td>${route.candidate_hosts.join(', ')}</td>
                        <td>${route.decision_latency_ms.toFixed(2)}</td>
                        <td>${route.method}</td>
                    </tr>
                `;
                routingBody.innerHTML += html;
            });
        }
        
        function formatUptime(seconds) {
            const hrs = Math.floor(seconds / 3600);
            const mins = Math.floor((seconds % 3600) / 60);
            return `${hrs}h ${mins}m`;
        }
        
        function showAlert(message, type) {
            const alerts = document.getElementById('alerts');
            const html = `<div class="alert alert-${type}">${message}</div>`;
            alerts.innerHTML += html;
            setTimeout(() => alerts.innerHTML = '', 5000);
        }
        
        function downloadReport() {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => {
                    const json = JSON.stringify(data, null, 2);
                    const blob = new Blob([json], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `kai-metrics-${Date.now()}.json`;
                    a.click();
                });
        }
        
        // Update dashboard every 5 seconds
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """Serve the dashboard HTML."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/metrics')
def get_metrics():
    """API endpoint returning comprehensive metrics."""
    collector = get_default_collector()
    summary = collector.get_summary(time_window_seconds=300)
    
    # Add recent history
    summary['recent_routing'] = collector.get_routing_history(limit=50)
    summary['recent_inferences'] = collector.get_inference_history(limit=20)
    
    return jsonify(summary)


@app.route('/api/export', methods=['GET'])
def export_metrics():
    """Export metrics as JSON file."""
    collector = get_default_collector()
    summary = collector.get_summary(time_window_seconds=int(request.args.get('window', 300)))
    return jsonify(summary)


def run_dashboard(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run the dashboard server."""
    logging.basicConfig(level=logging.INFO)
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    print("=" * 60)
    print("KAI Performance Dashboard")
    print("=" * 60)
    print()
    print("Starting dashboard on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print()
    
    run_dashboard(host='0.0.0.0', port=5000, debug=False)
