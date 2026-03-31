import json
import statistics
from collections import defaultdict

# Load results
with open('reports/vqa_results.json') as f:
    results = json.load(f)

# Prepare data
tasks = defaultdict(list)
for r in results:
    tasks[r['task']].append(r)

# Create HTML dashboard
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VQA Pipeline Results Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .chart-section {
            margin-bottom: 40px;
        }
        .chart-title {
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }
        .chart-container {
            position: relative;
            height: 400px;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        thead {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .accent-green { color: #28a745; font-weight: bold; }
        .accent-red { color: #dc3545; font-weight: bold; }
        .accent-orange { color: #fd7e14; font-weight: bold; }
        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>VQA Pipeline Results Dashboard</h1>
        <p class="subtitle">NVIDIA Jetson Orin NX | 28 Images Evaluated | March 30, 2026</p>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall Accuracy</div>
                <div class="metric-value accent-orange">46.4%</div>
                <div class="metric-label">13/28 Correct</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg E2E Latency</div>
                <div class="metric-value">1008 ms</div>
                <div class="metric-label">Median: 616 ms</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Best Task</div>
                <div class="metric-value accent-green">Stairs</div>
                <div class="metric-label">66.7% Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Worst Task</div>
                <div class="metric-value accent-red">Crosswalk</div>
                <div class="metric-label">33.3% Accuracy</div>
            </div>
        </div>

        <div class="chart-section">
            <h2 class="chart-title">Performance Metrics</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="accuracyChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="latencyChart"></canvas>
                </div>
            </div>
        </div>

        <div class="chart-section">
            <h2 class="chart-title">Latency Breakdown by Stage</h2>
            <div class="chart-container" style="height: 300px;">
                <canvas id="stageChart"></canvas>
            </div>
        </div>

        <div class="chart-section">
            <h2 class="chart-title">Detailed Task Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Task</th>
                        <th>Correct</th>
                        <th>Total</th>
                        <th>Accuracy</th>
                        <th>Avg E2E (ms)</th>
                        <th>Median E2E (ms)</th>
                    </tr>
                </thead>
                <tbody>
"""

# Add task rows
for task, samples in sorted(tasks.items()):
    correct = sum(1 for s in samples if s['pred'] == s['gt'])
    accuracy = correct / len(samples) * 100 if len(samples) > 0 else 0
    e2e_times = [s['e2e_total_ms'] for s in samples]
    avg_e2e = statistics.mean(e2e_times)
    median_e2e = statistics.median(e2e_times)
    
    acc_class = 'accent-green' if accuracy > 50 else 'accent-red' if accuracy < 40 else 'accent-orange'
    html += f"""
                    <tr>
                        <td><strong>{task}</strong></td>
                        <td>{correct}</td>
                        <td>{len(samples)}</td>
                        <td class="{acc_class}">{accuracy:.1f}%</td>
                        <td>{avg_e2e:.1f}</td>
                        <td>{median_e2e:.1f}</td>
                    </tr>
"""

html += """
                </tbody>
            </table>
        </div>

        <div class="chart-section">
            <h2 class="chart-title">Per-Stage Latency Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Mean (ms)</th>
                        <th>Median (ms)</th>
                        <th>Min (ms)</th>
                        <th>Max (ms)</th>
                    </tr>
                </thead>
                <tbody>
"""

# Stage metrics
stages = {
    'Image Capture': [r['capture_ms'] for r in results],
    'Vision Encode': [r['encode_ms'] for r in results],
    'Compression': [r['compress_ms'] for r in results],
    'VLM Inference': [r['vlm_total_ms'] for r in results],
    'End-to-End': [r['e2e_total_ms'] for r in results]
}

for stage, times in stages.items():
    html += f"""
                    <tr>
                        <td><strong>{stage}</strong></td>
                        <td>{statistics.mean(times):.1f}</td>
                        <td>{statistics.median(times):.1f}</td>
                        <td>{min(times):.1f}</td>
                        <td>{max(times):.1f}</td>
                    </tr>
"""

html += """
                </tbody>
            </table>
        </div>

        <footer>
            <p><strong>Note:</strong> This dashboard was auto-generated from VQA pipeline results. Data includes 28 evaluation images across 3 tasks (crosswalk_signal, obstacles, stairs). Results saved to <code>reports/vqa_results.json</code>.</p>
        </footer>
    </div>

    <script>
        // Accuracy by Task
        new Chart(document.getElementById('accuracyChart'), {
            type: 'bar',
            data: {
                labels: ['Stairs', 'Obstacles', 'Crosswalk Signal'],
                datasets: [{
                    label: 'Accuracy (%)',
                    data: [66.7, 40.0, 33.3],
                    backgroundColor: ['#28a745', '#fd7e14', '#dc3545'],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Task Accuracy Comparison' }
                },
                scales: {
                    y: { max: 100, beginAtZero: true }
                }
            }
        });

        // Latency by Task
        new Chart(document.getElementById('latencyChart'), {
            type: 'bar',
            data: {
                labels: ['Stairs', 'Obstacles', 'Crosswalk Signal'],
                datasets: [{
                    label: 'E2E Latency (ms)',
                    data: [559.7, 1095.2, 1359.8],
                    backgroundColor: ['#667eea', '#764ba2', '#f093fb'],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'E2E Latency by Task' }
                },
                scales: { y: { beginAtZero: true } }
            }
        });

        // Stage Breakdown
        new Chart(document.getElementById('stageChart'), {
            type: 'bar',
            data: {
                labels: ['Capture', 'Encode', 'Compress', 'VLM', 'Total'],
                datasets: [{
                    label: 'Mean Latency (ms)',
                    data: [20.2, 141.5, 141.8, 852.7, 1008.1],
                    backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#ff6b6b', '#4ecdc4'],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: 'Pipeline Stage Latency' }
                },
                scales: { y: { beginAtZero: true } }
            }
        });
    </script>
</body>
</html>
"""

# Write file
with open('reports/vqa_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Dashboard created: reports/vqa_dashboard.html")
print("Open in browser to view interactive charts and metrics")
