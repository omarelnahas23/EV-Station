#!/usr/bin/env python3
"""
Monitoring Setup Script for EV Charging LLM Pipeline

This script sets up comprehensive monitoring including:
- Prometheus configuration
- Grafana dashboards
- Alert rules
- Health check endpoints
"""

import os
import json
import yaml
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_prometheus_config():
    """Create Prometheus configuration file."""
    config = {
        'global': {
            'scrape_interval': '15s',
            'evaluation_interval': '15s'
        },
        'rule_files': [
            'alert_rules.yml'
        ],
        'scrape_configs': [
            {
                'job_name': 'ev-llm-api',
                'static_configs': [
                    {
                        'targets': ['ev-llm-api:8000']
                    }
                ],
                'metrics_path': '/metrics',
                'scrape_interval': '10s'
            },
            {
                'job_name': 'redis',
                'static_configs': [
                    {
                        'targets': ['redis:6379']
                    }
                ]
            },
            {
                'job_name': 'postgres',
                'static_configs': [
                    {
                        'targets': ['postgres:5432']
                    }
                ]
            }
        ],
        'alerting': {
            'alertmanagers': [
                {
                    'static_configs': [
                        {
                            'targets': ['alertmanager:9093']
                        }
                    ]
                }
            ]
        }
    }
    
    os.makedirs('monitoring', exist_ok=True)
    with open('monitoring/prometheus.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info("✅ Prometheus config created")


def create_alert_rules():
    """Create Prometheus alert rules."""
    rules = {
        'groups': [
            {
                'name': 'ev_llm_api_alerts',
                'rules': [
                    {
                        'alert': 'HighErrorRate',
                        'expr': 'rate(http_requests_total{status=~"5.."}[5m]) > 0.05',
                        'for': '5m',
                        'labels': {
                            'severity': 'critical'
                        },
                        'annotations': {
                            'summary': 'High error rate detected',
                            'description': 'API error rate is {{ $value }} which is above the threshold of 5%'
                        }
                    },
                    {
                        'alert': 'HighLatency',
                        'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2',
                        'for': '10m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'High response latency',
                            'description': '95th percentile latency is {{ $value }}s'
                        }
                    },
                    {
                        'alert': 'LowModelPerformance',
                        'expr': 'model_bleu_score < 0.2',
                        'for': '15m',
                        'labels': {
                            'severity': 'warning'
                        },
                        'annotations': {
                            'summary': 'Model performance degraded',
                            'description': 'BLEU score dropped to {{ $value }}'
                        }
                    },
                    {
                        'alert': 'APIDown',
                        'expr': 'up{job="ev-llm-api"} == 0',
                        'for': '1m',
                        'labels': {
                            'severity': 'critical'
                        },
                        'annotations': {
                            'summary': 'EV LLM API is down',
                            'description': 'API has been down for more than 1 minute'
                        }
                    }
                ]
            }
        ]
    }
    
    with open('monitoring/alert_rules.yml', 'w') as f:
        yaml.dump(rules, f, default_flow_style=False)
    
    logger.info("✅ Alert rules created")


def create_grafana_dashboard():
    """Create Grafana dashboard for EV LLM monitoring."""
    dashboard = {
        'dashboard': {
            'id': None,
            'title': 'EV Charging LLM Pipeline',
            'tags': ['llm', 'ev-charging', 'api'],
            'timezone': 'browser',
            'panels': [
                {
                    'id': 1,
                    'title': 'Request Rate',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'rate(http_requests_total[5m])',
                            'legendFormat': 'Requests/sec'
                        }
                    ],
                    'yAxes': [
                        {
                            'label': 'Requests/sec'
                        }
                    ],
                    'gridPos': {
                        'h': 8,
                        'w': 12,
                        'x': 0,
                        'y': 0
                    }
                },
                {
                    'id': 2,
                    'title': 'Response Time',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))',
                            'legendFormat': '95th percentile'
                        },
                        {
                            'expr': 'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))',
                            'legendFormat': '50th percentile'
                        }
                    ],
                    'yAxes': [
                        {
                            'label': 'Seconds'
                        }
                    ],
                    'gridPos': {
                        'h': 8,
                        'w': 12,
                        'x': 12,
                        'y': 0
                    }
                },
                {
                    'id': 3,
                    'title': 'Error Rate',
                    'type': 'singlestat',
                    'targets': [
                        {
                            'expr': 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])',
                            'legendFormat': 'Error Rate'
                        }
                    ],
                    'valueName': 'current',
                    'format': 'percentunit',
                    'thresholds': '0.01,0.05',
                    'colorBackground': True,
                    'gridPos': {
                        'h': 8,
                        'w': 6,
                        'x': 0,
                        'y': 8
                    }
                },
                {
                    'id': 4,
                    'title': 'Model Performance',
                    'type': 'singlestat',
                    'targets': [
                        {
                            'expr': 'model_bleu_score',
                            'legendFormat': 'BLEU Score'
                        }
                    ],
                    'valueName': 'current',
                    'format': 'none',
                    'decimals': 3,
                    'thresholds': '0.15,0.2',
                    'colorBackground': True,
                    'gridPos': {
                        'h': 8,
                        'w': 6,
                        'x': 6,
                        'y': 8
                    }
                },
                {
                    'id': 5,
                    'title': 'Memory Usage',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'process_resident_memory_bytes / 1024 / 1024',
                            'legendFormat': 'Memory (MB)'
                        }
                    ],
                    'yAxes': [
                        {
                            'label': 'MB'
                        }
                    ],
                    'gridPos': {
                        'h': 8,
                        'w': 12,
                        'x': 12,
                        'y': 8
                    }
                },
                {
                    'id': 6,
                    'title': 'Top Query Types',
                    'type': 'piechart',
                    'targets': [
                        {
                            'expr': 'topk(5, sum by (query_type) (rate(llm_queries_total[5m])))',
                            'legendFormat': '{{ query_type }}'
                        }
                    ],
                    'gridPos': {
                        'h': 8,
                        'w': 12,
                        'x': 0,
                        'y': 16
                    }
                },
                {
                    'id': 7,
                    'title': 'Model Inference Time',
                    'type': 'graph',
                    'targets': [
                        {
                            'expr': 'model_inference_duration_seconds',
                            'legendFormat': 'Inference Time'
                        }
                    ],
                    'yAxes': [
                        {
                            'label': 'Seconds'
                        }
                    ],
                    'gridPos': {
                        'h': 8,
                        'w': 12,
                        'x': 12,
                        'y': 16
                    }
                }
            ],
            'time': {
                'from': 'now-1h',
                'to': 'now'
            },
            'refresh': '10s'
        }
    }
    
    os.makedirs('monitoring/grafana/dashboards', exist_ok=True)
    with open('monitoring/grafana/dashboards/ev-llm-dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    logger.info("✅ Grafana dashboard created")


def create_grafana_provisioning():
    """Create Grafana provisioning configuration."""
    # Datasource provisioning
    datasource_config = {
        'apiVersion': 1,
        'datasources': [
            {
                'name': 'Prometheus',
                'type': 'prometheus',
                'access': 'proxy',
                'url': 'http://prometheus:9090',
                'isDefault': True,
                'editable': True
            }
        ]
    }
    
    os.makedirs('monitoring/grafana/provisioning/datasources', exist_ok=True)
    with open('monitoring/grafana/provisioning/datasources/prometheus.yaml', 'w') as f:
        yaml.dump(datasource_config, f)
    
    # Dashboard provisioning
    dashboard_config = {
        'apiVersion': 1,
        'providers': [
            {
                'name': 'default',
                'orgId': 1,
                'folder': '',
                'type': 'file',
                'disableDeletion': False,
                'updateIntervalSeconds': 10,
                'allowUiUpdates': True,
                'options': {
                    'path': '/var/lib/grafana/dashboards'
                }
            }
        ]
    }
    
    os.makedirs('monitoring/grafana/provisioning/dashboards', exist_ok=True)
    with open('monitoring/grafana/provisioning/dashboards/dashboard.yaml', 'w') as f:
        yaml.dump(dashboard_config, f)
    
    logger.info("✅ Grafana provisioning configured")


def create_health_check_script():
    """Create health check script for the monitoring system."""
    health_check = '''#!/usr/bin/env python3
"""
Health Check Script for EV Charging LLM Pipeline
"""

import requests
import sys
import argparse
import time
from typing import Dict, Any


def check_api_health(endpoint: str) -> Dict[str, Any]:
    """Check API health endpoint."""
    try:
        response = requests.get(f"{endpoint}/health", timeout=10)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds(),
            "status_code": response.status_code,
            "details": response.json() if response.status_code == 200 else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_prometheus(endpoint: str) -> Dict[str, Any]:
    """Check Prometheus health."""
    try:
        response = requests.get(f"{endpoint}/-/healthy", timeout=10)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_grafana(endpoint: str) -> Dict[str, Any]:
    """Check Grafana health."""
    try:
        response = requests.get(f"{endpoint}/api/health", timeout=10)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Health check for EV LLM pipeline')
    parser.add_argument('--api-endpoint', default='http://localhost:8000')
    parser.add_argument('--prometheus-endpoint', default='http://localhost:9090')
    parser.add_argument('--grafana-endpoint', default='http://localhost:3000')
    parser.add_argument('--wait-for-healthy', action='store_true')
    parser.add_argument('--timeout', type=int, default=300)
    
    args = parser.parse_args()
    
    checks = {
        'API': lambda: check_api_health(args.api_endpoint),
        'Prometheus': lambda: check_prometheus(args.prometheus_endpoint),
        'Grafana': lambda: check_grafana(args.grafana_endpoint)
    }
    
    if args.wait_for_healthy:
        print("Waiting for services to become healthy...")
        start_time = time.time()
        
        while time.time() - start_time < args.timeout:
            all_healthy = True
            
            for name, check_func in checks.items():
                result = check_func()
                if result['status'] != 'healthy':
                    all_healthy = False
                    print(f"❌ {name}: {result['status']}")
                    break
                else:
                    print(f"✅ {name}: healthy")
            
            if all_healthy:
                print("🎉 All services are healthy!")
                sys.exit(0)
            
            print("Waiting 10 seconds before retry...")
            time.sleep(10)
        
        print("❌ Timeout waiting for services to become healthy")
        sys.exit(1)
    
    else:
        all_healthy = True
        
        for name, check_func in checks.items():
            result = check_func()
            status_emoji = "✅" if result['status'] == 'healthy' else "❌"
            
            print(f"{status_emoji} {name}: {result['status']}")
            
            if 'response_time' in result:
                print(f"   Response time: {result['response_time']:.3f}s")
            
            if 'error' in result:
                print(f"   Error: {result['error']}")
            
            if result['status'] != 'healthy':
                all_healthy = False
        
        sys.exit(0 if all_healthy else 1)


if __name__ == '__main__':
    main()
'''
    
    os.makedirs('scripts', exist_ok=True)
    with open('scripts/health_check.py', 'w') as f:
        f.write(health_check)
    
    # Make executable
    os.chmod('scripts/health_check.py', 0o755)
    
    logger.info("✅ Health check script created")


def main():
    parser = argparse.ArgumentParser(description='Setup monitoring for EV Charging LLM Pipeline')
    parser.add_argument('--all', action='store_true', help='Setup all monitoring components')
    parser.add_argument('--prometheus', action='store_true', help='Setup Prometheus configuration')
    parser.add_argument('--grafana', action='store_true', help='Setup Grafana dashboards')
    parser.add_argument('--alerts', action='store_true', help='Setup alert rules')
    parser.add_argument('--health-check', action='store_true', help='Create health check script')
    
    args = parser.parse_args()
    
    if args.all:
        args.prometheus = True
        args.grafana = True
        args.alerts = True
        args.health_check = True
    
    if not any([args.prometheus, args.grafana, args.alerts, args.health_check]):
        parser.print_help()
        return
    
    logger.info("🔧 Setting up monitoring components...")
    
    if args.prometheus:
        create_prometheus_config()
    
    if args.alerts:
        create_alert_rules()
    
    if args.grafana:
        create_grafana_dashboard()
        create_grafana_provisioning()
    
    if args.health_check:
        create_health_check_script()
    
    logger.info("✅ Monitoring setup complete!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start monitoring stack: docker-compose --profile monitoring up -d")
    logger.info("2. Access Grafana: http://localhost:3000 (admin/admin)")
    logger.info("3. Access Prometheus: http://localhost:9090")
    logger.info("4. Run health check: python scripts/health_check.py")


if __name__ == '__main__':
    main() 