#!/usr/bin/env python3
"""
Sensitive Data Scanner for EV Charging LLM Pipeline

This script scans training data for potential PII, sensitive information,
and compliance violations before model training.
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sensitive data patterns
SENSITIVE_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'api_key': r'[Aa][Pp][Ii][-_]?[Kk][Ee][Yy][-_=]?[\'"]?[A-Za-z0-9]{20,}[\'"]?',
    'token': r'[Tt][Oo][Kk][Ee][Nn][-_=]?[\'"]?[A-Za-z0-9]{20,}[\'"]?',
    'password': r'[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][-_=]?[\'"]?[A-Za-z0-9!@#$%^&*]{8,}[\'"]?'
}

# EV-specific sensitive patterns
EV_SENSITIVE_PATTERNS = {
    'vin': r'\b[A-HJ-NPR-Z0-9]{17}\b',  # Vehicle Identification Number
    'license_plate': r'\b[A-Z]{1,3}[-\s]?\d{1,4}[-\s]?[A-Z]{0,3}\b',
    'charging_session_id': r'[Cc][Hh][Aa][Rr][Gg][Ee][-_]?[Ss][Ee][Ss][Ss][Ii][Oo][Nn][-_=]?[\'"]?[A-Za-z0-9]{10,}[\'"]?',
    'customer_id': r'[Cc][Uu][Ss][Tt][Oo][Mm][Ee][Rr][-_]?[Ii][Dd][-_=]?[\'"]?[A-Za-z0-9]{5,}[\'"]?'
}

# Combine all patterns
ALL_PATTERNS = {**SENSITIVE_PATTERNS, **EV_SENSITIVE_PATTERNS}


def scan_text_for_sensitive_data(text: str) -> Dict[str, List[str]]:
    """Scan text for sensitive data patterns."""
    findings = {}
    
    for pattern_name, pattern in ALL_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            findings[pattern_name] = matches
    
    return findings


def scan_json_data(data: Any, path: str = "") -> List[Dict[str, Any]]:
    """Recursively scan JSON data for sensitive information."""
    findings = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # Check if key name suggests sensitive data
            sensitive_keys = ['password', 'token', 'key', 'secret', 'api_key', 'auth', 'credential']
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                findings.append({
                    'type': 'sensitive_key',
                    'path': current_path,
                    'key': key,
                    'issue': f"Potentially sensitive key name: {key}"
                })
            
            # Recursively check value
            findings.extend(scan_json_data(value, current_path))
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]" if path else f"[{i}]"
            findings.extend(scan_json_data(item, current_path))
    
    elif isinstance(data, str):
        # Scan string content for sensitive patterns
        sensitive_data = scan_text_for_sensitive_data(data)
        for pattern_type, matches in sensitive_data.items():
            for match in matches:
                findings.append({
                    'type': 'sensitive_data',
                    'path': path,
                    'pattern_type': pattern_type,
                    'match': match,
                    'issue': f"Found {pattern_type}: {match}"
                })
    
    return findings


def scan_file(file_path: Path) -> Dict[str, Any]:
    """Scan a single file for sensitive data."""
    findings = {
        'file': str(file_path),
        'issues': [],
        'risk_level': 'low'
    }
    
    try:
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                findings['issues'] = scan_json_data(data)
        
        else:
            # Scan as text file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                sensitive_data = scan_text_for_sensitive_data(content)
                
                for pattern_type, matches in sensitive_data.items():
                    for match in matches:
                        findings['issues'].append({
                            'type': 'sensitive_data',
                            'pattern_type': pattern_type,
                            'match': match,
                            'issue': f"Found {pattern_type}: {match}"
                        })
        
        # Determine risk level
        if findings['issues']:
            high_risk_patterns = ['ssn', 'credit_card', 'password', 'api_key', 'token']
            if any(issue.get('pattern_type') in high_risk_patterns for issue in findings['issues']):
                findings['risk_level'] = 'high'
            else:
                findings['risk_level'] = 'medium'
    
    except Exception as e:
        findings['issues'].append({
            'type': 'scan_error',
            'issue': f"Error scanning file: {str(e)}"
        })
        findings['risk_level'] = 'unknown'
    
    return findings


def scan_directory(directory: Path, file_patterns: List[str] = None) -> Dict[str, Any]:
    """Scan all files in a directory for sensitive data."""
    if file_patterns is None:
        file_patterns = ['*.json', '*.txt', '*.md', '*.yaml', '*.yml', '*.py']
    
    results = {
        'directory': str(directory),
        'total_files_scanned': 0,
        'files_with_issues': 0,
        'total_issues': 0,
        'risk_summary': {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0},
        'file_results': []
    }
    
    for pattern in file_patterns:
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                file_result = scan_file(file_path)
                results['file_results'].append(file_result)
                results['total_files_scanned'] += 1
                
                if file_result['issues']:
                    results['files_with_issues'] += 1
                    results['total_issues'] += len(file_result['issues'])
                
                results['risk_summary'][file_result['risk_level']] += 1
    
    return results


def generate_report(scan_results: Dict[str, Any], output_file: str = None):
    """Generate a detailed security scan report."""
    report = f"""# Sensitive Data Scan Report

**Directory:** {scan_results['directory']}  
**Scan Date:** {scan_results.get('scan_date', 'Unknown')}  
**Total Files Scanned:** {scan_results['total_files_scanned']}  
**Files with Issues:** {scan_results['files_with_issues']}  
**Total Issues Found:** {scan_results['total_issues']}  

## Risk Summary

| Risk Level | Files |
|------------|-------|
| High | {scan_results['risk_summary']['high']} |
| Medium | {scan_results['risk_summary']['medium']} |
| Low | {scan_results['risk_summary']['low']} |
| Unknown | {scan_results['risk_summary']['unknown']} |

## Detailed Findings

"""
    
    for file_result in scan_results['file_results']:
        if file_result['issues']:
            report += f"\n### {file_result['file']} (Risk: {file_result['risk_level'].upper()})\n\n"
            
            for issue in file_result['issues']:
                if issue['type'] == 'sensitive_data':
                    report += f"- **{issue['pattern_type'].upper()}**: {issue['match']}\n"
                elif issue['type'] == 'sensitive_key':
                    report += f"- **SENSITIVE KEY**: {issue['key']} at path {issue['path']}\n"
                else:
                    report += f"- **{issue['type'].upper()}**: {issue['issue']}\n"
    
    if scan_results['total_issues'] == 0:
        report += "\n✅ **No sensitive data detected!**\n"
    
    report += f"""
## Recommendations

1. **Remove or Anonymize**: Replace any real sensitive data with synthetic alternatives
2. **Review Data Sources**: Ensure data collection processes exclude PII
3. **Implement Filtering**: Add automated PII detection to data processing pipeline
4. **Access Controls**: Limit access to training data containing any sensitive information
5. **Documentation**: Document any remaining data privacy considerations

## Compliance Notes

- **GDPR**: Ensure no EU personal data is included without proper consent
- **CCPA**: California residents' personal information should be excluded
- **Industry Standards**: Follow EV industry data privacy best practices
- **Internal Policies**: Comply with organizational data governance policies
"""
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"📄 Security scan report saved to: {output_file}")
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(description='Scan for sensitive data in training datasets')
    parser.add_argument('--directory', '-d', default='data', help='Directory to scan')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--patterns', nargs='+', help='File patterns to scan', 
                       default=['*.json', '*.txt', '*.md'])
    parser.add_argument('--fail-on-high-risk', action='store_true', 
                       help='Exit with error code if high-risk issues found')
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return 1
    
    logger.info(f"Scanning directory: {directory}")
    logger.info(f"File patterns: {args.patterns}")
    
    # Perform scan
    scan_results = scan_directory(directory, args.patterns)
    scan_results['scan_date'] = str(Path().cwd())
    
    # Generate report
    generate_report(scan_results, args.output)
    
    # Print summary
    logger.info(f"Scan complete:")
    logger.info(f"  Files scanned: {scan_results['total_files_scanned']}")
    logger.info(f"  Issues found: {scan_results['total_issues']}")
    logger.info(f"  High risk files: {scan_results['risk_summary']['high']}")
    
    # Check for failure conditions
    if args.fail_on_high_risk and scan_results['risk_summary']['high'] > 0:
        logger.error("❌ High-risk sensitive data detected! Pipeline should not proceed.")
        return 1
    
    if scan_results['total_issues'] > 0:
        logger.warning("⚠️  Potential sensitive data detected. Review findings before proceeding.")
    else:
        logger.info("✅ No sensitive data detected.")
    
    return 0


if __name__ == '__main__':
    exit(main()) 