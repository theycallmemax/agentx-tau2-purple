#!/usr/bin/env python3
"""
Analyze test results from the latest run
Usage: python analyze_results.py
"""

import json
from pathlib import Path
from datetime import datetime

def find_latest_run():
    """Find the most recent test run directory"""
    runs_dir = Path("analysis/local_runs")
    if not runs_dir.exists():
        print("No runs found in analysis/local_runs/")
        return None
    
    # Find the most recent directory
    runs = sorted(runs_dir.glob("airline_*"))
    if not runs:
        print("No airline test runs found")
        return None
    
    return runs[-1]

def analyze_run(run_dir: Path):
    """Analyze a test run and print summary"""
    summary_file = run_dir / "summary.json"
    if not summary_file.exists():
        print(f"No summary.json found in {run_dir}")
        return
    
    with open(summary_file) as f:
        summary = json.load(f)
    
    print("=" * 60)
    print(f"ANALYSIS: {run_dir.name}")
    print(f"Timestamp: {datetime.fromtimestamp(run_dir.stat().st_mtime)}")
    print("=" * 60)
    print(f"\n📊 Overall Performance:")
    print(f"  Tasks total:     {summary.get('tasks_total', 'N/A')}")
    print(f"  Tasks completed: {summary.get('tasks_completed', 'N/A')}")
    print(f"  Tasks passed:    {summary.get('tasks_passed', 'N/A')}")
    print(f"  Score:           {summary.get('score', 0) * 100:.1f}%")
    print(f"  Total steps:     {summary.get('total_steps', 'N/A')}")
    print(f"  Avg steps/task:  {summary.get('average_steps_per_task', 'N/A'):.2f}")
    
    # Analyze termination reasons
    results = summary.get('results', [])
    termination_reasons = {}
    for result in results:
        reason = result.get('termination_reason', 'UNKNOWN')
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
    
    print(f"\n🔍 Termination Reasons:")
    for reason, count in sorted(termination_reasons.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100 if results else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Check for loop patterns
    loop_count = 0
    for result in results:
        msg_tail = result.get('message_tail', [])
        tool_calls = []
        for msg in msg_tail:
            if msg.get('tool_calls'):
                for tc in msg['tool_calls']:
                    tool_key = f"{tc['name']}:{json.dumps(tc.get('arguments', {}), sort_keys=True)[:100]}"
                    tool_calls.append(tool_key)
        
        # Check for repetitions
        if len(tool_calls) > 2:
            if len(set(tool_calls[-3:])) == 1:
                loop_count += 1
    
    if loop_count > 0:
        print(f"\n⚠️  Loop Patterns Detected:")
        print(f"  Tasks with tool call loops: {loop_count}/{len(results)}")
    
    print("\n" + "=" * 60)
    
    return summary

def compare_runs():
    """Compare multiple runs if available"""
    runs_dir = Path("analysis/local_runs")
    runs = sorted(runs_dir.glob("airline_*"))
    
    if len(runs) < 2:
        print("Need at least 2 runs to compare")
        return
    
    print("\n" + "=" * 60)
    print("COMPARISON OF RECENT RUNS")
    print("=" * 60)
    
    for run_dir in runs[-3:]:  # Last 3 runs
        summary_file = run_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            
            print(f"\n{run_dir.name}:")
            print(f"  Score: {summary.get('score', 0) * 100:.1f}% | "
                  f"Passed: {summary.get('tasks_passed', 0)}/{summary.get('tasks_total', 0)} | "
                  f"Avg Steps: {summary.get('average_steps_per_task', 0):.1f}")

if __name__ == "__main__":
    latest_run = find_latest_run()
    if latest_run:
        analyze_run(latest_run)
        print()
        compare_runs()
