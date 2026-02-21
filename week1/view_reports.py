"""Script to view and summarize detection reports."""

import json
from pathlib import Path
from typing import List, Dict, Any

def view_reports(limit: int = 10, matched_only: bool = False):
    """View detection reports with filtering options.
    
    Args:
        limit: Number of latest reports to show
        matched_only: Only show reports with matches
    """
    reports_dir = Path("reports")
    all_reports = sorted(reports_dir.glob("report_*.json"), reverse=True)
    
    if not all_reports:
        print("No reports found!")
        return
    
    print(f"\n{'='*80}")
    print(f"Total Reports: {len(all_reports)}")
    print(f"{'='*80}\n")
    
    count = 0
    for report_file in all_reports:
        if count >= limit:
            break
            
        try:
            with open(report_file) as f:
                data = json.load(f)
            
            matched = data.get("comparison_result", {}).get("matched", False)
            if matched_only and not matched:
                continue
            
            score = data.get("comparison_result", {}).get("score", 0)
            name = data.get("comparison_result", {}).get("name", "Unknown")
            timestamp = data.get("timestamp", "N/A")
            face_path = data.get("face_path", "N/A")
            
            status = "MATCH" if matched else "NO MATCH"
            color = "\033[92m" if matched else "\033[91m"  # Green/Red
            reset = "\033[0m"
            
            print(f"{color}[{status}]{reset} {report_file.name}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Face: {face_path}")
            print(f"  Score: {score:.4f}")
            if name != "Unknown":
                print(f"  Matched: {name}")
            print()
            
            count += 1
        except Exception as e:
            print(f"Error reading {report_file.name}: {e}\n")

def view_matched_reports():
    """View only reports with matches."""
    view_reports(matched_only=True)

def export_report_summary(output_file: str = "report_summary.txt"):
    """Export a summary of all reports to a text file."""
    reports_dir = Path("reports")
    all_reports = list(reports_dir.glob("report_*.json"))
    
    matched_count = 0
    no_match_count = 0
    
    with open(output_file, "w") as f:
        f.write("DETECTION REPORT SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Reports: {len(all_reports)}\n\n")
        
        for report_file in sorted(all_reports, reverse=True):
            try:
                with open(report_file) as rf:
                    data = json.load(rf)
                
                matched = data.get("comparison_result", {}).get("matched", False)
                score = data.get("comparison_result", {}).get("score", 0)
                name = data.get("comparison_result", {}).get("name", "Unknown")
                timestamp = data.get("timestamp", "N/A")
                
                if matched:
                    matched_count += 1
                    status = "MATCH"
                else:
                    no_match_count += 1
                    status = "NO MATCH"
                
                f.write(f"{status} | {timestamp} | Score: {score:.4f}")
                if name != "Unknown":
                    f.write(f" | {name}")
                f.write("\n")
            except Exception as e:
                f.write(f"Error reading {report_file.name}: {e}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"Matched: {matched_count}\n")
        f.write(f"No Match: {no_match_count}\n")
    
    print(f"Report summary exported to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "matched":
            view_matched_reports()
        elif sys.argv[1] == "summary":
            export_report_summary()
        else:
            limit = int(sys.argv[1]) if sys.argv[1].isdigit() else 10
            view_reports(limit=limit)
    else:
        view_reports()
