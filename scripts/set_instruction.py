#!/usr/bin/env python3
"""
Simple script to set the natural language instruction for the VLA pipeline.
"""
import sys
import requests
import argparse

def set_instruction(instruction: str, host: str = "localhost", port: int = 8000):
    url = f"http://{host}:{port}/set_instruction"
    try:
        response = requests.post(url, params={"instruction": instruction})
        response.raise_for_status()
        print(f"✓ Instruction successfully updated to: '{instruction}'")
    except requests.exceptions.ConnectionError:
        print(f"✗ Error: Could not connect to Pipeline Server at {url}")
        print("  Make sure services/pipeline/server.py is running.")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set VLA Pipeline instruction")
    parser.add_argument("instruction", type=str, help="The natural language instruction for the robot")
    parser.add_argument("--host", type=str, default="localhost", help="Pipeline server host")
    parser.add_argument("--port", type=int, default=8000, help="Pipeline server port")
    
    args = parser.parse_args()
    set_instruction(args.instruction, args.host, args.port)
