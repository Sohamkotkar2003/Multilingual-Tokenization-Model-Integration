#!/usr/bin/env python3
"""
Test script for RL pipeline
Tests episode collection and cloud logging
"""

import subprocess
import json
import time
from pathlib import Path

def test_rl_pipeline():
    """Test the RL pipeline with sample episodes"""
    print("Testing RL Pipeline")
    print("=" * 40)
    
    # Test 1: Basic episode collection
    print("\n1. Testing basic episode collection...")
    try:
        result = subprocess.run([
            "python", "rl/collect.py",
            "--max_episodes", "3",
            "--out", "rl_runs/test_episodes.jsonl",
            "--env_name", "test-env"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   Basic collection: PASSED")
            
            # Check if episodes were created
            episodes_file = Path("rl_runs/test_episodes.jsonl")
            if episodes_file.exists():
                with open(episodes_file, 'r', encoding='utf-8') as f:
                    episodes = [json.loads(line) for line in f]
                print(f"   Episodes collected: {len(episodes)}")
                
                # Show sample episode
                if episodes:
                    sample = episodes[0]
                    print(f"   Sample episode:")
                    print(f"      Prompt: '{sample['prompt'][:50]}...'")
                    print(f"      Output: '{sample['output'][:50]}...'")
                    print(f"      Reward: {sample['reward']:.3f}")
                    print(f"      Latency: {sample['latency_s']:.2f}s")
            else:
                print("   No episodes file created")
        else:
            print(f"   Basic collection: FAILED")
            print(f"   Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("   Basic collection: TIMEOUT")
    except Exception as e:
        print(f"   Basic collection: ERROR ({e})")
    
    # Test 2: Multilingual prompts
    print("\n2. Testing multilingual prompts...")
    try:
        result = subprocess.run([
            "python", "rl/collect.py",
            "--max_episodes", "5",
            "--out", "rl_runs/multilingual_episodes.jsonl",
            "--env_name", "multilingual-test"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("   Multilingual collection: PASSED")
            
            # Check episodes
            episodes_file = Path("rl_runs/multilingual_episodes.jsonl")
            if episodes_file.exists():
                with open(episodes_file, 'r', encoding='utf-8') as f:
                    episodes = [json.loads(line) for line in f]
                
                print(f"   Episodes collected: {len(episodes)}")
                
                # Check for multilingual content
                multilingual_count = 0
                for episode in episodes:
                    if any(ord(char) > 127 for char in episode['output']):
                        multilingual_count += 1
                
                print(f"   Multilingual episodes: {multilingual_count}/{len(episodes)}")
                
                # Show rewards
                rewards = [ep['reward'] for ep in episodes]
                avg_reward = sum(rewards) / len(rewards)
                print(f"   Average reward: {avg_reward:.3f}")
            else:
                print("   No episodes file created")
        else:
            print(f"   Multilingual collection: FAILED")
            print(f"   Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("   Multilingual collection: TIMEOUT")
    except Exception as e:
        print(f"   Multilingual collection: ERROR ({e})")
    
    # Test 3: Custom prompt
    print("\n3. Testing custom prompt...")
    try:
        result = subprocess.run([
            "python", "rl/collect.py",
            "--max_episodes", "1",
            "--prompt", "Write a poem about AI in Hindi",
            "--out", "rl_runs/custom_episodes.jsonl",
            "--env_name", "custom-test"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("   Custom prompt: PASSED")
            
            # Check episode
            episodes_file = Path("rl_runs/custom_episodes.jsonl")
            if episodes_file.exists():
                with open(episodes_file, 'r', encoding='utf-8') as f:
                    episodes = [json.loads(line) for line in f]
                
                if episodes:
                    episode = episodes[0]
                    print(f"   Custom episode:")
                    print(f"      Prompt: '{episode['prompt']}'")
                    print(f"      Output: '{episode['output'][:100]}...'")
                    print(f"      Reward: {episode['reward']:.3f}")
            else:
                print("   No episodes file created")
        else:
            print(f"   Custom prompt: FAILED")
            print(f"   Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("   Custom prompt: TIMEOUT")
    except Exception as e:
        print(f"   Custom prompt: ERROR ({e})")
    
    print("\n" + "=" * 40)
    print("RL Pipeline Testing Complete!")
    print("\nGenerated files:")
    print("   - rl_runs/test_episodes.jsonl")
    print("   - rl_runs/multilingual_episodes.jsonl") 
    print("   - rl_runs/custom_episodes.jsonl")
    print("\nNext steps:")
    print("   1. Review generated episodes")
    print("   2. Configure cloud upload (S3, NAS)")
    print("   3. Integrate with API for real-time collection")

if __name__ == "__main__":
    test_rl_pipeline()
