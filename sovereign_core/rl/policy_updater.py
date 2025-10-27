#!/usr/bin/env python3
"""
RL Self-Improvement Loop

This module implements the reinforcement learning self-improvement loop that:
- Accepts { prompt, output, reward } feedback
- Updates local adapter delta or policy table
- Runs periodic reward-based adjustments
- Syncs logs to s3://bhiv/rl_feedback/sovereign_core/

Author: Soham Kotkar
"""

import json
import time
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class PolicyEntry:
    """Entry in the policy table"""
    state: str  # Encoded state (prompt + context)
    action: str  # Action taken (model output)
    reward: float
    count: int
    last_updated: float

class RLPolicyUpdater:
    """
    RL Self-Improvement Loop
    
    Maintains a policy table (Q-table or bandit style) and updates it
    based on user feedback to improve model performance over time.
    """
    
    def __init__(self):
        self.policy_table = {}  # state -> PolicyEntry
        self.adapter_deltas = {}  # Model parameter deltas
        self.reward_history = []
        self.update_frequency = 10  # Update policy every N feedbacks
        self.learning_rate = 0.1
        self.exploration_rate = 0.1
        self.initialized = False
        
        # S3 sync configuration
        self.s3_bucket = "bhiv"
        self.s3_prefix = "rl_feedback/sovereign_core/"
        
    async def initialize(self):
        """Initialize the RL policy updater"""
        try:
            logger.info("Initializing RL Self-Improvement Loop...")
            
            # Load existing policy if available
            await self._load_policy()
            
            # Start background tasks
            asyncio.create_task(self._periodic_policy_update())
            asyncio.create_task(self._sync_to_s3())
            
            self.initialized = True
            logger.info("✅ RL Policy Updater initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RL policy updater: {e}")
            raise
    
    async def _load_policy(self):
        """Load existing policy from storage"""
        try:
            policy_file = Path("data/rl_policy.json")
            if policy_file.exists():
                async with aiofiles.open(policy_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    # Restore policy table
                    for state, entry_data in data.get("policy_table", {}).items():
                        self.policy_table[state] = PolicyEntry(**entry_data)
                    
                    # Restore adapter deltas
                    self.adapter_deltas = data.get("adapter_deltas", {})
                    
                    logger.info(f"Loaded policy with {len(self.policy_table)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            self.policy_table = {}
            self.adapter_deltas = {}
    
    async def _save_policy(self):
        """Save policy to storage"""
        try:
            policy_file = Path("data/rl_policy.json")
            policy_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "policy_table": {
                    state: asdict(entry) 
                    for state, entry in self.policy_table.items()
                },
                "adapter_deltas": self.adapter_deltas,
                "last_updated": time.time()
            }
            
            async with aiofiles.open(policy_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, ensure_ascii=False, indent=2))
            
            logger.info(f"Saved policy with {len(self.policy_table)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save policy: {e}")
    
    async def process_feedback(self, prompt: str, output: str, reward: float,
                             user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process RL feedback and update policy
        
        Args:
            prompt: User prompt
            output: Model output
            reward: Reward score (0-1)
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Encode state from prompt and context
            state = self._encode_state(prompt, user_id, session_id)
            
            # Update policy table
            policy_updated = await self._update_policy_entry(state, output, reward)
            
            # Log reward
            reward_logged = await self._log_reward(prompt, output, reward, user_id, session_id)
            
            # Update adapter deltas if needed
            adapter_updated = await self._update_adapter_deltas(state, output, reward)
            
            # Check if we should sync to S3
            s3_synced = False
            if len(self.reward_history) % self.update_frequency == 0:
                s3_synced = await self._sync_to_s3()
            
            return {
                "policy_updated": policy_updated,
                "reward_logged": reward_logged,
                "adapter_updated": adapter_updated,
                "s3_synced": s3_synced,
                "policy_size": len(self.policy_table),
                "reward_count": len(self.reward_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to process RL feedback: {e}")
            return {
                "policy_updated": False,
                "reward_logged": False,
                "adapter_updated": False,
                "s3_synced": False,
                "error": str(e)
            }
    
    def _encode_state(self, prompt: str, user_id: Optional[str], session_id: Optional[str]) -> str:
        """Encode state from prompt and context"""
        # Simple state encoding - in practice, you might use more sophisticated methods
        context_parts = []
        
        if user_id:
            context_parts.append(f"user:{user_id}")
        if session_id:
            context_parts.append(f"session:{session_id}")
        
        # Use first 100 characters of prompt as state identifier
        prompt_hash = hash(prompt[:100]) % 10000
        
        state = f"{prompt_hash}:{':'.join(context_parts)}"
        return state
    
    async def _update_policy_entry(self, state: str, action: str, reward: float) -> bool:
        """Update policy table entry"""
        try:
            if state in self.policy_table:
                # Update existing entry
                entry = self.policy_table[state]
                
                # Q-learning update
                old_reward = entry.reward
                entry.reward = old_reward + self.learning_rate * (reward - old_reward)
                entry.count += 1
                entry.last_updated = time.time()
                
                # Update action if this one is better
                if reward > old_reward:
                    entry.action = action
                
            else:
                # Create new entry
                self.policy_table[state] = PolicyEntry(
                    state=state,
                    action=action,
                    reward=reward,
                    count=1,
                    last_updated=time.time()
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update policy entry: {e}")
            return False
    
    async def _log_reward(self, prompt: str, output: str, reward: float,
                         user_id: Optional[str], session_id: Optional[str]) -> bool:
        """Log reward to history"""
        try:
            reward_entry = {
                "timestamp": time.time(),
                "prompt": prompt[:200],  # Truncate for storage
                "output": output[:200],
                "reward": reward,
                "user_id": user_id,
                "session_id": session_id
            }
            
            self.reward_history.append(reward_entry)
            
            # Keep only last 1000 entries
            if len(self.reward_history) > 1000:
                self.reward_history = self.reward_history[-1000:]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log reward: {e}")
            return False
    
    async def _update_adapter_deltas(self, state: str, action: str, reward: float) -> bool:
        """Update adapter deltas based on reward"""
        try:
            # Simple delta update - in practice, you might use more sophisticated methods
            if reward > 0.7:  # High reward
                # Positive delta
                delta_key = f"{state}:positive"
                if delta_key not in self.adapter_deltas:
                    self.adapter_deltas[delta_key] = 0.0
                self.adapter_deltas[delta_key] += 0.01
                
            elif reward < 0.3:  # Low reward
                # Negative delta
                delta_key = f"{state}:negative"
                if delta_key not in self.adapter_deltas:
                    self.adapter_deltas[delta_key] = 0.0
                self.adapter_deltas[delta_key] -= 0.01
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update adapter deltas: {e}")
            return False
    
    async def _periodic_policy_update(self):
        """Background task for periodic policy updates"""
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                if self.policy_table:
                    # Perform policy optimization
                    await self._optimize_policy()
                    
                    # Save updated policy
                    await self._save_policy()
                
            except Exception as e:
                logger.error(f"Error in periodic policy update: {e}")
                await asyncio.sleep(300)
    
    async def _optimize_policy(self):
        """Optimize policy based on recent rewards"""
        try:
            if not self.reward_history:
                return
            
            # Calculate average reward over last 100 entries
            recent_rewards = [entry["reward"] for entry in self.reward_history[-100:]]
            avg_reward = np.mean(recent_rewards)
            
            # Adjust learning rate based on performance
            if avg_reward > 0.7:
                self.learning_rate = min(0.2, self.learning_rate * 1.1)
            elif avg_reward < 0.3:
                self.learning_rate = max(0.01, self.learning_rate * 0.9)
            
            # Adjust exploration rate
            if avg_reward > 0.8:
                self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
            elif avg_reward < 0.2:
                self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
            
            logger.info(f"Policy optimized: avg_reward={avg_reward:.3f}, lr={self.learning_rate:.3f}, exp={self.exploration_rate:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to optimize policy: {e}")
    
    async def _sync_to_s3(self) -> bool:
        """Sync policy and rewards to S3"""
        try:
            # Create sync data
            sync_data = {
                "timestamp": time.time(),
                "policy_table": {
                    state: asdict(entry) 
                    for state, entry in self.policy_table.items()
                },
                "adapter_deltas": self.adapter_deltas,
                "reward_history": self.reward_history[-100:],  # Last 100 rewards
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate
            }
            
            # Save to local file first
            sync_file = Path("data/rl_sync.json")
            sync_file.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(sync_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(sync_data, ensure_ascii=False, indent=2))
            
            # In a real implementation, you would upload to S3 here
            # For now, we'll simulate the upload
            logger.info(f"Simulating S3 sync to {self.s3_bucket}/{self.s3_prefix}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync to S3: {e}")
            return False
    
    async def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about the policy"""
        try:
            if not self.reward_history:
                avg_reward = 0.0
                recent_avg = 0.0
            else:
                avg_reward = np.mean([entry["reward"] for entry in self.reward_history])
                recent_avg = np.mean([entry["reward"] for entry in self.reward_history[-50:]])
            
            return {
                "initialized": self.initialized,
                "policy_entries": len(self.policy_table),
                "reward_history_size": len(self.reward_history),
                "average_reward": avg_reward,
                "recent_average_reward": recent_avg,
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "adapter_deltas": len(self.adapter_deltas),
                "last_sync": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get policy stats: {e}")
            return {"error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return await self.get_policy_stats()
