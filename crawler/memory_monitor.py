"""
Memory monitoring and fail-safe mechanisms for the YouTube crawler.

This module provides the MemoryMonitor class that tracks system memory usage
and implements safety mechanisms to prevent system crashes during processing.
"""

import logging
import psutil
import gc
import torch
from typing import Dict

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Memory monitoring and fail-safe mechanisms"""
    
    def __init__(self, memory_limit_percent: float = 85.0, critical_limit_percent: float = 95.0):
        """
        Initialize memory monitor
        
        Args:
            memory_limit_percent: Warning threshold (% of total RAM)
            critical_limit_percent: Critical threshold - stop processing (% of total RAM)
        """
        self.memory_limit_percent = memory_limit_percent
        self.critical_limit_percent = critical_limit_percent
        self.total_memory = psutil.virtual_memory().total
        self.memory_limit_bytes = self.total_memory * (memory_limit_percent / 100)
        self.critical_limit_bytes = self.total_memory * (critical_limit_percent / 100)
        
        logger.info(f"🛡️  Memory Monitor initialized:")
        logger.info(f"   • Total RAM: {self.total_memory / (1024**3):.1f} GB")
        logger.info(f"   • Warning threshold: {memory_limit_percent}% ({self.memory_limit_bytes / (1024**3):.1f} GB)")
        logger.info(f"   • Critical threshold: {critical_limit_percent}% ({self.critical_limit_bytes / (1024**3):.1f} GB)")
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'process_memory': process.memory_info().rss,
            'process_percent': process.memory_percent()
        }
    
    def check_memory_status(self) -> str:
        """
        Check current memory status
        
        Returns:
            'safe', 'warning', or 'critical'
        """
        memory = psutil.virtual_memory()
        
        if memory.used >= self.critical_limit_bytes:
            return 'critical'
        elif memory.used >= self.memory_limit_bytes:
            return 'warning'
        else:
            return 'safe'
    
    def log_memory_status(self, operation: str = ""):
        """Log current memory usage"""
        stats = self.get_memory_usage()
        status = self.check_memory_status()
        
        status_emoji = {
            'safe': '✅',
            'warning': '⚠️',
            'critical': '🚨'
        }
        
        logger.info(f"{status_emoji[status]} Memory Status {f'({operation})' if operation else ''}: "
                   f"{stats['percent']:.1f}% used "
                   f"({stats['used'] / (1024**3):.1f}/{stats['total'] / (1024**3):.1f} GB), "
                   f"Process: {stats['process_percent']:.1f}% "
                   f"({stats['process_memory'] / (1024**3):.2f} GB)")
    
    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        logger.info("🧹 Forcing memory cleanup...")
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("   • Cleared GPU cache")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"   • Garbage collected {collected} objects")
        
        # Log memory status after cleanup
        self.log_memory_status("after cleanup")
    
    def check_and_handle_memory(self, operation: str = "") -> bool:
        """
        Check memory and handle according to status
        
        Returns:
            True if safe to continue, False if should stop
        """
        status = self.check_memory_status()
        
        if status == 'critical':
            logger.error(f"🚨 CRITICAL MEMORY USAGE - Stopping operation!")
            self.log_memory_status(operation)
            self.force_cleanup()
            
            # Check again after cleanup
            if self.check_memory_status() == 'critical':
                logger.error("🚨 Memory still critical after cleanup. Aborting to prevent system crash!")
                return False
            else:
                logger.warning("✅ Memory recovered after cleanup. Continuing...")
                return True
                
        elif status == 'warning':
            logger.warning(f"⚠️  High memory usage detected!")
            self.log_memory_status(operation)
            self.force_cleanup()
            return True
            
        else:
            # Only log memory status occasionally when safe
            if operation in ['channel_start', 'batch_complete']:
                self.log_memory_status(operation)
            return True
    
    def safe_batch_size(self, desired_size: int, base_memory_per_item: float = 50.0) -> int:
        """
        Calculate safe batch size based on available memory
        
        Args:
            desired_size: Desired batch size
            base_memory_per_item: Estimated memory per item in MB
        
        Returns:
            Safe batch size
        """
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024**2)
        
        # Reserve 20% of available memory as buffer
        usable_memory_mb = available_mb * 0.8
        
        safe_size = max(1, int(usable_memory_mb / base_memory_per_item))
        recommended_size = min(desired_size, safe_size)
        
        if recommended_size < desired_size:
            logger.warning(f"⚠️  Reducing batch size from {desired_size} to {recommended_size} due to memory constraints")
        
        return recommended_size