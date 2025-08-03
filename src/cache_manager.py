"""
Cache Manager - Intelligent caching system for frames, models, and results
"""

import os
import json
import pickle
import hashlib
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import logging

@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    data_type: str
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    priority: int = 1  # Higher = more important
    ttl: Optional[float] = None  # Time to live in seconds

class CacheManager:
    """
    Intelligent caching system with LRU eviction, priority-based retention,
    and automatic memory management for optimal eye-tracking performance.
    """
    
    def __init__(self, 
                 cache_dir: str = "./cache",
                 max_memory_mb: int = 2048,
                 max_disk_gb: float = 10.0,
                 enable_compression: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = int(max_disk_gb * 1024 * 1024 * 1024)
        self.enable_compression = enable_compression
        
        # Memory cache
        self.memory_cache: Dict[str, Any] = {}
        self.memory_metadata: Dict[str, CacheEntry] = {}
        self.memory_usage = 0
        
        # Disk cache metadata
        self.disk_metadata: Dict[str, CacheEntry] = {}
        self.disk_usage = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "evictions": 0,
            "writes": 0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing disk cache metadata
        self._load_disk_metadata()
        
        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        self.logger.info(f"CacheManager initialized: {cache_dir}, {max_memory_mb}MB memory, {max_disk_gb}GB disk")
    
    def _generate_key(self, data: Any, prefix: str = "") -> str:
        """Generate cache key from data"""
        if isinstance(data, (str, int, float)):
            key_data = str(data)
        elif isinstance(data, (list, tuple)):
            key_data = str(hash(tuple(data)))
        elif isinstance(data, dict):
            key_data = str(hash(tuple(sorted(data.items()))))
        elif isinstance(data, np.ndarray):
            key_data = hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, torch.Tensor):
            key_data = hashlib.md5(data.cpu().numpy().tobytes()).hexdigest()
        elif hasattr(data, '__dict__'):
            key_data = str(hash(tuple(sorted(data.__dict__.items()))))
        else:
            key_data = str(hash(str(data)))
        
        full_key = f"{prefix}_{key_data}" if prefix else key_data
        return hashlib.md5(full_key.encode()).hexdigest()[:16]
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data"""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, Image.Image):
            return data.width * data.height * len(data.getbands()) * 4  # Assume 4 bytes per pixel
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        else:
            try:
                return len(pickle.dumps(data))
            except:
                return 1024  # Default estimate
    
    def _load_disk_metadata(self):
        """Load disk cache metadata"""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                for key, entry_dict in metadata_dict.items():
                    self.disk_metadata[key] = CacheEntry(**entry_dict)
                
                # Calculate disk usage
                self.disk_usage = sum(entry.size_bytes for entry in self.disk_metadata.values())
                
                self.logger.info(f"Loaded {len(self.disk_metadata)} disk cache entries ({self.disk_usage / 1024 / 1024:.1f} MB)")
            
            except Exception as e:
                self.logger.warning(f"Failed to load disk metadata: {e}")
    
    def _save_disk_metadata(self):
        """Save disk cache metadata"""
        metadata_file = self.cache_dir / "metadata.json"
        try:
            metadata_dict = {key: asdict(entry) for key, entry in self.disk_metadata.items()}
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save disk metadata: {e}")
    
    def _evict_memory_lru(self, required_space: int):
        """Evict least recently used items from memory cache"""
        if not self.memory_metadata:
            return
        
        # Sort by last accessed time (LRU)
        sorted_entries = sorted(
            self.memory_metadata.items(),
            key=lambda x: (x[1].priority, x[1].last_accessed)  # Priority first, then LRU
        )
        
        freed_space = 0
        for key, entry in sorted_entries:
            if freed_space >= required_space:
                break
            
            # Remove from memory
            if key in self.memory_cache:
                del self.memory_cache[key]
                freed_space += entry.size_bytes
                self.memory_usage -= entry.size_bytes
                del self.memory_metadata[key]
                self.stats["evictions"] += 1
        
        self.logger.debug(f"Evicted {freed_space / 1024 / 1024:.1f} MB from memory cache")
    
    def _evict_disk_lru(self, required_space: int):
        """Evict least recently used items from disk cache"""
        if not self.disk_metadata:
            return
        
        # Sort by last accessed time (LRU) and priority
        sorted_entries = sorted(
            self.disk_metadata.items(),
            key=lambda x: (x[1].priority, x[1].last_accessed)
        )
        
        freed_space = 0
        for key, entry in sorted_entries:
            if freed_space >= required_space:
                break
            
            # Remove file
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    freed_space += entry.size_bytes
                    self.disk_usage -= entry.size_bytes
                    del self.disk_metadata[key]
                    self.stats["evictions"] += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete cache file {cache_file}: {e}")
        
        self._save_disk_metadata()
        self.logger.debug(f"Evicted {freed_space / 1024 / 1024:.1f} MB from disk cache")
    
    def put(self, key: str, data: Any, priority: int = 1, ttl: Optional[float] = None,
            force_disk: bool = False) -> bool:
        """Store data in cache"""
        with self.lock:
            size_bytes = self._estimate_size(data)
            current_time = time.time()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data_type=type(data).__name__,
                size_bytes=size_bytes,
                created_at=current_time,
                last_accessed=current_time,
                access_count=0,
                priority=priority,
                ttl=ttl
            )
            
            # Decide storage location
            use_disk = force_disk or size_bytes > self.max_memory_bytes // 4  # Large items go to disk
            
            if not use_disk and size_bytes <= self.max_memory_bytes:
                # Try memory cache first
                required_space = size_bytes
                if self.memory_usage + required_space > self.max_memory_bytes:
                    self._evict_memory_lru(required_space)
                
                if self.memory_usage + required_space <= self.max_memory_bytes:
                    self.memory_cache[key] = data
                    self.memory_metadata[key] = entry
                    self.memory_usage += size_bytes
                    self.stats["writes"] += 1
                    return True
                else:
                    use_disk = True  # Fallback to disk
            
            if use_disk:
                # Use disk cache
                required_space = size_bytes
                if self.disk_usage + required_space > self.max_disk_bytes:
                    self._evict_disk_lru(required_space)
                
                if self.disk_usage + required_space <= self.max_disk_bytes:
                    cache_file = self.cache_dir / f"{key}.cache"
                    try:
                        self._save_to_disk(data, cache_file)
                        self.disk_metadata[key] = entry
                        self.disk_usage += size_bytes
                        self._save_disk_metadata()
                        self.stats["writes"] += 1
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to save to disk cache: {e}")
            
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache"""
        with self.lock:
            current_time = time.time()
            
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_metadata[key]
                
                # Check TTL
                if entry.ttl and current_time - entry.created_at > entry.ttl:
                    self._remove_from_memory(key)
                    self.stats["memory_misses"] += 1
                    return None
                
                # Update access info
                entry.last_accessed = current_time
                entry.access_count += 1
                
                self.stats["memory_hits"] += 1
                return self.memory_cache[key]
            
            # Check disk cache
            if key in self.disk_metadata:
                entry = self.disk_metadata[key]
                
                # Check TTL
                if entry.ttl and current_time - entry.created_at > entry.ttl:
                    self._remove_from_disk(key)
                    self.stats["disk_misses"] += 1
                    return None
                
                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    try:
                        data = self._load_from_disk(cache_file)
                        
                        # Update access info
                        entry.last_accessed = current_time
                        entry.access_count += 1
                        self._save_disk_metadata()
                        
                        # Maybe promote to memory cache
                        if entry.access_count > 2 and entry.size_bytes < self.max_memory_bytes // 8:
                            self.put(key, data, entry.priority, entry.ttl)
                        
                        self.stats["disk_hits"] += 1
                        return data
                    
                    except Exception as e:
                        self.logger.error(f"Failed to load from disk cache: {e}")
                        self._remove_from_disk(key)
            
            # Cache miss
            self.stats["memory_misses" if key not in self.disk_metadata else "disk_misses"] += 1
            return None
    
    def _save_to_disk(self, data: Any, cache_file: Path):
        """Save data to disk with appropriate format"""
        if isinstance(data, np.ndarray):
            np.save(str(cache_file).replace('.cache', '.npy'), data)
            cache_file.write_text('.npy')  # Store format info
        elif isinstance(data, torch.Tensor):
            torch.save(data, str(cache_file).replace('.cache', '.pt'))
            cache_file.write_text('.pt')
        elif isinstance(data, Image.Image):
            data.save(str(cache_file).replace('.cache', '.png'))
            cache_file.write_text('.png')
        else:
            # Use pickle for general objects
            with open(cache_file, 'wb') as f:
                if self.enable_compression:
                    import gzip
                    with gzip.open(f, 'wb') as gz:
                        pickle.dump(data, gz)
                else:
                    pickle.dump(data, f)
    
    def _load_from_disk(self, cache_file: Path) -> Any:
        """Load data from disk with appropriate format"""
        if cache_file.exists() and cache_file.stat().st_size < 10:  # Format indicator file
            format_ext = cache_file.read_text().strip()
            actual_file = str(cache_file).replace('.cache', format_ext)
            
            if format_ext == '.npy':
                return np.load(actual_file)
            elif format_ext == '.pt':
                return torch.load(actual_file, map_location='cpu')
            elif format_ext == '.png':
                return Image.open(actual_file)
        
        # Default pickle loading
        with open(cache_file, 'rb') as f:
            if self.enable_compression:
                import gzip
                try:
                    with gzip.open(f, 'rb') as gz:
                        return pickle.load(gz)
                except:
                    f.seek(0)
                    return pickle.load(f)
            else:
                return pickle.load(f)
    
    def _remove_from_memory(self, key: str):
        """Remove item from memory cache"""
        if key in self.memory_cache:
            entry = self.memory_metadata[key]
            self.memory_usage -= entry.size_bytes
            del self.memory_cache[key]
            del self.memory_metadata[key]
    
    def _remove_from_disk(self, key: str):
        """Remove item from disk cache"""
        if key in self.disk_metadata:
            entry = self.disk_metadata[key]
            self.disk_usage -= entry.size_bytes
            
            # Remove files
            cache_file = self.cache_dir / f"{key}.cache"
            for ext in ['.cache', '.npy', '.pt', '.png']:
                file_path = self.cache_dir / f"{key}{ext}"
                if file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        self.logger.error(f"Failed to delete {file_path}: {e}")
            
            del self.disk_metadata[key]
    
    def remove(self, key: str) -> bool:
        """Remove item from cache"""
        with self.lock:
            removed = False
            
            if key in self.memory_cache:
                self._remove_from_memory(key)
                removed = True
            
            if key in self.disk_metadata:
                self._remove_from_disk(key)
                self._save_disk_metadata()
                removed = True
            
            return removed
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            # Clear memory
            self.memory_cache.clear()
            self.memory_metadata.clear()
            self.memory_usage = 0
            
            # Clear disk
            for key in list(self.disk_metadata.keys()):
                self._remove_from_disk(key)
            
            self.disk_metadata.clear()
            self.disk_usage = 0
            self._save_disk_metadata()
            
            self.logger.info("Cache cleared")
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while True:
            try:
                time.sleep(60)  # Run every minute
                
                with self.lock:
                    current_time = time.time()
                    
                    # Clean expired memory entries
                    expired_keys = []
                    for key, entry in self.memory_metadata.items():
                        if entry.ttl and current_time - entry.created_at > entry.ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_from_memory(key)
                    
                    # Clean expired disk entries
                    expired_keys = []
                    for key, entry in self.disk_metadata.items():
                        if entry.ttl and current_time - entry.created_at > entry.ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_from_disk(key)
                    
                    if expired_keys:
                        self._save_disk_metadata()
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                        
            except Exception as e:
                self.logger.error(f"Cleanup worker error: {e}")
    
    # Convenience methods for specific data types
    
    def cache_frame(self, frame_id: str, frame: np.ndarray, priority: int = 1) -> bool:
        """Cache a video frame"""
        return self.put(f"frame_{frame_id}", frame, priority=priority)
    
    def get_frame(self, frame_id: str) -> Optional[np.ndarray]:
        """Get cached video frame"""
        return self.get(f"frame_{frame_id}")
    
    def cache_landmarks(self, frame_id: str, landmarks: np.ndarray, priority: int = 2) -> bool:
        """Cache facial landmarks"""
        return self.put(f"landmarks_{frame_id}", landmarks, priority=priority)
    
    def get_landmarks(self, frame_id: str) -> Optional[np.ndarray]:
        """Get cached facial landmarks"""
        return self.get(f"landmarks_{frame_id}")
    
    def cache_depth_map(self, frame_id: str, depth: np.ndarray, priority: int = 2) -> bool:
        """Cache depth map"""
        return self.put(f"depth_{frame_id}", depth, priority=priority)
    
    def get_depth_map(self, frame_id: str) -> Optional[np.ndarray]:
        """Get cached depth map"""
        return self.get(f"depth_{frame_id}")
    
    def cache_model_output(self, model_name: str, input_hash: str, output: Any, 
                          priority: int = 3, ttl: float = 300) -> bool:
        """Cache model inference output"""
        key = f"model_{model_name}_{input_hash}"
        return self.put(key, output, priority=priority, ttl=ttl)
    
    def get_model_output(self, model_name: str, input_hash: str) -> Optional[Any]:
        """Get cached model output"""
        key = f"model_{model_name}_{input_hash}"
        return self.get(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_hits = self.stats["memory_hits"] + self.stats["disk_hits"]
            total_misses = self.stats["memory_misses"] + self.stats["disk_misses"]
            total_requests = total_hits + total_misses
            
            hit_rate = total_hits / max(total_requests, 1) * 100
            memory_hit_rate = self.stats["memory_hits"] / max(total_requests, 1) * 100
            
            return {
                **self.stats,
                "total_requests": total_requests,
                "hit_rate_percent": hit_rate,
                "memory_hit_rate_percent": memory_hit_rate,
                "memory_usage_mb": self.memory_usage / 1024 / 1024,
                "memory_usage_percent": self.memory_usage / self.max_memory_bytes * 100,
                "disk_usage_mb": self.disk_usage / 1024 / 1024,
                "disk_usage_percent": self.disk_usage / self.max_disk_bytes * 100,
                "memory_entries": len(self.memory_metadata),
                "disk_entries": len(self.disk_metadata)
            }
    
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print("📊 Cache Statistics:")
        print(f"   Hit Rate: {stats['hit_rate_percent']:.1f}% ({stats['memory_hit_rate_percent']:.1f}% memory)")
        print(f"   Requests: {stats['total_requests']} (Hits: {stats['memory_hits'] + stats['disk_hits']}, Misses: {stats['memory_misses'] + stats['disk_misses']})")
        print(f"   Memory: {stats['memory_entries']} entries, {stats['memory_usage_mb']:.1f} MB ({stats['memory_usage_percent']:.1f}%)")
        print(f"   Disk: {stats['disk_entries']} entries, {stats['disk_usage_mb']:.1f} MB ({stats['disk_usage_percent']:.1f}%)")
        print(f"   Operations: {stats['writes']} writes, {stats['evictions']} evictions")