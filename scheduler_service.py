# """
# Scheduler Service for MCP Tools
# Handles delayed and scheduled task execution
# """

# import asyncio
# import logging
# import json
# import re
# from datetime import datetime, timedelta
# from typing import Dict, List, Optional, Callable
# from dataclasses import dataclass, asdict
# import threading
# import time
# import os

# @dataclass
# class ScheduledTask:
#     """Represents a scheduled task"""
#     id: str
#     user_id: str
#     original_query: str
#     action: str
#     execute_at: datetime
#     is_recurring: bool = False
#     recurring_pattern: Optional[str] = None  # "daily", "weekly", etc.
#     status: str = "pending"  # pending, executing, completed, failed
#     created_at: datetime = None
    
#     def __post_init__(self):
#         if self.created_at is None:
#             self.created_at = datetime.now()

# class TimeParser:
#     """Parses natural language time expressions"""
    
#     @staticmethod
#     def parse_time_expression(text: str) -> Optional[datetime]:
#         """Parse time expressions like 'in 5 minutes', 'at 6am', 'daily at 6am'"""
#         text = text.lower().strip()
#         now = datetime.now()
        
#         # Relative time patterns (with common abbreviations)
#         relative_patterns = [
#             (r'in (\d+)\s*(minutes?|mins?|m)\b', lambda m: now + timedelta(minutes=int(m.group(1)))),
#             (r'in (\d+)\s*(hours?|hrs?|h)\b', lambda m: now + timedelta(hours=int(m.group(1)))),
#             (r'in (\d+)\s*(days?|d)\b', lambda m: now + timedelta(days=int(m.group(1)))),
#             (r'in (\d+)\s*(seconds?|secs?|s)\b', lambda m: now + timedelta(seconds=int(m.group(1)))),
#             # also support "after X ..."
#             (r'after (\d+)\s*(minutes?|mins?|m)\b', lambda m: now + timedelta(minutes=int(m.group(1)))),
#             (r'after (\d+)\s*(hours?|hrs?|h)\b', lambda m: now + timedelta(hours=int(m.group(1)))),
#             (r'after (\d+)\s*(days?|d)\b', lambda m: now + timedelta(days=int(m.group(1)))),
#             (r'after (\d+)\s*(seconds?|secs?|s)\b', lambda m: now + timedelta(seconds=int(m.group(1)))),
#         ]
        
#         for pattern, func in relative_patterns:
#             match = re.search(pattern, text)
#             if match:
#                 return func(match)
        
#         # Absolute time patterns
#         absolute_patterns = [
#             (r'at (\d{1,2}):(\d{2})\s*(am|pm)?', TimeParser._parse_absolute_time),
#             (r'at (\d{1,2})\s*(am|pm)', TimeParser._parse_absolute_time_simple),
#         ]
        
#         for pattern, func in absolute_patterns:
#             match = re.search(pattern, text)
#             if match:
#                 return func(match, now)
        
#         return None
    
#     @staticmethod
#     def _parse_absolute_time(match, now: datetime) -> datetime:
#         """Parse absolute time like 'at 6:30am' or 'at 14:30'"""
#         hour = int(match.group(1))
#         minute = int(match.group(2))
#         ampm = match.group(3) if len(match.groups()) > 2 else None
        
#         if ampm:
#             if ampm == 'pm' and hour != 12:
#                 hour += 12
#             elif ampm == 'am' and hour == 12:
#                 hour = 0
        
#         # If time has passed today, schedule for tomorrow
#         target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
#         if target_time <= now:
#             target_time += timedelta(days=1)
        
#         return target_time
    
#     @staticmethod
#     def _parse_absolute_time_simple(match, now: datetime) -> datetime:
#         """Parse simple absolute time like 'at 6am'"""
#         hour = int(match.group(1))
#         ampm = match.group(2)
        
#         if ampm == 'pm' and hour != 12:
#             hour += 12
#         elif ampm == 'am' and hour == 12:
#             hour = 0
        
#         # If time has passed today, schedule for tomorrow
#         target_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
#         if target_time <= now:
#             target_time += timedelta(days=1)
        
#         return target_time
    
#     @staticmethod
#     def is_recurring(text: str) -> bool:
#         """Check if the text indicates a recurring task"""
#         recurring_keywords = ['daily', 'every day', 'weekly', 'every week', 'monthly', 'every month']
#         return any(keyword in text.lower() for keyword in recurring_keywords)
    
#     @staticmethod
#     def get_recurring_pattern(text: str) -> Optional[str]:
#         """Extract recurring pattern from text"""
#         text = text.lower()
#         if 'daily' in text or 'every day' in text:
#             return 'daily'
#         elif 'weekly' in text or 'every week' in text:
#             return 'weekly'
#         elif 'monthly' in text or 'every month' in text:
#             return 'monthly'
#         return None

# class SchedulerService:
#     """Manages scheduled tasks for MCP tools"""
    
#     def __init__(self, mcp_executor: Callable, storage_path: str = "scheduled_tasks.json"):
#         self.tasks: Dict[str, ScheduledTask] = {}
#         self.mcp_executor = mcp_executor
#         self.running = False
#         self.scheduler_thread = None
#         self.task_counter = 0
#         self.storage_path = storage_path
        
#     def start(self):
#         """Start the scheduler service"""
#         if not self.running:
#             self.running = True
#             self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
#             self.scheduler_thread.start()
#             logging.info("🕐 Scheduler service started")
#             # Load any persisted tasks from disk on start
#             try:
#                 self._load_from_disk()
#             except Exception as e:
#                 logging.warning(f"Could not load scheduled tasks from disk: {e}")
    
#     def stop(self):
#         """Stop the scheduler service"""
#         self.running = False
#         if self.scheduler_thread:
#             self.scheduler_thread.join(timeout=5)
#         logging.info("🕐 Scheduler service stopped")
    
#     def schedule_task(self, user_id: str, query: str, action: str) -> Optional[str]:
#         """Schedule a new task"""
#         # Parse time from the query
#         execute_at = TimeParser.parse_time_expression(query)
#         if not execute_at:
#             return None
        
#         # Check if it's recurring
#         is_recurring = TimeParser.is_recurring(query)
#         recurring_pattern = TimeParser.get_recurring_pattern(query) if is_recurring else None
        
#         # Create task
#         task_id = f"task_{self.task_counter}_{int(time.time())}"
#         self.task_counter += 1
        
#         task = ScheduledTask(
#             id=task_id,
#             user_id=user_id,
#             original_query=query,
#             action=action,
#             execute_at=execute_at,
#             is_recurring=is_recurring,
#             recurring_pattern=recurring_pattern
#         )
        
#         self.tasks[task_id] = task
#         logging.info(f"📅 Scheduled task {task_id}: '{action}' at {execute_at}")
#         # Persist tasks
#         try:
#             self._save_to_disk()
#         except Exception as e:
#             logging.warning(f"Could not persist scheduled tasks: {e}")
        
#         return task_id
    
#     def get_scheduled_tasks(self, user_id: str = None) -> List[Dict]:
#         """Get scheduled tasks, optionally filtered by user"""
#         tasks = list(self.tasks.values())
#         if user_id:
#             tasks = [t for t in tasks if t.user_id == user_id]
        
#         return [asdict(task) for task in tasks]
    
#     def cancel_task(self, task_id: str) -> bool:
#         """Cancel a scheduled task"""
#         if task_id in self.tasks:
#             del self.tasks[task_id]
#             logging.info(f"❌ Cancelled task {task_id}")
#             try:
#                 self._save_to_disk()
#             except Exception as e:
#                 logging.warning(f"Could not persist scheduled tasks after cancel: {e}")
#             return True
#         return False
    
#     def _scheduler_loop(self):
#         """Main scheduler loop - runs in background thread"""
#         while self.running:
#             try:
#                 now = datetime.now()
#                 tasks_to_execute = []
                
#                 # Find tasks ready for execution
#                 for task_id, task in self.tasks.items():
#                     if task.status == "pending" and task.execute_at <= now:
#                         tasks_to_execute.append(task)
                
#                 # Execute ready tasks
#                 for task in tasks_to_execute:
#                     self._execute_task(task)
                
#                 # Sleep for a short interval
#                 time.sleep(1)
                
#             except Exception as e:
#                 logging.error(f"Error in scheduler loop: {e}")
#                 time.sleep(5)
    
#     def _execute_task(self, task: ScheduledTask):
#         """Execute a scheduled task"""
#         try:
#             logging.info(f"🚀 Executing scheduled task {task.id}: {task.action}")
#             task.status = "executing"
            
#             # Execute the action using MCP
#             result = asyncio.run(self.mcp_executor(task.action, task.user_id))
            
#             if result:
#                 task.status = "completed"
#                 logging.info(f"✅ Task {task.id} completed successfully")
                
#                 # Handle recurring tasks
#                 if task.is_recurring and task.recurring_pattern:
#                     self._reschedule_recurring_task(task)
#             else:
#                 task.status = "failed"
#                 logging.error(f"❌ Task {task.id} failed")
                
#         except Exception as e:
#             task.status = "failed"
#             logging.error(f"❌ Error executing task {task.id}: {e}")
    
#     def _reschedule_recurring_task(self, task: ScheduledTask):
#         """Reschedule a recurring task"""
#         if task.recurring_pattern == "daily":
#             task.execute_at += timedelta(days=1)
#         elif task.recurring_pattern == "weekly":
#             task.execute_at += timedelta(weeks=1)
#         elif task.recurring_pattern == "monthly":
#             # Simple monthly increment
#             if task.execute_at.month == 12:
#                 task.execute_at = task.execute_at.replace(year=task.execute_at.year + 1, month=1)
#             else:
#                 task.execute_at = task.execute_at.replace(month=task.execute_at.month + 1)
        
#         task.status = "pending"
#         logging.info(f"🔄 Rescheduled recurring task {task.id} for {task.execute_at}")
#         try:
#             self._save_to_disk()
#         except Exception as e:
#             logging.warning(f"Could not persist scheduled tasks after reschedule: {e}")

#     def _save_to_disk(self):
#         """Persist current tasks to disk as JSON (ISO datetimes)."""
#         data = []
#         for t in self.tasks.values():
#             item = asdict(t)
#             item["execute_at"] = t.execute_at.isoformat()
#             item["created_at"] = t.created_at.isoformat() if t.created_at else None
#             data.append(item)
#         with open(self.storage_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2)

#     def _load_from_disk(self):
#         """Load tasks from disk if file exists; only restore future tasks."""
#         if not os.path.exists(self.storage_path):
#             return
#         with open(self.storage_path, "r", encoding="utf-8") as f:
#             raw = json.load(f)
#         self.tasks = {}
#         now = datetime.now()
#         for item in raw:
#             try:
#                 execute_at = datetime.fromisoformat(item.get("execute_at"))
#                 created_at_str = item.get("created_at")
#                 created_at = datetime.fromisoformat(created_at_str) if created_at_str else None
#                 # Only restore tasks that are still in the future or recurring
#                 if execute_at >= now or item.get("is_recurring"):
#                     t = ScheduledTask(
#                         id=item["id"],
#                         user_id=item.get("user_id", "system"),
#                         original_query=item.get("original_query", ""),
#                         action=item.get("action", ""),
#                         execute_at=execute_at,
#                         is_recurring=item.get("is_recurring", False),
#                         recurring_pattern=item.get("recurring_pattern"),
#                         status=item.get("status", "pending"),
#                         created_at=created_at
#                     )
#                     # Reset status to pending for restored tasks
#                     t.status = "pending"
#                     self.tasks[t.id] = t
#             except Exception as e:
#                 logging.warning(f"Skipping invalid scheduled task from disk: {e}")

# # Global scheduler instance
# _scheduler_instance: Optional[SchedulerService] = None

# def get_scheduler() -> Optional[SchedulerService]:
#     """Get the global scheduler instance"""
#     return _scheduler_instance

# def initialize_scheduler(mcp_executor: Callable):
#     """Initialize the global scheduler"""
#     global _scheduler_instance
#     _scheduler_instance = SchedulerService(mcp_executor)
#     _scheduler_instance.start()
#     return _scheduler_instance

# def shutdown_scheduler():
#     """Shutdown the global scheduler"""
#     global _scheduler_instance
#     if _scheduler_instance:
#         _scheduler_instance.stop()
#         _scheduler_instance = None


# scheduler_service.py
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.cron import CronTrigger

@dataclass
class _Task:
    id: str
    user_id: str
    description: str
    action: str
    next_run_time: Optional[datetime]

_scheduler: Optional[AsyncIOScheduler] = None
_executor: Optional[Callable[[str, str], asyncio.Future]] = None  # async (action, user_id) -> bool
_tasks: Dict[str, _Task] = {}

def initialize_scheduler(executor: Callable[[str, str], asyncio.Future]) -> None:
    """Start a single AsyncIOScheduler instance bound to the current event loop."""
    global _scheduler, _executor
    if _scheduler is not None:
        return
    _executor = executor
    _scheduler = AsyncIOScheduler(timezone="Asia/Kolkata", job_defaults={
        "misfire_grace_time": 60,   # run if we miss by up to 60s
        "coalesce": True
    })
    _scheduler.start()
    logging.info("✅ AsyncIOScheduler started")

def get_scheduler():
    return _scheduler

def shutdown_scheduler():
    global _scheduler
    if _scheduler:
        try:
            _scheduler.shutdown(wait=False)
            logging.info("Scheduler has been shut down")
        except Exception as e:
            logging.warning(f"Scheduler shutdown warning: {e}")
        finally:
            _scheduler = None

async def _execute_job(user_id: str, action: str, job_id: str):
    """Actual async job that calls your MCP executor."""
    try:
        if _executor is None:
            logging.error("Executor not set")
            return
        logging.info(f"▶️ Running scheduled job {job_id}: {action}")
        await _executor(action, user_id)   # awaits ChatSession.chat(...)
    except Exception as e:
        logging.exception(f"Scheduled job {job_id} failed: {e}")

def _store_task(job_id: str, user_id: str, description: str, action: str, next_time: Optional[datetime]):
    _tasks[job_id] = _Task(id=job_id, user_id=user_id, description=description, action=action, next_run_time=next_time)

# ---------- Public helpers your API can call ----------

def schedule_in_seconds(user_id: str, action: str, seconds: int, description: str) -> Optional[str]:
    if not _scheduler:
        return None
    run_at = datetime.now(_scheduler.timezone) + timedelta(seconds=seconds)
    job = _scheduler.add_job(_execute_job, DateTrigger(run_date=run_at),
                             args=[user_id, action, None], replace_existing=False)
    # store job_id in job args for logging
    job.modify(args=[user_id, action, job.id])
    _store_task(job.id, user_id, description, action, run_at)
    return job.id

def schedule_at(user_id: str, action: str, run_at: datetime, description: str) -> Optional[str]:
    if not _scheduler:
        return None
    # ensure timezone-aware
    if run_at.tzinfo is None:
        run_at = _scheduler.timezone.localize(run_at)  # type: ignore[attr-defined]
    job = _scheduler.add_job(_execute_job, DateTrigger(run_date=run_at),
                             args=[user_id, action, None], replace_existing=False)
    job.modify(args=[user_id, action, job.id])
    _store_task(job.id, user_id, description, action, run_at)
    return job.id

def schedule_daily_at(user_id: str, action: str, hour: int, minute: int, description: str) -> Optional[str]:
    if not _scheduler:
        return None
    trigger = CronTrigger(hour=hour, minute=minute, timezone=_scheduler.timezone)
    job = _scheduler.add_job(_execute_job, trigger, args=[user_id, action, None], replace_existing=False)
    job.modify(args=[user_id, action, job.id])
    _store_task(job.id, user_id, description, action, job.next_run_time)
    return job.id

def get_scheduled_tasks(user_id: Optional[str] = None) -> List[Dict]:
    out = []
    for t in _tasks.values():
        if user_id and t.user_id != user_id:
            continue
        out.append({
            "task_id": t.id,
            "user_id": t.user_id,
            "action": t.action,
            "description": t.description,
            "next_run_time": t.next_run_time.isoformat() if t.next_run_time else None
        })
    return out

def cancel_task(task_id: str) -> bool:
    if not _scheduler:
        return False
    try:
        _scheduler.remove_job(task_id)
        _tasks.pop(task_id, None)
        return True
    except Exception:
        return False
