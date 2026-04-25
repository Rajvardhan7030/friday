from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
import logging

logger = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self, db_url: str = 'sqlite:///jobs.sqlite'):
        jobstores = {
            'default': SQLAlchemyJobStore(url=db_url)
        }
        self.scheduler = AsyncIOScheduler(jobstores=jobstores)

    def start(self):
        self.scheduler.start()
        logger.info("Scheduler started.")

    def add_job(self, name: str, func, *args, **kwargs):
        job = self.scheduler.add_job(
            func, 'cron', id=name, replace_existing=True,
            args=args, kwargs=kwargs, **kwargs.get('cron_kwargs', {})
        )
        logger.info(f"Added scheduled job: {name}")
        return job

    def remove_job(self, name: str):
        self.scheduler.remove_job(name)
        logger.info(f"Removed scheduled job: {name}")

    def list_jobs(self):
        return self.scheduler.get_jobs()

    def pause_job(self, name: str):
        self.scheduler.pause_job(name)

    def resume_job(self, name: str):
        self.scheduler.resume_job(name)
