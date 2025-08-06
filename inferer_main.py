import logging
from sliding_window_inferer import SlidingWindowInferer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class InfererManager:
    def __init__(self):
        self.inferer = self._setup_inferer()

    def _setup_inferer(self):
        """Initialize the SlidingWindowInferer with default configuration."""
        try:
            logger.info("Setting up SlidingWindowInferer")
            inferer = SlidingWindowInferer(config={
                "roi_size": (96, 96, 96),
                "sw_batch_size": 2,
                "overlap": 0.7
            })
            logger.info("SlidingWindowInferer setup completed")
            return inferer
        except Exception as e:
            logger.error(f"Failed to setup SlidingWindowInferer: {e}")
            raise

    def get_inferer(self):
        """Return the configured SlidingWindowInferer instance."""
        return self.inferer