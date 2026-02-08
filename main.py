"""Main application entry point."""
import sys

from src.core.config import settings
from src.core.logging_config import get_logger, setup_logging
from src.presentation.gradio_ui import GradioUI


def main() -> None:
    """Main application entry."""
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment
    )
    
    try:
        # Create and launch UI
        ui = GradioUI()
        ui.launch()
        
    except KeyboardInterrupt:
        logger.info("application_interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error("application_error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
