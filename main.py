import os
from demo_interface import TrojanDemoInterface
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Trojan Attack demonstration."""
    # Load environment variables
    load_dotenv()
    
    logger.info("Starting Trojan Attack Demonstration...")
    
    try:
        # Initialize and launch the demo interface
        demo = TrojanDemoInterface()
        demo.launch(share=False)  # Set share=True to create a public URL
        
    except Exception as e:
        logger.error(f"Error running demonstration: {str(e)}")
        raise

if __name__ == "__main__":
    main()
