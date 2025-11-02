"""
Test Supabase connection and setup
Run this after creating the tables in Supabase
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from exporters.export_supabase import SupabaseExporter
from logs import logger


def test_supabase_connection():
    """Test Supabase connection and table setup"""
    try:
        logger.info("Testing Supabase connection...")
        exporter = SupabaseExporter()

        # Test connection
        if exporter.test_connection():
            logger.info("✅ Supabase connection successful!")
            logger.info("✅ Tables are set up correctly")
            logger.info("\nYou can now run your strategy and data will be exported to Supabase")
            return True
        else:
            logger.error("❌ Connection test failed")
            logger.info("\n📋 Setup Instructions:")
            logger.info("1. Go to your Supabase dashboard SQL Editor")
            logger.info("2. Run the SQL in 'supabase_schema.sql' to create tables")
            logger.info("3. Run this test again")
            return False

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.info("\n📋 Troubleshooting:")
        logger.info("1. Check that .env file has sb_db and sb_api variables")
        logger.info("2. Verify your Supabase credentials are correct")
        logger.info("3. Ensure you've created the tables using supabase_schema.sql")
        return False


if __name__ == "__main__":
    test_supabase_connection()
