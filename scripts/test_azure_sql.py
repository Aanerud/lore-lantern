#!/usr/bin/env python3
"""
Test script for Azure SQL Database connection.

Reads credentials from .env file:
- AZURE_SQL_SERVER
- AZURE_SQL_DATABASE
- AZURE_SQL_USERNAME
- AZURE_SQL_PASSWORD

Run with: python scripts/test_azure_sql.py
"""

import sys
import os
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from src.services.database import DatabaseService
from src.models import ParentAccount, ChildProfile, LearningProgress
from uuid import uuid4
from datetime import datetime


async def test_azure_sql():
    """Test Azure SQL connection and basic operations."""
    print("=" * 60)
    print("Azure SQL Database Test")
    print("=" * 60)

    # Load configuration from .env
    server = os.getenv("AZURE_SQL_SERVER")
    database = os.getenv("AZURE_SQL_DATABASE")
    username = os.getenv("AZURE_SQL_USERNAME")
    password = os.getenv("AZURE_SQL_PASSWORD")

    # Validate required environment variables
    missing = []
    if not server:
        missing.append("AZURE_SQL_SERVER")
    if not database:
        missing.append("AZURE_SQL_DATABASE")
    if not username:
        missing.append("AZURE_SQL_USERNAME")
    if not password:
        missing.append("AZURE_SQL_PASSWORD")

    if missing:
        print(f"\n❌ Missing required environment variables: {', '.join(missing)}")
        print("   Please set them in .env file")
        return False

    print(f"\n1. Connecting to {server}...")
    db = DatabaseService(
        server=server,
        database=database,
        username=username,
        password=password
    )

    try:
        db.initialize()
        print("   ✅ Connection successful!")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

    # Test household operations
    print("\n2. Testing Household (ParentAccount) operations...")
    test_household_id = str(uuid4())
    household = ParentAccount(
        parent_id=test_household_id,
        display_name="Test Family",
        language="en"
    )

    try:
        saved_household = await db.save_household(household)
        print(f"   ✅ Created household: {saved_household.parent_id}")

        retrieved = await db.get_household(test_household_id)
        if retrieved and retrieved.display_name == "Test Family":
            print(f"   ✅ Retrieved household: {retrieved.display_name}")
        else:
            print("   ❌ Failed to retrieve household")
            return False
    except Exception as e:
        print(f"   ❌ Household operations failed: {e}")
        return False

    # Test child operations
    print("\n3. Testing Child operations...")
    test_child_id = str(uuid4())
    child = ChildProfile(
        child_id=test_child_id,
        parent_id=test_household_id,
        name="Test Child",
        birth_year=2018
    )

    try:
        saved_child = await db.save_child_profile(child)
        print(f"   ✅ Created child: {saved_child.name}")

        retrieved = await db.get_child_profile(test_child_id)
        if retrieved and retrieved.name == "Test Child":
            print(f"   ✅ Retrieved child: {retrieved.name}")
            print(f"   ✅ Computed age: {datetime.now().year - retrieved.birth_year} years")
        else:
            print("   ❌ Failed to retrieve child")
            return False

        # Test get children by household
        children = await db.get_children_by_household(test_household_id)
        if len(children) == 1:
            print(f"   ✅ Get children by household: {len(children)} child(ren)")
        else:
            print(f"   ⚠️  Expected 1 child, got {len(children)}")
    except Exception as e:
        print(f"   ❌ Child operations failed: {e}")
        return False

    # Test learning progress
    print("\n4. Testing Learning Progress operations...")
    try:
        progress = await db.get_or_create_learning_progress(test_child_id)
        print(f"   ✅ Created learning progress for child")
        print(f"   ✅ Initial reading level: {progress.reading_level}")

        # Update learning progress
        progress.reading_level = 3
        # vocabulary_bank expects Dict[str, VocabularyEntry] but for simplicity,
        # we'll just update reading_level in this test
        await db.save_learning_progress(progress)

        retrieved_progress = await db.get_learning_progress(test_child_id)
        if retrieved_progress.reading_level == 3:
            print(f"   ✅ Updated reading level: {retrieved_progress.reading_level}")
        else:
            print("   ❌ Failed to update learning progress")
    except Exception as e:
        print(f"   ❌ Learning progress operations failed: {e}")
        return False

    # Cleanup
    print("\n5. Cleaning up test data...")
    try:
        # Delete in correct order (child's learning_progress, children, then household)
        # Note: CASCADE should handle this, but let's be explicit

        # The cascade should clean up everything when we delete the household
        # But we need to delete the learning_progress first (no cascade from children)
        import pyodbc
        conn = pyodbc.connect(db.connection_string)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM learning_progress WHERE child_id = ?", (test_child_id,))
        cursor.execute("DELETE FROM children WHERE id = ?", (test_child_id,))
        cursor.execute("DELETE FROM households WHERE id = ?", (test_household_id,))
        conn.commit()
        conn.close()

        print("   ✅ Test data cleaned up")
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")

    print("\n" + "=" * 60)
    print("All tests passed! Azure SQL is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_azure_sql())
    sys.exit(0 if success else 1)
