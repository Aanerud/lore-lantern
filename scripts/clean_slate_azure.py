#!/usr/bin/env python3
"""
Clean Slate Script for Lore Lantern - Azure Edition

Deletes ALL data in Azure SQL Database and Azure Blob Storage,
optionally creates fresh sample data.
Use with caution - this is destructive!

Usage:
    python scripts/clean_slate_azure.py              # Interactive mode
    python scripts/clean_slate_azure.py --confirm    # Skip confirmation (dangerous!)
    python scripts/clean_slate_azure.py --dry-run    # Show what would be deleted
    python scripts/clean_slate_azure.py --no-sample  # Don't create sample data after cleanup
    python scripts/clean_slate_azure.py --keep-profiles  # Keep households/children, only delete stories
    python scripts/clean_slate_azure.py --db-only    # Only clean database, skip blob storage
    python scripts/clean_slate_azure.py --blob-only  # Only clean blob storage, skip database
"""

import sys
from pathlib import Path
import argparse
import uuid
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Tables in deletion order (respects foreign key constraints)
# Delete in reverse order of creation
TABLES_DELETE_ORDER = [
    "fact_check_reports",
    "dialogues",
    "reading_states",
    "characters",
    "chapters",
    "stories",
    "learning_progress",
    "children",
    "households",
]

# Tables to keep when using --keep-profiles
PROFILE_TABLES = ["households", "children", "learning_progress"]

# Story-related tables (deleted when --keep-profiles)
STORY_TABLES = [
    "fact_check_reports",
    "dialogues",
    "reading_states",
    "characters",
    "chapters",
    "stories",
]


def get_azure_config():
    """Load Azure configuration from environment."""
    from dotenv import load_dotenv
    import os

    load_dotenv(project_root / ".env")

    config = {
        "sql_server": os.getenv("AZURE_SQL_SERVER"),
        "sql_database": os.getenv("AZURE_SQL_DATABASE"),
        "sql_username": os.getenv("AZURE_SQL_USERNAME"),
        "sql_password": os.getenv("AZURE_SQL_PASSWORD"),
        "blob_connection_string": os.getenv("AZURE_BLOB_CONNECTION_STRING"),
        "blob_container": os.getenv("AZURE_BLOB_CONTAINER", "lorelantern-audio"),
    }

    if not all([config["sql_server"], config["sql_database"],
                config["sql_username"], config["sql_password"]]):
        print("Error: Azure SQL configuration incomplete in .env")
        print("  Required: AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_USERNAME, AZURE_SQL_PASSWORD")
        sys.exit(1)

    return config


def get_sql_connection(config):
    """Create Azure SQL connection."""
    import pyodbc

    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={config['sql_server']};"
        f"DATABASE={config['sql_database']};"
        f"UID={config['sql_username']};"
        f"PWD={config['sql_password']};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )

    try:
        conn = pyodbc.connect(conn_str)
        print(f"Connected to Azure SQL: {config['sql_database']}")
        return conn
    except Exception as e:
        print(f"Error connecting to Azure SQL: {e}")
        sys.exit(1)


def get_blob_client(config):
    """Create Azure Blob Storage client."""
    if not config.get("blob_connection_string"):
        print("  Warning: AZURE_BLOB_CONNECTION_STRING not set, skipping blob storage")
        return None, None

    try:
        from azure.storage.blob import BlobServiceClient

        blob_service = BlobServiceClient.from_connection_string(
            config["blob_connection_string"]
        )
        container_client = blob_service.get_container_client(config["blob_container"])

        # Check if container exists
        if container_client.exists():
            print(f"Connected to Azure Blob Storage: {config['blob_container']}")
            return blob_service, container_client
        else:
            print(f"  Warning: Container '{config['blob_container']}' does not exist")
            return blob_service, None
    except Exception as e:
        print(f"  Warning: Could not connect to Blob Storage: {e}")
        return None, None


def count_table_rows(conn, table_name: str) -> int:
    """Count rows in a SQL table."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    except Exception:
        return 0


def list_sql_data(conn) -> dict:
    """List all tables with row counts."""
    data = {}
    for table in TABLES_DELETE_ORDER:
        count = count_table_rows(conn, table)
        if count > 0:
            data[table] = count
    return data


def list_blob_data(container_client) -> tuple[int, int]:
    """List blobs in container."""
    if not container_client:
        return 0, 0

    try:
        blobs = list(container_client.list_blobs())
        count = len(blobs)
        size = sum(blob.size for blob in blobs)
        return count, size
    except Exception as e:
        print(f"  Error listing blobs: {e}")
        return 0, 0


def delete_table_data(conn, table_name: str, dry_run: bool = False) -> int:
    """Delete all rows from a table."""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        if count > 0 and not dry_run:
            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()

        cursor.close()
        return count
    except Exception as e:
        print(f"  Error deleting from {table_name}: {e}")
        return 0


def delete_blob_data(container_client, dry_run: bool = False) -> tuple[int, int]:
    """Delete all blobs in container."""
    if not container_client:
        return 0, 0

    try:
        blobs = list(container_client.list_blobs())
        count = len(blobs)
        size = sum(blob.size for blob in blobs)

        if count > 0 and not dry_run:
            for blob in blobs:
                container_client.delete_blob(blob.name)

        return count, size
    except Exception as e:
        print(f"  Error deleting blobs: {e}")
        return 0, 0


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def create_sample_data(conn):
    """Create sample household and child for testing."""
    print("\n--- Creating Sample Data ---")

    cursor = conn.cursor()

    # Create sample household (Aanerud Family)
    household_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO households (id, display_name, language, created_at, updated_at)
        VALUES (?, ?, ?, GETUTCDATE(), GETUTCDATE())
    """, (household_id, "Aanerud Family", "no"))
    print(f"  Created household: Aanerud Family ({household_id[:8]}...)")

    # Create sample child (Inger Helene)
    child_id = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO children (id, household_id, name, birth_year, created_at, updated_at)
        VALUES (?, ?, ?, ?, GETUTCDATE(), GETUTCDATE())
    """, (child_id, household_id, "Inger Helene", 2018))
    print(f"  Created child: Inger Helene (born 2018) ({child_id[:8]}...)")

    # Create learning progress
    cursor.execute("""
        INSERT INTO learning_progress (child_id, reading_level, total_stories, total_chapters_read, curiosity_score, updated_at)
        VALUES (?, 1, 0, 0, 0, GETUTCDATE())
    """, (child_id,))
    print(f"  Created learning progress for Inger Helene")

    conn.commit()
    cursor.close()

    print("\n  Sample data created!")
    print(f"  Household ID: {household_id}")
    print(f"  Child ID: {child_id}")

    return household_id, child_id


def main():
    parser = argparse.ArgumentParser(description="Clean slate - delete all Azure data")
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("--no-sample", action="store_true", help="Don't create sample data after cleanup")
    parser.add_argument("--keep-profiles", action="store_true", help="Keep households/children, only delete stories")
    parser.add_argument("--db-only", action="store_true", help="Only clean database, skip blob storage")
    parser.add_argument("--blob-only", action="store_true", help="Only clean blob storage, skip database")
    args = parser.parse_args()

    print("=" * 60)
    print("  LORE LANTERN - CLEAN SLATE AZURE SCRIPT")
    print("=" * 60)

    mode_parts = []
    if args.dry_run:
        mode_parts.append("DRY RUN")
    if args.keep_profiles:
        mode_parts.append("KEEP PROFILES")
    if args.db_only:
        mode_parts.append("DB ONLY")
    if args.blob_only:
        mode_parts.append("BLOB ONLY")

    if mode_parts:
        print(f"  MODE: {' + '.join(mode_parts)}")
    else:
        print("  MODE: FULL WIPE (will delete ALL data!)")
    print()

    # Get configuration
    config = get_azure_config()

    # Connect to services
    conn = None
    blob_service = None
    container_client = None

    if not args.blob_only:
        conn = get_sql_connection(config)

    if not args.db_only:
        blob_service, container_client = get_blob_client(config)

    # List current data
    sql_data = {}
    blob_count = 0
    blob_size = 0

    if conn:
        sql_data = list_sql_data(conn)

    if container_client:
        blob_count, blob_size = list_blob_data(container_client)

    # Display SQL data
    print("\n--- Azure SQL Database ---")
    if sql_data:
        for table, count in sql_data.items():
            status = "rows" if count != 1 else "row"
            marker = " [KEEP]" if args.keep_profiles and table in PROFILE_TABLES else ""
            print(f"  {table}: {count} {status}{marker}")
        print(f"  Subtotal: {sum(sql_data.values())} rows")
    else:
        print("  (empty)")

    # Display Blob data
    print("\n--- Azure Blob Storage ---")
    if blob_count > 0:
        print(f"  Container: {config['blob_container']}")
        print(f"  Files: {blob_count}")
        print(f"  Total size: {format_size(blob_size)}")
    else:
        print("  (empty or not connected)")

    total_items = sum(sql_data.values()) + blob_count
    print(f"\n  GRAND TOTAL: {total_items} items")

    if total_items == 0:
        print("\n  Database and blob storage are already empty!")
        if not args.no_sample and not args.dry_run and conn:
            create_sample_data(conn)
        if conn:
            conn.close()
        return

    # Determine what to delete
    tables_to_delete = []
    delete_blobs = not args.db_only

    if args.blob_only:
        tables_to_delete = []
        delete_description = "blob storage only"
    elif args.keep_profiles:
        tables_to_delete = [t for t in TABLES_DELETE_ORDER if t in STORY_TABLES]
        delete_description = "stories + blobs (keeping profiles)"
    else:
        tables_to_delete = TABLES_DELETE_ORDER
        delete_description = "ALL data"

    if args.db_only:
        delete_blobs = False
        delete_description = delete_description.replace(" + blobs", "")

    # Confirm deletion
    if not args.confirm and not args.dry_run:
        print("\n" + "!" * 60)
        print(f"  WARNING: This will DELETE {delete_description} permanently!")
        print("!" * 60)
        response = input("\n  Type 'DELETE' to confirm: ")
        if response != "DELETE":
            print("\n  Aborted. No changes made.")
            if conn:
                conn.close()
            return

    # Delete data
    print(f"\n--- Deleting {delete_description} ---")
    deleted_rows = 0
    deleted_blobs = 0

    # Delete SQL data (in proper order for foreign keys)
    if tables_to_delete and conn:
        for table in tables_to_delete:
            if table in sql_data:
                count = delete_table_data(conn, table, dry_run=args.dry_run)
                action = "Would delete" if args.dry_run else "Deleted"
                if count > 0:
                    print(f"  {action} {table}: {count} rows")
                    deleted_rows += count

    # Delete blob data
    if delete_blobs and container_client and blob_count > 0:
        count, size = delete_blob_data(container_client, dry_run=args.dry_run)
        action = "Would delete" if args.dry_run else "Deleted"
        print(f"  {action} blobs: {count} files ({format_size(size)})")
        deleted_blobs = count

    if args.dry_run:
        print(f"\n  DRY RUN: Would delete {deleted_rows} rows + {deleted_blobs} blobs")
    else:
        print(f"\n  Deleted {deleted_rows} rows + {deleted_blobs} blobs")

        # Create sample data (only if full wipe)
        if not args.no_sample and not args.keep_profiles and not args.blob_only and conn:
            create_sample_data(conn)

    # Clean up
    if conn:
        conn.close()

    print("\n" + "=" * 60)
    print("  CLEAN SLATE AZURE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
