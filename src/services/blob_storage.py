"""
Azure Blob Storage service for LoreLantern.

Handles audio file storage for TTS-generated chapter audio.
Replaces Firebase Storage / inline base64 audio.
"""

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
from azure.core.exceptions import ResourceNotFoundError
from datetime import datetime, timedelta
from typing import Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor


class BlobStorageService:
    """Azure Blob Storage service for audio files."""

    def __init__(
        self,
        connection_string: str,
        container_name: str = "lorelantern-audio",
        logger=None
    ):
        """
        Initialize blob storage service.

        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the blob container
            logger: Optional logger instance
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.logger = logger
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._initialized = False

        # Parse account info from connection string for SAS generation
        self._account_name = None
        self._account_key = None
        for part in connection_string.split(';'):
            if part.startswith('AccountName='):
                self._account_name = part.split('=', 1)[1]
            elif part.startswith('AccountKey='):
                self._account_key = part.split('=', 1)[1]

    def initialize(self):
        """Initialize the blob storage client."""
        if self._initialized:
            return

        try:
            self.client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container = self.client.get_container_client(self.container_name)

            # Verify container exists
            if not self.container.exists():
                print(f"   Creating container: {self.container_name}")
                self.container.create_container()

            self._initialized = True
            print(f"   Azure Blob Storage initialized: {self.container_name}")
        except Exception as e:
            print(f"   Warning: Blob Storage initialization failed: {e}")
            raise

    async def _run_async(self, func, *args, **kwargs):
        """Run a sync function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )

    def _get_blob_path(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int,
        segment: Optional[int] = None
    ) -> str:
        """
        Generate blob path for audio file.

        Structure: {household_id}/{story_id}/chapter_{num}[_segment_{n}].mp3
        """
        if segment is not None:
            return f"{household_id}/{story_id}/chapter_{chapter_num}_segment_{segment}.mp3"
        return f"{household_id}/{story_id}/chapter_{chapter_num}.mp3"

    def _upload_audio_sync(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int,
        audio_data: bytes,
        segment: Optional[int] = None
    ) -> str:
        """Upload audio data and return blob URL."""
        blob_path = self._get_blob_path(household_id, story_id, chapter_num, segment)
        blob_client = self.container.get_blob_client(blob_path)

        # Use ContentSettings object (not dict) - SDK requires object with cache_control attribute
        content_settings = ContentSettings(content_type='audio/mpeg')
        blob_client.upload_blob(
            audio_data,
            overwrite=True,
            content_settings=content_settings
        )

        return blob_client.url

    async def upload_audio(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int,
        audio_data: bytes,
        segment: Optional[int] = None
    ) -> str:
        """
        Upload audio data and return blob URL.

        Args:
            household_id: Household ID for path
            story_id: Story ID for path
            chapter_num: Chapter number
            audio_data: MP3 audio bytes
            segment: Optional segment number for chunked audio

        Returns:
            Blob URL
        """
        return await self._run_async(
            self._upload_audio_sync,
            household_id, story_id, chapter_num, audio_data, segment
        )

    def _get_audio_url_sync(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int,
        segment: Optional[int] = None,
        expiry_hours: int = 24
    ) -> Optional[str]:
        """Get SAS URL for audio playback."""
        blob_path = self._get_blob_path(household_id, story_id, chapter_num, segment)
        blob_client = self.container.get_blob_client(blob_path)

        # Check if blob exists
        if not blob_client.exists():
            return None

        # Generate SAS token for secure access
        sas_token = generate_blob_sas(
            account_name=self._account_name,
            container_name=self.container_name,
            blob_name=blob_path,
            account_key=self._account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
        )

        return f"{blob_client.url}?{sas_token}"

    async def get_audio_url(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int,
        segment: Optional[int] = None,
        expiry_hours: int = 24
    ) -> Optional[str]:
        """
        Get SAS URL for audio playback.

        Args:
            household_id: Household ID
            story_id: Story ID
            chapter_num: Chapter number
            segment: Optional segment number
            expiry_hours: How long the URL should be valid (default 24h)

        Returns:
            SAS URL for audio playback, or None if not found
        """
        return await self._run_async(
            self._get_audio_url_sync,
            household_id, story_id, chapter_num, segment, expiry_hours
        )

    def _delete_story_audio_sync(self, household_id: str, story_id: str) -> int:
        """Delete all audio files for a story."""
        prefix = f"{household_id}/{story_id}/"
        deleted_count = 0

        blobs = self.container.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            blob_client = self.container.get_blob_client(blob.name)
            blob_client.delete_blob()
            deleted_count += 1

        return deleted_count

    async def delete_story_audio(self, household_id: str, story_id: str) -> int:
        """
        Delete all audio files for a story.

        Args:
            household_id: Household ID
            story_id: Story ID

        Returns:
            Number of files deleted
        """
        return await self._run_async(
            self._delete_story_audio_sync,
            household_id, story_id
        )

    def _delete_chapter_audio_sync(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int
    ) -> int:
        """Delete all audio files for a chapter."""
        prefix = f"{household_id}/{story_id}/chapter_{chapter_num}"
        deleted_count = 0

        blobs = self.container.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            blob_client = self.container.get_blob_client(blob.name)
            blob_client.delete_blob()
            deleted_count += 1

        return deleted_count

    async def delete_chapter_audio(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int
    ) -> int:
        """
        Delete all audio files for a chapter.

        Args:
            household_id: Household ID
            story_id: Story ID
            chapter_num: Chapter number

        Returns:
            Number of files deleted
        """
        return await self._run_async(
            self._delete_chapter_audio_sync,
            household_id, story_id, chapter_num
        )

    def _list_chapter_segments_sync(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int
    ) -> List[str]:
        """List all segment URLs for a chapter."""
        prefix = f"{household_id}/{story_id}/chapter_{chapter_num}"
        segments = []

        blobs = self.container.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            # Generate SAS URL for each segment
            sas_token = generate_blob_sas(
                account_name=self._account_name,
                container_name=self.container_name,
                blob_name=blob.name,
                account_key=self._account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=24)
            )
            blob_client = self.container.get_blob_client(blob.name)
            segments.append(f"{blob_client.url}?{sas_token}")

        # Sort by segment number
        segments.sort()
        return segments

    async def list_chapter_segments(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int
    ) -> List[str]:
        """
        List all audio segment URLs for a chapter.

        Args:
            household_id: Household ID
            story_id: Story ID
            chapter_num: Chapter number

        Returns:
            List of SAS URLs for audio segments, sorted by segment number
        """
        return await self._run_async(
            self._list_chapter_segments_sync,
            household_id, story_id, chapter_num
        )

    def _check_audio_exists_sync(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int,
        segment: Optional[int] = None
    ) -> bool:
        """Check if audio file exists."""
        blob_path = self._get_blob_path(household_id, story_id, chapter_num, segment)
        blob_client = self.container.get_blob_client(blob_path)
        return blob_client.exists()

    async def check_audio_exists(
        self,
        household_id: str,
        story_id: str,
        chapter_num: int,
        segment: Optional[int] = None
    ) -> bool:
        """
        Check if audio file exists.

        Args:
            household_id: Household ID
            story_id: Story ID
            chapter_num: Chapter number
            segment: Optional segment number

        Returns:
            True if audio exists
        """
        return await self._run_async(
            self._check_audio_exists_sync,
            household_id, story_id, chapter_num, segment
        )

    async def get_storage_stats(self, household_id: str) -> dict:
        """
        Get storage statistics for a household.

        Returns:
            Dict with total_files, total_size_mb, stories
        """
        def _get_stats():
            prefix = f"{household_id}/"
            total_size = 0
            total_files = 0
            stories = set()

            blobs = self.container.list_blobs(name_starts_with=prefix)
            for blob in blobs:
                total_files += 1
                total_size += blob.size
                # Extract story_id from path
                parts = blob.name.split('/')
                if len(parts) >= 2:
                    stories.add(parts[1])

            return {
                'total_files': total_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'story_count': len(stories)
            }

        return await self._run_async(_get_stats)
