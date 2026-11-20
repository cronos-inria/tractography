"""Download example datasets used by tractography workflows."""
from pathlib import Path, PurePosixPath, PureWindowsPath
from shutil import copyfileobj, rmtree
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile, is_zipfile

from platformdirs import user_cache_path


_PCLOUD_LINK_CODE = "kZQcDV7ZaDfd2biArjL3KDHSoqMcCJsgqmck"
_PCLOUD_DOWNLOAD_API = "https://eapi.pcloud.com/getpubzip"
_ARCHIVE_ROOT_NAME = "iflow"
_DOWNLOAD_MARKER = ".tractography-download-complete"


def download(directory: str | Path | None = None, *, force: bool = False) -> Path:
    """Download and extract the public iflow dataset.

    By default, the archive is extracted into the user's tractography data
    directory at ``<cache>/tractography/data``. Since the archive contains an
    ``iflow`` folder, this function returns ``<cache>/tractography/data/iflow``.
    Pass ``directory`` to choose another extraction directory. An existing
    completed download is reused unless ``force`` is true.

    Returns:
        The path to the directory containing the extracted dataset.
    """
    extraction_directory = (
        Path(directory).expanduser()
        if directory is not None
        else user_cache_path("tractography") / "data"
    )
    dataset_directory = extraction_directory / _ARCHIVE_ROOT_NAME
    marker = dataset_directory / _DOWNLOAD_MARKER
    if marker.is_file() and not force:
        return dataset_directory

    extraction_directory.mkdir(parents=True, exist_ok=True)
    request_url = f"{_PCLOUD_DOWNLOAD_API}?{urlencode({'code': _PCLOUD_LINK_CODE})}"

    with TemporaryDirectory(dir=extraction_directory) as temporary_directory:
        temporary_directory = Path(temporary_directory)
        archive_path = temporary_directory / "iflow.zip"

        with urlopen(request_url, timeout=60) as response, archive_path.open("wb") as archive:
            copyfileobj(response, archive)

        if not is_zipfile(archive_path):
            details = archive_path.read_bytes()[:500].decode("utf-8", errors="replace")
            raise RuntimeError(f"pCloud did not return a ZIP archive: {details}")

        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                posix_path = PurePosixPath(member.filename)
                windows_path = PureWindowsPath(member.filename)
                if (
                    posix_path.is_absolute()
                    or windows_path.is_absolute()
                    or windows_path.drive
                    or ".." in posix_path.parts
                    or ".." in windows_path.parts
                ):
                    raise RuntimeError(f"Unsafe path in pCloud archive: {member.filename}")
            # The ZIP already contains its top-level "iflow" directory, so
            # extract it into the cache data directory rather than into a
            # second directory named "iflow".
            if dataset_directory.exists():
                if dataset_directory.is_dir() and not dataset_directory.is_symlink():
                    rmtree(dataset_directory)
                else:
                    dataset_directory.unlink()
            archive.extractall(extraction_directory)

        if not dataset_directory.is_dir() or not any(dataset_directory.iterdir()):
            raise RuntimeError("The pCloud archive did not contain an iflow dataset.")
        (dataset_directory / _DOWNLOAD_MARKER).touch()

    return dataset_directory

