"""Utilities for listing and downloading PDFs from S3."""

import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class S3PdfObject:
    bucket: str
    key: str
    local_path: str = ""

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri or not s3_uri.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")
    rest = s3_uri[5:]
    if not rest:
        raise ValueError("S3 path is missing bucket name")
    if "/" not in rest:
        return rest, ""
    bucket, prefix = rest.split("/", 1)
    if not bucket:
        raise ValueError("S3 path is missing bucket name")
    return bucket, prefix


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return cleaned or "index"


def build_doc_name(bucket: str, key_or_prefix: str) -> str:
    return slugify(f"{bucket}_{key_or_prefix}")


def _make_s3_client(region_name: Optional[str] = None, profile: Optional[str] = None, endpoint_url: Optional[str] = None):
    import boto3

    if profile:
        session = boto3.Session(profile_name=profile, region_name=region_name)
        return session.client("s3", endpoint_url=endpoint_url)
    return boto3.client("s3", region_name=region_name, endpoint_url=endpoint_url)


def list_pdf_objects(
    s3_uri: str,
    *,
    region_name: Optional[str] = None,
    profile: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> List[S3PdfObject]:
    bucket, prefix = parse_s3_uri(s3_uri)
    s3 = _make_s3_client(region_name=region_name, profile=profile, endpoint_url=endpoint_url)

    items: List[S3PdfObject] = []
    token = None
    while True:
        params = {"Bucket": bucket, "Prefix": prefix}
        if token:
            params["ContinuationToken"] = token
        resp = s3.list_objects_v2(**params)

        for obj in resp.get("Contents", []):
            key = obj.get("Key", "")
            if key.lower().endswith(".pdf"):
                items.append(S3PdfObject(bucket=bucket, key=key))

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    items.sort(key=lambda item: item.key)
    return items


def download_pdf_objects(
    objects: List[S3PdfObject],
    dest_dir: str,
    *,
    region_name: Optional[str] = None,
    profile: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> List[S3PdfObject]:
    os.makedirs(dest_dir, exist_ok=True)
    s3 = _make_s3_client(region_name=region_name, profile=profile, endpoint_url=endpoint_url)

    out: List[S3PdfObject] = []
    for idx, item in enumerate(objects):
        name = os.path.basename(item.key) or f"file_{idx + 1}.pdf"
        local_name = f"{idx:04d}_{name}"
        local_path = os.path.join(dest_dir, local_name)
        s3.download_file(item.bucket, item.key, local_path)
        out.append(S3PdfObject(bucket=item.bucket, key=item.key, local_path=local_path))
    return out
