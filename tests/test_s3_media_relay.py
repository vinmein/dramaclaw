from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from fastapi import FastAPI
from fastapi.testclient import TestClient

from novelvideo import model_gateway_settings as settings
from novelvideo.api.routes import model_gateway
from novelvideo.storage.media_relay import (
    S3Relay,
    ServiceEgressDenied,
    MediaRelayConfigError,
    get_media_relay,
)


@pytest.fixture
def s3_client(monkeypatch):
    client = boto3.client(
        "s3",
        region_name="ap-south-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
    )
    monkeypatch.setattr(boto3, "client", lambda *a, **kw: client)
    return client


def relay():
    return S3Relay(
        region="ap-south-1",
        bucket_name="test-media",
        access_key_id="test-key",
        access_key_secret="test-secret",
    )


def test_private_upload_and_signed_get(s3_client):
    with Stubber(s3_client) as stub:
        stub.add_response(
            "put_object",
            {},
            {
                "Bucket": "test-media",
                "Key": "relay/example.mp4",
                "Body": b"video",
                "ContentType": "video/mp4",
            },
        )
        signer = Mock(return_value="https://signed.example/video")
        s3_client.generate_presigned_url = signer
        assert (
            relay().upload_bytes(
                b"video",
                ext="mp4",
                ttl=900,
                resource_type="video",
                object_key="relay/example.mp4",
            )
            == "https://signed.example/video"
        )
        signer.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-media", "Key": "relay/example.mp4"},
            ExpiresIn=900,
            HttpMethod="GET",
        )
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("ttl", [0, -1, 604801])
def test_rejects_invalid_ttl_before_upload(s3_client, ttl):
    with Stubber(s3_client):
        with pytest.raises(ValueError):
            relay().upload_bytes(b"image", ttl=ttl)


def test_rejects_unsafe_key_and_empty_upload(s3_client):
    with Stubber(s3_client):
        with pytest.raises(ServiceEgressDenied):
            relay().upload_bytes(b"image", object_key="../escape.png")
        with pytest.raises(ValueError):
            relay().upload_bytes(b"")


def test_upload_error_does_not_leak_secrets(s3_client):
    s3_client.put_object = Mock(
        side_effect=ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "test-secret"}}, "PutObject"
        )
    )
    with pytest.raises(MediaRelayConfigError) as error:
        relay().upload_bytes(b"image")
    assert "test-secret" not in str(error.value)


@pytest.fixture
def settings_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ST_EDITION", "ce")
    monkeypatch.delenv("ST_CONTROL_PLANE_DSN", raising=False)
    monkeypatch.setattr(settings, "_settings_db_path", lambda: tmp_path / "settings.db")
    for key in ("REGION", "BUCKET", "AK", "SK"):
        monkeypatch.delenv("S3_RELAY_" + key, raising=False)
    app = FastAPI()
    app.include_router(model_gateway.router)
    return TestClient(app)


def test_save_mask_preserve_rotate_and_switch(settings_client, s3_client):
    path = "/model-gateway/media-relay/config"
    payload = {
        "provider": "aws_s3",
        "ttlSeconds": 900,
        "s3Region": "ap-south-1",
        "s3Bucket": "test-media",
        "s3AccessKeyId": "test-key-secret",
        "s3AccessKeySecret": "test-secret-private",
    }
    response = settings_client.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["configured"] is True
    assert "test-key-secret" not in response.text
    assert "test-secret-private" not in response.text
    response = settings_client.post(
        path,
        json={
            "provider": "aws_s3",
            "ttlSeconds": 1800,
            "s3AccessKeyId": "",
            "s3AccessKeySecret": "",
        },
    )
    assert response.status_code == 200
    assert (
        settings.get_effective_media_relay_config().s3_access_key_secret
        == "test-secret-private"
    )
    assert isinstance(get_media_relay(), S3Relay)
    settings_client.post(
        path, json={"provider": "aws_s3", "s3AccessKeySecret": "rotated-secret"}
    )
    assert (
        settings.get_effective_media_relay_config().s3_access_key_secret
        == "rotated-secret"
    )
    response = settings_client.post(
        path,
        json={
            "provider": "cloudinary",
            "cloudName": "demo",
            "apiKey": "cloud-key",
            "apiSecret": "cloud-secret",
        },
    )
    assert response.status_code == 200
    response = settings_client.post(path, json={"provider": "aws_s3"})
    assert response.json()["data"]["configured"] is True
    assert (
        settings.get_effective_media_relay_config().s3_access_key_secret
        == "rotated-secret"
    )


def test_settings_reject_missing_fields_and_excessive_ttl(settings_client):
    path = "/model-gateway/media-relay/config"
    assert settings_client.post(path, json={"provider": "aws_s3"}).status_code == 400
    response = settings_client.post(
        path, json={"provider": "aws_s3", "ttlSeconds": 604801}
    )
    assert response.status_code == 400


def test_environment_configuration(settings_client, monkeypatch):
    for key, value in {
        "REGION": "ap-south-1",
        "BUCKET": "test-media",
        "AK": "test-key",
        "SK": "test-secret",
    }.items():
        monkeypatch.setenv("S3_RELAY_" + key, value)
    cfg = settings.get_effective_media_relay_config(env_provider="aws_s3")
    assert cfg.source == "environment"
    assert cfg.s3_bucket == "test-media"
    assert cfg.s3_region == "ap-south-1"
    assert (
        settings.build_media_relay_status(env_provider="aws_s3")["configured"] is True
    )
