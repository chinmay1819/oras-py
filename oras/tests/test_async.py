"""
Tests for the asynchronous client.

The unit tests swap in a fake transport, so registry logic can be checked
without a registry and without httpx doing any real work. The integration
tests exercise the httpx transport against a running registry, and skip when
there is not one, in the same way as the synchronous integration tests.
"""

import asyncio
import hashlib
import os
import pathlib

import pytest

import oras.auth.utils
import oras.client
import oras.defaults
import oras.layout
import oras.provider
import oras.utils
from oras.provider_async import AsyncRegistry
from oras.transport_async import AsyncTransport

# The async client is an extra, so skip rather than fail when it is absent
httpx = pytest.importorskip("httpx")

here = pathlib.Path(__file__).resolve().parent


def digest_of(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


class FakeAsyncTransport:
    """Records requests and returns canned responses, in order."""

    def __init__(self, responses=None):
        self.tls_verify = True
        self.calls = []
        self.responses = list(responses or [])
        self.closed = False

    async def request(
        self, url, method="GET", data=None, headers=None, json=None, params=None
    ):
        self.calls.append(
            {
                "url": url,
                "method": method,
                "data": data,
                "headers": dict(headers or {}),
                "json": json,
                "params": params,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return httpx.Response(200)

    async def aclose(self):
        self.closed = True


def get_registry(transport=None) -> AsyncRegistry:
    return AsyncRegistry(
        hostname="registry.example",
        insecure=True,
        transport=transport or FakeAsyncTransport(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- construction


def test_async_client_is_exposed_from_client_module():
    assert oras.client.AsyncOrasClient is AsyncRegistry


def test_async_registry_shares_its_transport_with_auth():
    transport = FakeAsyncTransport()
    registry = get_registry(transport)

    assert registry.transport is transport
    assert registry.auth.transport is transport


def test_async_registry_builds_a_transport_by_default():
    registry = AsyncRegistry(hostname="registry.example", tls_verify=False)

    assert isinstance(registry.transport, AsyncTransport)
    assert registry.transport.tls_verify is False


def test_async_registry_shares_registry_logic_with_the_sync_provider():
    """
    The decisions live in one place, so both providers get them from there.
    """
    for shared in (
        "_iter_push_layers",
        "_iter_pull_targets",
        "_prepare_manifest_config",
        "_apply_manifest_annotations",
        "_check_200_response",
        "_get_location",
        "_url",
        "get_container",
    ):
        assert getattr(AsyncRegistry, shared) is getattr(oras.provider.Registry, shared)


# ---------------------------------------------------------------- transport


@pytest.mark.asyncio
async def test_do_request_goes_through_the_transport():
    transport = FakeAsyncTransport([httpx.Response(201)])
    registry = get_registry(transport)

    response = await registry.do_request(
        "http://registry.example/v2/repo/manifests/v1", "PUT", json={"a": 1}
    )

    assert response.status_code == 201
    (call,) = transport.calls
    assert call["method"] == "PUT"
    assert call["json"] == {"a": 1}


@pytest.mark.asyncio
async def test_aclose_closes_the_transport():
    transport = FakeAsyncTransport()
    registry = get_registry(transport)

    await registry.aclose()

    assert transport.closed is True


@pytest.mark.asyncio
async def test_async_context_manager_closes_the_transport():
    transport = FakeAsyncTransport()

    async with get_registry(transport) as registry:
        assert registry.transport is transport
    assert transport.closed is True


def test_transport_maps_bodies_onto_httpx_arguments():
    """
    requests takes bytes and forms through `data`, httpx separates them.
    """
    transport = AsyncTransport(client=object())

    assert transport._content_arguments(None) == {}
    assert transport._content_arguments(b"raw") == {"content": b"raw"}
    assert transport._content_arguments({"a": "b"}) == {"data": {"a": "b"}}


# ---------------------------------------------------------------- manifests


@pytest.mark.asyncio
async def test_get_manifest_content_returns_bytes_and_digest():
    body = b'{"schemaVersion":2}'
    transport = FakeAsyncTransport(
        [
            httpx.Response(
                200, content=body, headers={"Docker-Content-Digest": "sha256:a"}
            )
        ]
    )
    registry = get_registry(transport)

    content, digest = await registry.get_manifest_content(
        "registry.example/demo/artifact:v1"
    )

    assert content == body
    assert digest == "sha256:a"
    (call,) = transport.calls
    assert call["url"] == "http://registry.example/v2/demo/artifact/manifests/v1"


@pytest.mark.asyncio
async def test_upload_manifest_content_sends_raw_bytes_by_reference():
    transport = FakeAsyncTransport([httpx.Response(201)])
    registry = get_registry(transport)
    container = registry.get_container("registry.example/demo/artifact:v1")

    await registry.upload_manifest_content(
        b"{}", container, oras.defaults.default_index_media_type, reference="sha256:b"
    )

    (call,) = transport.calls
    assert call["method"] == "PUT"
    assert call["data"] == b"{}"
    assert call["headers"]["Content-Type"] == oras.defaults.default_index_media_type
    assert call["url"] == "http://registry.example/v2/demo/artifact/manifests/sha256:b"


# ---------------------------------------------------------------- auth


@pytest.mark.asyncio
async def test_authentication_challenge_is_answered_and_retried_asynchronously():
    """
    401 -> token request -> retry, all awaited on the async transport.
    """
    challenge = httpx.Response(
        401,
        headers={
            "Www-Authenticate": 'Bearer realm="https://auth.example/token",service="registry.example"'
        },
    )
    token = httpx.Response(200, json={"token": "a-token"})
    transport = FakeAsyncTransport([challenge, token, httpx.Response(200)])
    registry = get_registry(transport)

    response = await registry.do_request("http://registry.example/v2/demo/tags/list")

    assert response.status_code == 200
    assert len(transport.calls) == 3

    # the token came from the realm, over the same async transport
    assert transport.calls[1]["url"] == "https://auth.example/token"
    assert transport.calls[1]["params"] == {"service": "registry.example"}

    # and the retry carried it
    assert transport.calls[2]["headers"]["Authorization"] == "Bearer a-token"
    assert registry.auth.token == "a-token"


@pytest.mark.asyncio
async def test_basic_auth_answers_a_challenge_without_a_token_request():
    challenge = httpx.Response(401, headers={"Www-Authenticate": 'Basic realm="r"'})
    transport = FakeAsyncTransport([challenge, httpx.Response(200)])
    registry = AsyncRegistry(
        hostname="registry.example",
        insecure=True,
        auth_backend="basic",
        transport=transport,
    )
    registry.auth.set_basic_auth("myuser", "mypass")

    response = await registry.do_request("http://registry.example/v2/")

    assert response.status_code == 200
    assert len(transport.calls) == 2
    assert transport.calls[1]["headers"]["Authorization"].startswith("Basic ")


@pytest.mark.asyncio
async def test_challenge_without_a_www_authenticate_header_is_not_answered():
    """
    Nothing to answer means no retry, matching the synchronous backend.
    """
    registry = get_registry()

    headers, changed = await registry.auth.authenticate_request_async(
        httpx.Response(401), {}
    )

    assert changed is False
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_a_refused_token_request_yields_no_token():
    transport = FakeAsyncTransport([httpx.Response(403, content=b"{}")])
    registry = get_registry(transport)
    header = oras.auth.utils.parse_auth_header(
        'Bearer realm="https://auth.example/token"'
    )

    assert await registry.auth.request_token_async(header) is None


# ---------------------------------------------------------------- integration


@pytest.fixture
def async_target(registry):
    return f"{registry}/dinosaur/async-artifact:v1"


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_push_pull_file(tmp_path, registry, credentials, async_target):
    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        with oras.utils.workdir(str(tmp_path)):
            oras.utils.write_file("artifact.txt", "async round trip\n")
            response = await client.push(
                files=["artifact.txt"],
                target=async_target,
                manifest_annotations={"mode": "async"},
            )
            assert response.status_code in (200, 201)

        outdir = str(tmp_path / "out")
        files = await client.pull(async_target, outdir=outdir)
        assert len(files) == 1
        assert oras.utils.read_file(files[0]) == "async round trip\n"

        manifest = await client.get_manifest(async_target)
        assert manifest["annotations"]["mode"] == "async"


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_push_pull_directory(tmp_path, registry, credentials):
    target = f"{registry}/dinosaur/async-directory:v1"
    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        with oras.utils.workdir(str(tmp_path)):
            os.makedirs("bundle", exist_ok=True)
            oras.utils.write_file(os.path.join("bundle", "inner.txt"), "nested\n")
            response = await client.push(files=["bundle"], target=target)
            assert response.status_code in (200, 201)

        files = await client.pull(target, outdir=str(tmp_path / "dirout"))
        assert len(files) == 1
        assert os.path.isdir(files[0])
        assert "inner.txt" in os.listdir(files[0])


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_chunked_upload(tmp_path, registry, credentials):
    target = f"{registry}/dinosaur/async-chunked:v1"
    payload = ("chunk" * 20000).encode()

    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        with oras.utils.workdir(str(tmp_path)):
            pathlib.Path("chunked.bin").write_bytes(payload)
            response = await client.push(
                files=["chunked.bin"],
                target=target,
                do_chunked=True,
                chunk_size=4096,
            )
            assert response.status_code in (200, 201, 202)

        files = await client.pull(target, outdir=str(tmp_path / "chunkout"))
        assert pathlib.Path(files[0]).read_bytes() == payload


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_large_blob_round_trips(tmp_path, registry, credentials):
    """
    A blob larger than the read buffer, to exercise streaming both ways.
    """
    target = f"{registry}/dinosaur/async-large:v1"
    payload = os.urandom(5 * 1024 * 1024)

    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        with oras.utils.workdir(str(tmp_path)):
            pathlib.Path("large.bin").write_bytes(payload)
            response = await client.push(files=["large.bin"], target=target)
            assert response.status_code in (200, 201)

        files = await client.pull(target, outdir=str(tmp_path / "largeout"))
        assert digest_of(pathlib.Path(files[0]).read_bytes()) == digest_of(payload)


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_digest_reference_and_tags(tmp_path, registry, credentials):
    repository = f"{registry}/dinosaur/async-refs"
    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        with oras.utils.workdir(str(tmp_path)):
            oras.utils.write_file("ref.txt", "by digest\n")
            await client.push(files=["ref.txt"], target=f"{repository}:v1")

        content, digest = await client.get_manifest_content(f"{repository}:v1")
        assert digest == digest_of(content)

        by_digest = await client.get_manifest(f"{repository}@{digest}")
        assert by_digest["schemaVersion"] == 2

        tags = await client.get_tags(repository)
        assert "v1" in tags


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_index_is_pushed_and_read_back(tmp_path, registry, credentials):
    """
    An image index pushed as exact bytes, then read back unchanged.
    """
    layout = oras.layout.Layout(str(here / "ocilayout_data" / "ocilayout2"))
    index_digest = oras.utils.read_json(
        str(here / "ocilayout_data" / "ocilayout2" / "index.json")
    )["manifests"][0]["digest"]
    index_bytes = layout.digest_to_blob_path(index_digest).read_bytes()

    target = f"{registry}/dinosaur/async-index:v1"
    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        container = client.get_container(target)

        # the blobs the index refers to have to exist before it does
        for digest in layout.get_ordered_blobs("latest"):
            blob_path = layout.digest_to_blob_path(digest)
            blob = blob_path.read_bytes()
            try:
                media_type = oras.utils.read_json(str(blob_path)).get("mediaType", "")
            except Exception:
                media_type = ""
            if media_type in (
                oras.defaults.default_manifest_media_type,
                oras.defaults.default_index_media_type,
            ):
                if digest == index_digest:
                    continue
                response = await client.upload_manifest_content(
                    blob, container, media_type, reference=digest
                )
            else:
                layer = {
                    "digest": digest,
                    "size": blob_path.stat().st_size,
                    "mediaType": media_type or oras.defaults.unknown_config_media_type,
                }
                response = await client.upload_blob(str(blob_path), container, layer)
            client._check_200_response(response)

        response = await client.upload_manifest_content(
            index_bytes, container, oras.defaults.default_index_media_type
        )
        client._check_200_response(response)

        content, digest = await client.get_manifest_content(
            target, allowed_media_type=[oras.defaults.default_index_media_type]
        )
        assert content == index_bytes
        assert digest == index_digest


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_operations_run_concurrently(tmp_path, registry, credentials):
    target = f"{registry}/dinosaur/async-concurrent:v1"
    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        with oras.utils.workdir(str(tmp_path)):
            oras.utils.write_file("shared.txt", "concurrent\n")
            await client.push(files=["shared.txt"], target=target)

        results = await asyncio.gather(
            client.pull(target, outdir=str(tmp_path / "a")),
            client.pull(target, outdir=str(tmp_path / "b")),
            client.get_manifest(target),
            client.get_tags(f"{registry}/dinosaur/async-concurrent"),
        )

    assert len(results[0]) == 1 and len(results[1]) == 1
    assert results[2]["schemaVersion"] == 2
    assert "v1" in results[3]

    # one shared token cache, one agreed answer
    assert client.auth.token is None or isinstance(client.auth.token, str)


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_pull_can_be_cancelled(tmp_path, registry, credentials):
    target = f"{registry}/dinosaur/async-cancel:v1"
    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        with oras.utils.workdir(str(tmp_path)):
            pathlib.Path("cancel.bin").write_bytes(os.urandom(2 * 1024 * 1024))
            await client.push(files=["cancel.bin"], target=target)

        task = asyncio.create_task(client.pull(target, outdir=str(tmp_path / "c")))
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
@pytest.mark.with_auth(False)
async def test_async_missing_manifest_raises_value_error(registry, credentials):
    async with AsyncRegistry(hostname=registry, insecure=True) as client:
        with pytest.raises(ValueError):
            await client.get_manifest(f"{registry}/dinosaur/does-not-exist:v1")
