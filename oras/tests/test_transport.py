"""
Tests for the transport boundary, and the registry operations built on it.

These do not need a running registry: the transport can be swapped for a fake
one, which is the point of keeping request execution behind a small interface.
"""

import requests

import oras.defaults
import oras.provider
from oras.transport import Transport, successful_response


def make_response(status_code: int = 200, content: bytes = b"", headers=None):
    """
    Build a requests response to hand back from a fake transport.
    """
    response = requests.Response()
    response.status_code = status_code
    response._content = content
    response.headers.update(headers or {})
    return response


class FakeSession:
    """Records requests instead of sending them."""

    def __init__(self, response=None):
        self.calls = []
        self.cookies = requests.cookies.RequestsCookieJar()
        self.response = response or make_response()

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


class FakeTransport:
    """Stands in for a Transport, returning canned responses in order."""

    def __init__(self, responses=None):
        self.session = None
        self.tls_verify = True
        self.calls = []
        self.responses = list(responses or [])

    def request(
        self, url, method="GET", data=None, headers=None, json=None, stream=False
    ):
        self.calls.append(
            {
                "url": url,
                "method": method,
                "data": data,
                "headers": headers,
                "json": json,
                "stream": stream,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return make_response()


def get_registry(transport=None):
    """
    An insecure registry, with an optional transport swapped in.
    """
    registry = oras.provider.Registry(hostname="registry.example", insecure=True)
    if transport is not None:
        registry.transport = transport
    return registry


def test_transport_request_forwards_arguments():
    session = FakeSession()
    transport = Transport(session=session, tls_verify=False)

    transport.request(
        "http://registry.example/v2/",
        "PUT",
        data=b"content",
        headers={"Content-Type": "application/octet-stream"},
        stream=True,
    )

    (call,) = session.calls
    assert call["method"] == "PUT"
    assert call["url"] == "http://registry.example/v2/"
    assert call["data"] == b"content"
    assert call["headers"] == {"Content-Type": "application/octet-stream"}
    assert call["stream"] is True

    # tls_verify is applied by the transport, not by callers
    assert call["verify"] is False


def test_transport_applies_custom_ca_bundle():
    session = FakeSession()
    transport = Transport(session=session, tls_verify="/path/to/ca-bundle.crt")

    transport.request("https://registry.example/v2/")

    assert session.calls[0]["verify"] == "/path/to/ca-bundle.crt"


def test_transport_rejects_cookies():
    """
    Registries that see cookies accepted may treat the client as a browser.
    """
    transport = Transport()
    assert transport.session.cookies.get_policy().allowed_domains() == ()


def test_successful_response():
    assert successful_response().status_code == 200
    assert successful_response(201).status_code == 201


def test_registry_session_comes_from_transport():
    registry = get_registry()

    assert registry.session is registry.transport.session

    # The auth backend shares the session with the provider
    assert registry.auth.session is registry.session


def test_registry_session_can_be_replaced():
    registry = get_registry()
    session = requests.Session()

    registry.session = session

    assert registry.session is session
    assert registry.transport.session is session


def test_registry_tls_verify_reads_and_writes_transport():
    registry = oras.provider.Registry(hostname="registry.example", tls_verify=False)
    assert registry._tls_verify is False

    registry._tls_verify = True
    assert registry.transport.tls_verify is True


def test_do_request_goes_through_the_transport():
    transport = FakeTransport([make_response(201)])
    registry = get_registry(transport)

    response = registry.do_request(
        "http://registry.example/v2/repository/manifests/tag", "PUT", json={"a": 1}
    )

    assert response.status_code == 201
    (call,) = transport.calls
    assert call["method"] == "PUT"
    assert call["url"] == "http://registry.example/v2/repository/manifests/tag"
    assert call["json"] == {"a": 1}


def test_get_manifest_content_returns_bytes_and_digest():
    digest = "sha256:aaaa"
    transport = FakeTransport(
        [
            make_response(
                content=b'{"schemaVersion":2}',
                headers={"Docker-Content-Digest": digest},
            )
        ]
    )
    registry = get_registry(transport)

    content, reported_digest = registry.get_manifest_content(
        "registry.example/dinosaur/artifact:v1"
    )

    # The bytes are handed back untouched, so the digest still matches
    assert content == b'{"schemaVersion":2}'
    assert reported_digest == digest

    (call,) = transport.calls
    assert call["method"] == "GET"
    assert call["url"] == "http://registry.example/v2/dinosaur/artifact/manifests/v1"
    assert call["headers"]["Accept"] == ", ".join(
        oras.defaults.default_manifest_accepted_media_types
    )


def test_get_manifest_content_without_digest_header():
    transport = FakeTransport([make_response(content=b"{}")])
    registry = get_registry(transport)

    _, reported_digest = registry.get_manifest_content(
        "registry.example/dinosaur/artifact:v1"
    )

    assert reported_digest is None


def test_get_manifest_content_by_reference():
    digest = "sha256:bbbb"
    transport = FakeTransport([make_response(content=b"{}")])
    registry = get_registry(transport)

    registry.get_manifest_content(
        "registry.example/dinosaur/artifact:v1",
        allowed_media_type=[oras.defaults.default_manifest_media_type],
        reference=digest,
    )

    (call,) = transport.calls
    assert (
        call["url"]
        == f"http://registry.example/v2/dinosaur/artifact/manifests/{digest}"
    )
    assert call["headers"]["Accept"] == oras.defaults.default_manifest_media_type


def test_upload_manifest_content_sends_raw_bytes():
    transport = FakeTransport([make_response(201)])
    registry = get_registry(transport)
    container = registry.get_container("registry.example/dinosaur/artifact:v1")

    response = registry.upload_manifest_content(
        b'{"schemaVersion":2}',
        container,
        oras.defaults.default_index_media_type,
    )

    assert response.status_code == 201
    (call,) = transport.calls
    assert call["method"] == "PUT"
    assert call["data"] == b'{"schemaVersion":2}'
    assert call["json"] is None
    assert call["headers"] == {"Content-Type": oras.defaults.default_index_media_type}
    assert call["url"] == "http://registry.example/v2/dinosaur/artifact/manifests/v1"


def test_upload_manifest_content_by_reference():
    digest = "sha256:cccc"
    transport = FakeTransport([make_response(201)])
    registry = get_registry(transport)
    container = registry.get_container("registry.example/dinosaur/artifact:v1")

    registry.upload_manifest_content(
        b"{}",
        container,
        oras.defaults.default_manifest_media_type,
        reference=digest,
    )

    (call,) = transport.calls
    assert (
        call["url"]
        == f"http://registry.example/v2/dinosaur/artifact/manifests/{digest}"
    )
