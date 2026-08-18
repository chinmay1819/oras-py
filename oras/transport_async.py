"""
Asynchronous HTTP transport for registry interactions.

This is the async counterpart to :class:`oras.transport.Transport`. It has the
same responsibility - own the connection and carry out one request - and the
same deliberate lack of responsibility: it knows nothing about manifests,
blobs or media types, and it decides neither when a request is authenticated
nor when one is retried.

httpx is imported lazily so that the async extra stays optional and importing
oras keeps working without it.
"""

__copyright__ = "Copyright The ORAS Authors."
__license__ = "Apache-2.0"

from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Union

if TYPE_CHECKING:
    import httpx

ASYNC_EXTRA_MESSAGE = """the `httpx` dependency is required for asynchronous support.
Make sure to install the required extra "async", e.g.: pip install oras[async].
"""


def get_httpx():
    """
    Import httpx, with a message pointing at the extra when it is missing.
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError(ASYNC_EXTRA_MESSAGE) from e
    return httpx


class AsyncTransport:
    """
    Send HTTP requests to a registry with an httpx async client.
    """

    def __init__(
        self,
        client: Optional["httpx.AsyncClient"] = None,
        tls_verify: Union[bool, str] = True,
    ):
        """
        Create a new async transport.

        The client is created once and reused, so connections are pooled for
        the lifetime of the transport rather than per request.

        :param client: an existing httpx client to use, one is created if not provided
        :type client: httpx.AsyncClient
        :param tls_verify: enable/disable tls verification or use a custom CA-Bundle
        :type tls_verify: bool or str
        """
        httpx_module = get_httpx()
        self.tls_verify = tls_verify

        # requests follows redirects by default and the sync transport relies on
        # that, notably for registries that redirect blob uploads. httpx does
        # not, so it is asked to here to keep the two behaving the same way.
        self.client: "httpx.AsyncClient" = client or httpx_module.AsyncClient(
            verify=tls_verify, follow_redirects=True
        )

    def _forget_cookies(self):
        """
        Drop any cookies the registry tried to set.

        Some registries take an accepted cookie as a sign they are talking to a
        browser and start requiring CSRF tokens (Harbor is such a case). The
        sync transport refuses cookies with a cookie policy; httpx has no
        equivalent, so the jar is emptied instead.
        """
        self.client.cookies.clear()

    def _content_arguments(self, data: Any) -> dict:
        """
        Map a body onto the argument httpx expects for it.

        requests takes bytes, iterables and form dicts all through `data`,
        while httpx separates raw bodies (`content`) from form data (`data`).
        """
        if data is None:
            return {}
        if isinstance(data, dict):
            return {"data": data}
        return {"content": data}

    async def request(
        self,
        url: str,
        method: str = "GET",
        data: Optional[Any] = None,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> "httpx.Response":
        """
        Send a single request, without any authentication or retry handling.

        :param url: the URL to issue the request to
        :type url: str
        :param method: the method to use (GET, DELETE, POST, PUT, PATCH)
        :type method: str
        :param data: body for the request. Bytes, or an iterator of bytes for a
                     body that should not be held in memory
        :type data: bytes or iterator or dict
        :param headers: headers for the request
        :type headers: dict
        :param json: json data for the request
        :type json: dict
        :param params: query string parameters to add to the url
        :type params: dict
        """
        response = await self.client.request(
            method,
            url,
            headers=headers,
            json=json,
            params=params,
            **self._content_arguments(data),
        )
        self._forget_cookies()
        return response

    def stream(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        params: Optional[dict] = None,
    ):
        """
        Send a request and keep the body unread, for streaming a response.

        Returns an async context manager yielding the response, so that a
        caller can consume it in chunks without holding a whole blob in
        memory. Streaming lives here, and not in the registry layer, so that
        the registry never depends on the HTTP client being used.

        :param url: the URL to issue the request to
        :type url: str
        :param method: the method to use
        :type method: str
        :param headers: headers for the request
        :type headers: dict
        :param params: query string parameters to add to the url
        :type params: dict
        """
        return self.client.stream(method, url, headers=headers, params=params)

    async def aclose(self):
        """
        Close the underlying client and release its connections.
        """
        await self.client.aclose()


async def iter_file(path: str, chunk_size: int) -> AsyncIterator[bytes]:
    """
    Read a file in chunks, for uploading it without reading it all at once.

    :param path: the file to read
    :type path: str
    :param chunk_size: how much to read at a time
    :type chunk_size: int
    """
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            yield chunk
