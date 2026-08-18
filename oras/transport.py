"""
HTTP transport for registry interactions.

This module is the only place where requests are actually sent to a registry,
whether they come from the provider or from an authentication backend.

The transport owns the connection: the session, TLS verification and cookie
policy. It deliberately does not own anything above HTTP. It knows nothing
about manifests, blobs, digests or media types, it does not decide when a
request should be authenticated, and it does not decide when one should be
retried. Those are registry concerns, and they stay in
:class:`oras.provider.Registry`.

Keeping execution in one small place is also what makes a different execution
model possible later: an alternative transport only has to send a request and
return a response, without any registry logic being written a second time.
"""

__copyright__ = "Copyright The ORAS Authors."
__license__ = "Apache-2.0"

from http.cookiejar import DefaultCookiePolicy
from typing import IO, Optional, Union

import requests


class Transport:
    """
    Send HTTP requests to a registry with a requests session.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        tls_verify: Union[bool, str] = True,
    ):
        """
        Create a new transport.

        :param session: an existing session to use, one is created if not provided
        :type session: requests.Session
        :param tls_verify: enable/disable tls verification or use a custom CA-Bundle
        :type tls_verify: bool or str
        """
        self.session: requests.Session = session or requests.Session()
        self.tls_verify = tls_verify

        if not tls_verify:
            requests.packages.urllib3.disable_warnings()  # type: ignore

        # Ignore all cookies: some registries try to set one
        # and take it as a sign they are talking to a browser,
        # trying to set further CSRF cookies (Harbor is such a case)
        self.session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))

    def request(
        self,
        url: str,
        method: str = "GET",
        data: Optional[Union[dict, bytes, IO]] = None,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        stream: bool = False,
        params: Optional[dict] = None,
    ) -> requests.Response:
        """
        Send a single request, without any authentication or retry handling.

        :param url: the URL to issue the request to
        :type url: str
        :param method: the method to use (GET, DELETE, POST, PUT, PATCH)
        :type method: str
        :param data: body for the request. A file object or iterator is sent
                     without being read into memory first
        :type data: dict or bytes or IO
        :param headers: headers for the request
        :type headers: dict
        :param json: json data for requests
        :type json: dict
        :param stream: stream the response instead of downloading it at once
        :type stream: bool
        :param params: query string parameters to add to the url
        :type params: dict
        """
        return self.session.request(
            method,
            url,
            data=data,
            json=json,
            headers=headers,
            stream=stream,
            params=params,
            verify=self.tls_verify,
        )

    def close(self):
        """
        Release the underlying connections.

        Calling this is optional. It is offered for callers that create many
        short lived clients and do not want to wait for garbage collection to
        close the pooled connections.
        """
        self.session.close()


def successful_response(status_code: int = 200) -> requests.Response:
    """
    Create a response for an interaction that did not need to hit the registry.

    This is used when an upload can be skipped (e.g., the blob is already known
    to exist) but the caller still expects a response to inspect.

    :param status_code: the status code to report
    :type status_code: int
    """
    response = requests.Response()
    response.status_code = status_code
    return response
