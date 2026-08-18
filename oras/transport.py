"""
HTTP transport for registry interactions.

This module is the only place where requests are actually sent to a registry.
Keeping execution behind a small boundary means that the rest of the SDK deals
with urls, headers and responses, and never with session or TLS handling.
"""

__copyright__ = "Copyright The ORAS Authors."
__license__ = "Apache-2.0"

from http.cookiejar import DefaultCookiePolicy
from typing import Optional, Union

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
        data: Optional[Union[dict, bytes]] = None,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
        stream: bool = False,
    ) -> requests.Response:
        """
        Send a single request, without any authentication or retry handling.

        :param url: the URL to issue the request to
        :type url: str
        :param method: the method to use (GET, DELETE, POST, PUT, PATCH)
        :type method: str
        :param data: data for requests
        :type data: dict or bytes
        :param headers: headers for the request
        :type headers: dict
        :param json: json data for requests
        :type json: dict
        :param stream: stream the responses
        :type stream: bool
        """
        return self.session.request(
            method,
            url,
            data=data,
            json=json,
            headers=headers,
            stream=stream,
            verify=self.tls_verify,
        )


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
