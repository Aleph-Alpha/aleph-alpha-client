import base64
import requests


def load_base64_from_url(url: str):
    """
    download a file and return the base64 encoded content
    """
    im_b64 = base64.b64encode(requests.get(url).content).decode()
    return im_b64


def load_base64_from_file(path_and_filename: str):
    """
    load a file from disk and return the base64 encoded content
    """
    with open(path_and_filename, "rb") as f:
        im_b64 = base64.b64encode(f.read()).decode()
    return im_b64
