#! /Users/greg/miniforge3/bin/python

from paramiko import SSHClient, AuthenticationException, AutoAddPolicy
import sys, os
import re
import posixpath
import tempfile
import secrets
import pathlib
import requests
from urllib.parse import unquote
import json
import base64
import numpy as np
import pandas as pd

_SIGNATURE_TAG = "<!-- plotlyServer-signature: "


def _decode(values):
    # plotly >= 6 stores numeric arrays as {"dtype": "f8", "bdata": <base64>, "shape": ...}
    if isinstance(values, dict) and "bdata" in values:
        arr = np.frombuffer(base64.b64decode(values["bdata"]), dtype="<" + values["dtype"])
        if "shape" in values:
            arr = arr.reshape([int(n) for n in str(values["shape"]).split(",")])
        return arr
    return np.asarray(values)


def _parse_html(html):
    # write_html embeds the figure as Plotly.newPlot("<div id>", [traces], layout, config).
    # Take the last call: older pages also contain the string inside plotly.js itself.
    start = html.rindex("Plotly.newPlot(")
    start = html.index(",", start) + 1
    decoder = json.JSONDecoder()
    rest = html[start:].lstrip()
    traces, n = decoder.raw_decode(rest)
    rest = rest[n:].lstrip()[1:].lstrip()
    layout, _ = decoder.raw_decode(rest)
    return traces, layout


def _code(link):
    # the random code of a page: the part of its link before "&figname="
    return unquote(link.rsplit("/", 1)[-1]).split("&figname=")[0]


def _signature(traces, layout):
    # What makes two uploads "the same figure": the kinds of traces and the axis
    # titles. Data, trace names and colors can change between versions.
    def title(ax):
        t = ax.get("title")
        if isinstance(t, dict):
            t = t.get("text")
        return (t or "").strip()

    axes = {
        k: title(v)
        for k, v in layout.items()
        if re.fullmatch(r"[xy]axis\d*", k) and isinstance(v, dict)
    }
    return {"types": sorted({t.get("type", "scatter") for t in traces}), "axes": dict(sorted(axes.items()))}


class plotlyServer:
    def __init__(self, **kwargs):
        self._fig = kwargs.get("fig", None)
        self._figname = kwargs.get("figname", None)
        # Plots live on the NAS (Amun), shared folder "plots"; Cloud Sync copies
        # every new file to Cloudflare, which serves the public link.
        # Paths are SFTP paths: on the NAS, shared folders sit at the top ("/plots").
        self._server = kwargs.get("server", "amun.lab")
        self._path = kwargs.get("path", "/plots/")
        # replaced versions are kept here (one per page), outside the public folder
        self._backup = kwargs.get("backup", "/home/plots_backup/")
        self._username = kwargs.get("username", "nphoton")
        # No password in the code: log in with your SSH key (see the lab wiki),
        # or set the PLOTLY_SERVER_PASSWORD environment variable.
        self._password = kwargs.get("password", os.getenv("PLOTLY_SERVER_PASSWORD"))
        self._base_url = kwargs.get("base_url", "https://nanophotonicsjqi.io/plots/")
        # With fig: link (or just its code) of a page to replace, keeping that link.
        # Without fig: link of a page to download the data from.
        self._link = kwargs.get("url", None)

        if not self._fig is None:
            self.createTempHTML()
            self.transferHTML()
            self.figurl = self._base_url + self._filename
            print("URL to plot: ")
            print(self.figurl)
        else:
            try:
                self.plot = self.retrieveHTMLdata(self._link)
            except Exception as e:
                print("Error retrieving HTML data: ", e)


    def retrieveHTMLdata(self, url):
        resp = requests.get(url, verify=False)
        resp.raise_for_status()
        traces, _ = _parse_html(resp.text)
        output = []
        for tr in traces:
            # x/y curves only: heatmaps etc. have x, y of different lengths
            if tr.get("type", "scatter") in ("scatter", "scattergl") and "x" in tr and "y" in tr:
                df = pd.DataFrame({"x": _decode(tr["x"]), "y": _decode(tr["y"])})
                df.attrs["name"] = tr.get("name")
                output.append(df)
        return output

    def createFileName(self):
        secret = secrets.token_urlsafe(50)
        self._filename = secret + "&figname=" + self._figname + '.html'

    def createTempHTML(self):
        tmpdir = pathlib.Path(tempfile.gettempdir())
        self._tempFile = str(tmpdir.joinpath(self._figname + ".html"))
        self._fig.write_html(self._tempFile)
        fig = self._fig.to_dict()
        self._signature = _signature(fig["data"], fig["layout"])
        # store the signature at the top of the page, so later uploads can read
        # it without downloading the whole file
        html = pathlib.Path(self._tempFile).read_text(encoding="utf-8")
        pathlib.Path(self._tempFile).write_text(
            _SIGNATURE_TAG + json.dumps(self._signature) + " -->\n" + html, encoding="utf-8"
        )
        os.chmod(self._tempFile, 0o666)

    def _connect(self):
        ssh = SSHClient()
        ssh.load_system_host_keys()
        # first connection to the NAS: accept its host key instead of failing
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        try:
            ssh.connect(self._server, username=self._username, password=self._password)
        except AuthenticationException:
            raise AuthenticationException(
                f"Could not log in to {self._username}@{self._server}: your SSH key is "
                "not authorized there and PLOTLY_SERVER_PASSWORD is not set. See the "
                "plotlyServer page of the lab wiki."
            ) from None
        return ssh

    def _remoteSignature(self, sftp, filename):
        try:
            with sftp.open(self._path + filename) as fh:
                head = fh.read(4096).decode("utf-8", "replace")
                if head.startswith(_SIGNATURE_TAG):
                    return json.loads(head[len(_SIGNATURE_TAG):head.index(" -->")])
                # page uploaded before signatures existed: read the figure itself
                fh.seek(0)
                fh.prefetch()
                html = fh.read().decode("utf-8", "replace")
            return _signature(*_parse_html(html))
        except Exception:
            return None

    def _replace(self, ssh, sftp, filename):
        self._filename = filename
        # keep the version being replaced, so nothing is ever lost
        # (copied over SFTP: the NAS gives lab accounts SFTP, not a shell)
        try:
            sftp.stat(self._backup)
        except IOError:
            sftp.mkdir(self._backup)
        try:
            with sftp.open(self._path + filename, "rb") as src, \
                 sftp.open(posixpath.join(self._backup, filename), "wb") as dst:
                src.prefetch()
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
        except IOError as e:
            raise IOError(f"Could not back up the previous version: {e}") from None
        sftp.put(self._tempFile, self._path + filename)

    def transferHTML(self):
        ssh = self._connect()
        sftp = ssh.open_sftp()
        if self._link:
            # replace exactly the page given by url=, whatever its figure or figname
            code = _code(self._link)
            target = [f for f in sftp.listdir(self._path) if f.startswith(code + "&figname=")]
            if not target:
                ssh.close()
                os.remove(self._tempFile)
                raise FileNotFoundError(f"No page with the code {code} on the server")
            self._replace(ssh, sftp, target[0])
            print("Updated the page given by url= (same link); previous version backed up.")
            sftp.close()
            ssh.close()
            os.remove(self._tempFile)
            return
        # Same figname AND same figure (trace types + axis titles) -> new version:
        # replace that page, so the link (and Notion embeds) stay the same and
        # copies don't pile up. Same figname but a different figure -> new page.
        suffix = "&figname=" + self._figname + ".html"
        same = sorted(
            (f for f in sftp.listdir_attr(self._path) if f.filename.endswith(suffix)),
            key=lambda f: f.st_mtime,
            reverse=True,
        )
        match = next((f for f in same if self._remoteSignature(sftp, f.filename) == self._signature), None)
        if match:
            self._replace(ssh, sftp, match.filename)
            print(f"Updated the existing '{self._figname}' page (same figure); previous version backed up.")
        else:
            self.createFileName()
            sftp.put(self._tempFile, self._path + self._filename)
            # group-writable, so other lab accounts can update it later
            # (on the NAS the share's permissions already allow it)
            try:
                sftp.chmod(self._path + self._filename, 0o664)
            except IOError:
                pass
            if same:
                print(
                    f"Created a new page: the {len(same)} page(s) already named "
                    f"'{self._figname}' are different figures (other axes or trace types)."
                )
        sftp.close()
        ssh.close()
        os.remove(self._tempFile)

    def removeHTML(self):
        ssh = self._connect()
        sftp = ssh.open_sftp()
        sftp.remove(self._path + self._filename)
        sftp.close()
        ssh.close()
