import base64
import requests
import streamlit as st

def github_put_file(path: str, content_text: str, message: str):
    token = st.secrets.get("GH_TOKEN", "")
    owner = st.secrets.get("GH_OWNER", "")
    repo = st.secrets.get("GH_REPO", "")
    branch = st.secrets.get("GH_BRANCH", "main")

    if not token or not owner or not repo:
        return {"ok": False, "error": "Missing GH_* secrets"}

    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    r0 = requests.get(api, headers=headers, params={"ref": branch}, timeout=30)
    sha = None
    if r0.status_code == 200:
        sha = r0.json().get("sha")

    b64 = base64.b64encode(content_text.encode("utf-8")).decode("utf-8")
    payload = {"message": message, "content": b64, "branch": branch}
    if sha:
        payload["sha"] = sha

    r = requests.put(api, headers=headers, json=payload, timeout=30)
    if r.status_code in (200, 201):
        return {"ok": True}
    return {"ok": False, "error": f"{r.status_code}: {r.text[:300]}"}
