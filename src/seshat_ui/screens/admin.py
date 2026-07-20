from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from seshat_ui.api_client import ApiClient

_ROLES = ["viewer", "reviewer", "operator", "admin"]


def render(root_client: ApiClient) -> None:
    st.title("Admin — API Keys")
    list_tab, create_tab = st.tabs(["List keys", "Create key"])

    with list_tab:
        _render_list(root_client)

    with create_tab:
        _render_create(root_client)


def _render_list(root_client: ApiClient) -> None:
    if st.button("Refresh", key="adm_refresh"):
        st.session_state.pop("adm_keys", None)

    if "adm_keys" not in st.session_state:
        try:
            st.session_state["adm_keys"] = root_client.list_api_keys()
        except Exception as exc:
            st.error(f"Failed to list keys: {exc}")
            return

    keys = st.session_state.get("adm_keys", [])
    if not keys:
        st.info("No API keys found.")
        return

    for key in keys:
        active = key.get("is_active", True)
        status = "🟢 active" if active else "🔴 revoked"
        label = f"{status} — `{key['user_id']}` ({key['role']}) — id {key['id']}"

        with st.expander(label):
            st.write(f"Created: {key['created_at']}")
            if key.get("revoked_at"):
                st.write(f"Revoked: {key['revoked_at']}")

            if active and st.button("Revoke", key=f"adm_revoke_{key['id']}"):
                resp = root_client.revoke_api_key(key["id"])
                if resp.is_success:
                    st.toast("Key revoked.", icon="✅")
                    st.session_state.pop("adm_keys", None)
                    st.rerun()
                else:
                    st.toast(f"Failed: {resp.text}", icon="❌")


def _render_create(root_client: ApiClient) -> None:
    st.header("Create API key")
    with st.form("create_key_form"):
        user_id = st.text_input("User ID", key="adm_c_user")
        role = st.selectbox("Role", _ROLES, key="adm_c_role")
        submitted = st.form_submit_button("Create")

    if submitted:
        if not user_id:
            st.toast("User ID is required.", icon="⚠️")
            return

        resp = root_client.create_api_key(user_id, role)
        if resp.is_success:
            data = resp.json()
            st.toast(f"Key created for {data['user_id']} ({data['role']})", icon="✅")
            st.warning("Store this key safely — it will not be shown again.")
            st.code(data["api_key"], language=None)
            st.session_state.pop("adm_keys", None)
        else:
            st.toast(f"Failed ({resp.status_code}): {resp.text}", icon="❌")
