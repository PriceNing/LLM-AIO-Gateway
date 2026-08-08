"""ComfyUI API-format workflow inspection and image generation adapter."""

import asyncio
import base64
import copy
import random
import re
import time
import uuid
from typing import Any
from urllib.parse import quote, urlparse

import httpx


_MAPPING_FIELDS = (
    "prompt", "negative_prompt", "width", "height", "seed",
    "steps", "cfg", "batch_size",
)


def workflow_userdata_url(api_base: str, workflow_name: str) -> str:
    """Return the ComfyUI userdata URL for a saved UI workflow."""
    name = str(workflow_name or "").strip().replace("\\", "/").strip("/")
    if not name:
        raise ValueError("ComfyUI workflow name is required")
    if not name.lower().endswith(".json"):
        name += ".json"
    return f"{_base_url(api_base)}/api/userdata/{quote('workflows/' + name, safe='')}"


async def list_saved_workflows(api_base: str, *, api_key: str = "", timeout: int = 30) -> list[str]:
    """List saved ComfyUI workflow paths from its userdata API."""
    base = _base_url(api_base)
    headers = _headers({"api_key": api_key})
    async with httpx.AsyncClient(timeout=max(1, min(60, int(timeout))), headers=headers) as client:
        response = await client.get(f"{base}/api/userdata", params={"dir": "workflows", "recurse": "true"})
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, list):
        raise ValueError("ComfyUI workflow list returned an unexpected response")
    return sorted({str(item) for item in body if isinstance(item, str) and item.lower().endswith(".json")})


async def load_saved_workflow(api_base: str, workflow_name: str, *, api_key: str = "", timeout: int = 30) -> dict[str, Any]:
    """Load a saved ComfyUI workflow JSON from userdata."""
    headers = _headers({"api_key": api_key})
    async with httpx.AsyncClient(timeout=max(1, min(60, int(timeout))), headers=headers) as client:
        response = await client.get(workflow_userdata_url(api_base, workflow_name))
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, dict):
        raise ValueError("ComfyUI workflow returned an unexpected response")
    return body


def _widget_value_matches(value: Any, input_type: Any) -> bool:
    """Best-effort match between a UI widget value and its API input type."""
    type_name = str(input_type or "").upper()
    if type_name in {"INT", "FLOAT", "NUMBER"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name in {"BOOLEAN", "BOOL"}:
        return isinstance(value, bool)
    if type_name in {"STRING", "COMBO"}:
        return isinstance(value, str)
    return not isinstance(value, (dict, list))


def convert_ui_workflow(value: Any) -> dict[str, dict[str, Any]]:
    """Convert a regular ComfyUI graph JSON into API prompt format.

    Regular saved workflows store graph links and widget values separately.
    The API accepts a node-id mapping with ``class_type`` and ``inputs``.
    This converter reconstructs that representation without requiring users to
    export the same graph a second time.
    """
    if not isinstance(value, dict) or not isinstance(value.get("nodes"), list):
        raise ValueError("ComfyUI UI workflow must contain a nodes array")
    raw_links = value.get("links") or []
    links: dict[str, tuple[str, int]] = {}
    for raw_link in raw_links:
        if not isinstance(raw_link, list) or len(raw_link) < 6:
            continue
        links[str(raw_link[0])] = (str(raw_link[1]), int(raw_link[2]))

    converted: dict[str, dict[str, Any]] = {}
    for raw_node in value["nodes"]:
        if not isinstance(raw_node, dict):
            continue
        node_id = str(raw_node.get("id") if raw_node.get("id") is not None else "")
        class_type = str(raw_node.get("type") or "").strip()
        if not node_id or not class_type:
            raise ValueError("ComfyUI UI workflow contains a node without id or type")
        # mode=2 is LiteGraph's muted mode. Excluding it mirrors ComfyUI's
        # prompt serialization for ordinary, non-bypassed graphs.
        if int(raw_node.get("mode") or 0) == 2:
            continue
        widget_values = list(raw_node.get("widgets_values") or [])
        widget_cursor = 0
        inputs: dict[str, Any] = {}
        for descriptor in raw_node.get("inputs") or []:
            if not isinstance(descriptor, dict):
                continue
            input_name = str(descriptor.get("name") or "")
            if not input_name:
                continue
            link_id = descriptor.get("link")
            if link_id is not None and str(link_id) in links:
                source_id, source_slot = links[str(link_id)]
                inputs[input_name] = [source_id, source_slot]
                continue
            if not isinstance(descriptor.get("widget"), dict):
                continue
            # Some widgets add a hidden control value (for example KSampler's
            # seed is followed by "randomize"). Scan forward for the next
            # value compatible with the declared input type.
            selected = None
            while widget_cursor < len(widget_values):
                candidate = widget_values[widget_cursor]
                widget_cursor += 1
                if _widget_value_matches(candidate, descriptor.get("type")):
                    selected = candidate
                    break
            if selected is not None:
                inputs[input_name] = selected
        title = str(raw_node.get("title") or (raw_node.get("properties") or {}).get("Node name for S&R") or class_type)
        converted[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
            "_meta": {"title": title},
        }
    if not converted:
        raise ValueError("ComfyUI UI workflow contains no executable nodes")
    return converted


def normalize_workflow(value: Any) -> dict[str, dict[str, Any]]:
    """Return a validated API workflow, converting regular UI JSON."""
    workflow = value
    if isinstance(workflow, dict) and isinstance(workflow.get("prompt"), dict):
        workflow = workflow["prompt"]
    elif isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list):
        workflow = convert_ui_workflow(workflow)
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("ComfyUI workflow must be a non-empty API-format JSON object")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_id, raw_node in workflow.items():
        node_id = str(raw_id)
        if not isinstance(raw_node, dict) or not str(raw_node.get("class_type") or "").strip():
            raise ValueError(f"ComfyUI workflow node {node_id!r} is missing class_type")
        inputs = raw_node.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError(f"ComfyUI workflow node {node_id!r} is missing inputs")
        normalized[node_id] = copy.deepcopy(raw_node)
    return normalized


def _node_label(node_id: str, node: dict[str, Any]) -> str:
    title = str((node.get("_meta") or {}).get("title") or "").strip()
    class_type = str(node.get("class_type") or "")
    return f"{node_id} · {title or class_type} ({class_type})"


def _primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def analyze_workflow(value: Any) -> dict[str, Any]:
    """List selectable node inputs and suggest common generation mappings."""
    source_format = "ui" if isinstance(value, dict) and isinstance(value.get("nodes"), list) else "api"
    workflow = normalize_workflow(value)
    candidates: dict[str, list[dict[str, str]]] = {field: [] for field in _MAPPING_FIELDS}
    outputs: list[dict[str, str]] = []
    for node_id, node in workflow.items():
        class_type = str(node.get("class_type") or "")
        class_lower = class_type.lower()
        label = _node_label(node_id, node)
        if any(token in class_lower for token in ("saveimage", "previewimage", "output", "save_image")):
            outputs.append({"node_id": node_id, "label": label})
        for input_name, current in node["inputs"].items():
            if not _primitive(current):
                continue
            name = str(input_name)
            lower = name.lower()
            item = {"node_id": node_id, "input": name, "label": f"{label} → {name}"}
            if isinstance(current, str):
                candidates["prompt"].append(item)
                candidates["negative_prompt"].append(item)
            if lower in {"width", "height", "seed", "noise_seed", "steps", "cfg", "cfg_scale", "batch_size", "batch"}:
                target = {"noise_seed": "seed", "cfg_scale": "cfg", "batch": "batch_size"}.get(lower, lower)
                if target in candidates:
                    candidates[target].append(item)

    def score(field: str, item: dict[str, str]) -> tuple[int, str]:
        node = workflow[item["node_id"]]
        class_name = str(node.get("class_type") or "").lower()
        title = str((node.get("_meta") or {}).get("title") or "").lower()
        name = item["input"].lower()
        combined = f"{title} {name}"
        points = 0
        if field == "prompt":
            points += 6 if name in {"text", "prompt", "positive"} else 0
            points += 5 if any(x in combined for x in ("positive", "prompt", "正向")) else 0
            points += 2 if "textencode" in class_name else 0
            points -= 8 if any(x in combined for x in ("negative", "负向")) else 0
        elif field == "negative_prompt":
            points += 8 if any(x in combined for x in ("negative", "负向")) else 0
            points += 2 if "textencode" in class_name else 0
        else:
            points += 8 if name == field else 0
            points += 6 if field == "seed" and name == "noise_seed" else 0
            points += 6 if field == "cfg" and name == "cfg_scale" else 0
        return (-points, item["label"])

    suggestions: dict[str, Any] = {}
    for field, items in candidates.items():
        items.sort(key=lambda item: score(field, item))
        if items and -score(field, items[0])[0] > 0:
            suggestions[field] = {"node_id": items[0]["node_id"], "input": items[0]["input"]}
    if outputs:
        suggestions["output_node_id"] = outputs[0]["node_id"]
    return {
        "workflow": workflow,
        "source_format": source_format,
        "converted": source_format == "ui",
        "node_count": len(workflow),
        "candidates": candidates,
        "outputs": outputs,
        "suggestions": suggestions,
    }


def validate_mapping(workflow_value: Any, mapping: Any) -> dict[str, Any]:
    workflow = normalize_workflow(workflow_value)
    if not isinstance(mapping, dict):
        raise ValueError("ComfyUI workflow mapping must be an object")
    normalized: dict[str, Any] = {}
    for field in _MAPPING_FIELDS:
        selected = mapping.get(field)
        if selected in (None, ""):
            continue
        if not isinstance(selected, dict):
            raise ValueError(f"ComfyUI mapping {field} must select a node and input")
        node_id = str(selected.get("node_id") or "")
        input_name = str(selected.get("input") or "")
        if node_id not in workflow or input_name not in workflow[node_id]["inputs"]:
            raise ValueError(f"ComfyUI mapping {field} references an unknown node input")
        normalized[field] = {"node_id": node_id, "input": input_name}
    if "prompt" not in normalized:
        raise ValueError("ComfyUI positive prompt mapping is required")
    output_node_id = str(mapping.get("output_node_id") or "")
    if output_node_id and output_node_id not in workflow:
        raise ValueError("ComfyUI output mapping references an unknown node")
    normalized["output_node_id"] = output_node_id
    return normalized


def _base_url(value: str) -> str:
    base = str(value or "").strip().rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("ComfyUI Base URL must be an http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("ComfyUI Base URL must not contain credentials")
    return base


def _headers(config: dict[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    if config.get("api_key"):
        headers["Authorization"] = f"Bearer {config['api_key']}"
    if isinstance(config.get("extra_headers"), dict):
        headers.update({str(k): str(v) for k, v in config["extra_headers"].items()})
    return headers


def _set_mapping(workflow: dict[str, Any], mapping: dict[str, Any], field: str, value: Any) -> None:
    selected = mapping.get(field)
    if selected and value is not None:
        workflow[selected["node_id"]]["inputs"][selected["input"]] = value


def _size_pair(size: str | None) -> tuple[int, int] | None:
    if not size or str(size).lower() == "auto":
        return None
    match = re.fullmatch(r"(\d+)x(\d+)", str(size).strip().lower())
    if not match:
        raise ValueError(f"unsupported image size: {size}")
    return int(match.group(1)), int(match.group(2))


async def generate_comfyui_images(
    config: dict[str, Any], *, prompt: str, n: int = 1, size: str | None = None,
    extra: dict[str, Any] | None = None,
):
    """Submit a configured API workflow, wait for it, and download its images."""
    from app.adapters.imagegen import ImageBackendHTTPError, ImageGenerationResult, _mime_from_bytes

    workflow_template = normalize_workflow(config.get("workflow"))
    mapping = validate_mapping(workflow_template, config.get("workflow_mapping") or {})
    requested = max(1, min(int(n or 1), 10))
    # Workflows without a latent/batch input can only produce their fixed
    # output count. Submit one job per requested image instead of silently
    # returning fewer images than the OpenAI Images contract requested.
    if requested > 1 and not mapping.get("batch_size"):
        results = []
        for _ in range(requested):
            results.extend(await generate_comfyui_images(
                config, prompt=prompt, n=1, size=size, extra=extra,
            ))
        return results[:requested]
    workflow = copy.deepcopy(workflow_template)
    options = extra or {}
    _set_mapping(workflow, mapping, "prompt", prompt)
    if "negative_prompt" in options:
        _set_mapping(workflow, mapping, "negative_prompt", str(options["negative_prompt"]))
    pair = _size_pair(size)
    if pair:
        _set_mapping(workflow, mapping, "width", pair[0])
        _set_mapping(workflow, mapping, "height", pair[1])
    for field in ("seed", "steps", "cfg"):
        if options.get(field) is not None:
            _set_mapping(workflow, mapping, field, options[field])
    if mapping.get("seed") and options.get("seed") is None:
        _set_mapping(workflow, mapping, "seed", random.randint(0, 2**63 - 1))
    _set_mapping(workflow, mapping, "batch_size", requested)

    base = _base_url(config.get("api_base") or "")
    timeout = max(1, min(3600, int(config.get("timeout") or 300)))
    poll_interval = max(0.2, min(10.0, float(config.get("poll_interval") or 1.0)))
    max_bytes = max(64 * 1024, min(100 * 1024 * 1024, int(config.get("result_max_bytes") or 25 * 1024 * 1024)))
    client_id = uuid.uuid4().hex
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=httpx.Timeout(min(timeout, 60), connect=min(timeout, 15)), headers=_headers(config)) as client:
        response = await client.post(f"{base}/prompt", json={"prompt": workflow, "client_id": client_id})
        if response.status_code >= 400:
            raise ImageBackendHTTPError(response.status_code, response.text[:1000].strip())
        try:
            submission = response.json()
            prompt_id = str(submission.get("prompt_id") or "")
        except ValueError as exc:
            raise ValueError("ComfyUI returned invalid JSON from /prompt") from exc
        if not prompt_id:
            details = submission.get("node_errors") or submission.get("error") or submission
            raise ValueError(f"ComfyUI rejected the workflow: {str(details)[:1000]}")

        history_item: dict[str, Any] | None = None
        while time.monotonic() - started < timeout:
            history_response = await client.get(f"{base}/history/{prompt_id}")
            if history_response.status_code >= 400:
                raise ImageBackendHTTPError(history_response.status_code, history_response.text[:1000].strip())
            body = history_response.json()
            candidate = body.get(prompt_id) if isinstance(body, dict) else None
            if isinstance(candidate, dict):
                status = candidate.get("status") or {}
                if status.get("status_str") == "error":
                    messages = status.get("messages") or []
                    raise RuntimeError(f"ComfyUI workflow failed: {str(messages)[-1000:]}")
                if isinstance(candidate.get("outputs"), dict):
                    history_item = candidate
                    break
            await asyncio.sleep(poll_interval)
        if history_item is None:
            raise TimeoutError(f"ComfyUI workflow did not complete within {timeout} seconds")

        outputs = history_item.get("outputs") or {}
        selected_output = mapping.get("output_node_id")
        node_outputs = [(selected_output, outputs.get(selected_output))] if selected_output else list(outputs.items())
        descriptors: list[dict[str, Any]] = []
        for _, output in node_outputs:
            if not isinstance(output, dict):
                continue
            descriptors.extend(item for item in (output.get("images") or []) if isinstance(item, dict))
        if not descriptors:
            raise RuntimeError("ComfyUI workflow completed without image output")

        results = []
        for descriptor in descriptors[:requested]:
            filename = str(descriptor.get("filename") or "")
            if not filename:
                continue
            image_response = await client.get(f"{base}/view", params={
                "filename": filename,
                "subfolder": str(descriptor.get("subfolder") or ""),
                "type": str(descriptor.get("type") or "output"),
            })
            if image_response.status_code >= 400:
                raise ImageBackendHTTPError(image_response.status_code, image_response.text[:1000].strip())
            raw = image_response.content
            if len(raw) > max_bytes:
                raise ValueError("ComfyUI image result exceeds the configured size limit")
            mime_type = _mime_from_bytes(raw, (image_response.headers.get("content-type") or "image/png").split(";", 1)[0])
            results.append(ImageGenerationResult(
                data_uri=f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}",
                mime_type=mime_type, size=size, backend_attempts=1,
            ))
        if not results:
            raise RuntimeError("ComfyUI returned no downloadable image output")
        return results
