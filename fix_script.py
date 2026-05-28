import sys
sys.stdout.reconfigure(encoding="utf-8")
content = open("app/router/proxy.py", encoding="utf-8").read()
old = "    except Exception:\n        pass  # Never let DB errors block the response"
new = "    except Exception as e:\n        _app_log.warning(\"Failed to log request: %s\", e)"
content = content.replace(old, new)
open("app/router/proxy.py", "w", encoding="utf-8").write(content)
print("done")
