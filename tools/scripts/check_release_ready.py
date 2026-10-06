#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发版前门禁：把发布规则 §8 里可以被机器检查的项一次性跑完。

存在理由（v0.15.0 事故）：发布流水线从 app/__init__.py 读取版本号，
读到什么就命名什么。跳过 bump 直接打 tag 时 CI 静默通过，
产出的 assets 与镜像里自报版本全是上一个版本号。

检查项：
1. 当前分支必须是 main。
2. 版本号三处一致：命令行目标版本 == app/__init__.py == CHANGELOG 顶部条目。
   （bump 是 tag 的前置条件，"文档计数已经同步"不能作为 bump 已执行的证据。）
3. Git 工作区干净且无未推送提交（用本地 ref 判断，不联网 fetch）。
4. SVN 工作副本无待提交项（未版本化/已忽略项不计）。
5. 忽略边界：命中 .gitignore / .git/info/exclude 的文件不得被 Git 跟踪；
   命中 svn:ignore / svn:global-ignores 的文件不得被 SVN 跟踪。
   （防的是"临时排查文档误提交进版本库"与"凭据文件误入库"两类事故。）
6. 文档测试计数与实际收集数一致（委托 check_error_mapping.py，可 --skip-counts 降级）。
7. tag 与源码版本一致性（bump_version.py --check）：打 tag 之前只作提示，
   加 --post-tag 后升级为硬校验。

退出码 0 = 全部通过；1 = 存在失败项。

用法：
    python tools/scripts/check_release_ready.py 0.15.0
    python tools/scripts/check_release_ready.py 0.15.0 --post-tag
    python tools/scripts/check_release_ready.py 0.15.0 --skip-counts
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BS = chr(92)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
VERSION_FILE = REPO_ROOT / "app" / "__init__.py"
CHANGELOG_FILE = REPO_ROOT / "CHANGELOG.md"
VERSION_LINE_RE = re.compile(
    r"__version__\s*=\s*[" + chr(34) + chr(39) + "]([^" + chr(34) + chr(39) + "]+)"
)
CHANGELOG_VER_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.M)


def norm(p):
    return str(p).replace(BS, "/").rstrip("/")


def run(cmd, cwd=REPO_ROOT, input_text=None):
    """Run a command, return (ok, stdout_lines, stderr_text)."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", input=input_text,
        )
    except FileNotFoundError:
        return False, [], "command not found: " + cmd[0]
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return proc.returncode == 0, lines, proc.stderr.strip()


def git_lines(args):
    return run(["git"] + args)


def svn_lines(args):
    return run(["svn"] + args)


def read_source_version():
    m = VERSION_LINE_RE.search(VERSION_FILE.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def changelog_top_version():
    m = CHANGELOG_VER_RE.search(CHANGELOG_FILE.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def tracked_paths(cmd_stdout):
    return {norm(p) for p in cmd_stdout if p}


class Gate:
    def __init__(self):
        self.fails = 0
        self.skips = 0

    def ok(self, name, detail=""):
        print("  PASS  " + name + ("  (" + detail + ")" if detail else ""))

    def fail(self, name, detail=""):
        self.fails += 1
        print("  FAIL  " + name + ("  (" + detail + ")" if detail else ""))

    def info(self, name, detail=""):
        print("  INFO  " + name + ("  (" + detail + ")" if detail else ""))

    def skip(self, name, detail=""):
        self.skips += 1
        print("  SKIP  " + name + ("  (" + detail + ")" if detail else ""))


def check_branch(g):
    ok, out, _ = git_lines(["rev-parse", "--abbrev-ref", "HEAD"])
    branch = out[0].strip() if ok and out else "?"
    if branch == "main":
        g.ok("branch is main")
    else:
        g.fail("branch is main", "current=" + branch)


def check_version_agreement(g, target):
    src = read_source_version()
    top = changelog_top_version()
    if src is None:
        g.fail("version agreement", "cannot parse app/__init__.py")
        return
    if not target:
        g.info("version agreement", "no target given, source=" + src)
        return
    if not SEMVER_RE.match(target):
        g.fail("version agreement", repr(target) + " is not valid semver")
        return
    if src != target:
        g.fail(
            "version agreement",
            "target=" + target + " but app/__init__.py=" + src
            + " -> run bump_version.py " + target + " BEFORE tagging",
        )
    elif top != target:
        g.fail("version agreement", "CHANGELOG top entry is [" + str(top) + "], expected [" + target + "]")
    else:
        g.ok("version agreement", target + " == source == CHANGELOG")


def check_git_clean(g):
    ok, out, _ = git_lines(["status", "--porcelain"])
    if not ok:
        g.fail("git clean", "git status failed")
        return
    dirty = [ln for ln in out if ln.strip()]
    if dirty:
        g.fail("git clean", str(len(dirty)) + " pending change(s)")
    else:
        g.ok("git clean")

    ok, out, err = git_lines(["rev-parse", "--abbrev-ref", "origin/main"])
    if not ok:
        g.info("git pushed", "no origin/main ref locally, cannot verify (skipped, not fetched)")
        return
    ok, out, _ = git_lines(["rev-list", "--count", "origin/main..HEAD"])
    ahead = out[0].strip() if ok and out else "?"
    if ahead == "0":
        g.ok("git pushed", "no unpushed commit")
    else:
        g.fail("git pushed", ahead + " commit(s) not pushed to origin/main")


def check_svn_clean(g):
    ok, out, err = svn_lines(["status"])
    if not ok:
        if "not a working copy" in (err or ""):
            g.skip("svn clean", "working copy is not an svn checkout")
        else:
            g.fail("svn clean", "svn status failed: " + (err or "unknown"))
        return
    pending = []
    for ln in out:
        flag = ln[:1].strip()
        # '?'/unversioned and 'I'/ignored are not pending commits; '~' is svn 1.x noise.
        if flag and flag not in ("?", "I", "~"):
            pending.append(ln)
    if pending:
        g.fail("svn clean", str(len(pending)) + " item(s) pending commit")
    else:
        g.ok("svn clean")


def check_ignore_boundary(g):
    # Git side: anything matching ignore rules must not be tracked.
    ok, tracked, _ = git_lines(["ls-files"])
    git_side = []
    if not ok:
        g.fail("ignore boundary (git)", "git ls-files failed")
    else:
        tracked_set = tracked_paths(tracked)
        if tracked_set:
            feed = chr(10).join(sorted(tracked_set)) + chr(10)
            ok2, matched, _ = run(["git", "check-ignore", "--stdin"], input_text=feed)
            # exit 0 = at least one path matched, 1 = none matched
            git_side = sorted({norm(p) for p in matched if norm(p)} & tracked_set)
    if git_side:
        g.fail("ignore boundary (git)", str(len(git_side)) + " tracked file(s) match ignore rules: " + ", ".join(git_side[:6]))
    elif ok:
        g.ok("ignore boundary (git)")

    # SVN side: anything matching svn:ignore must not be svn-tracked.
    ok, listed, _ = svn_lines(["ls", "-R"])
    if not ok:
        g.skip("ignore boundary (svn)", "svn ls failed / not a working copy")
        return
    svn_tracked = {norm(p) + "/" for p in listed}
    ignored = set()
    for prop in ("svn:ignore", "svn:global-ignores"):
        ok2, vals, _ = svn_lines(["propget", prop, "-R", "."])
        if not ok2:
            continue
        base = ""
        for ln in vals:
            s = ln.strip()
            if s.endswith(":"):
                base = norm(s[:-1])
                if base == ".":
                    base = ""
                continue
            if not s:
                continue
            pat = norm(s).lstrip("/")
            if base:
                pat = base + "/" + pat
            ignored.add(pat)
    offenders = sorted(p for p in svn_tracked if p.rstrip("/") in ignored)
    if offenders:
        g.fail("ignore boundary (svn)", str(len(offenders)) + " svn-tracked file(s) match svn:ignore: " + ", ".join(offenders[:6]))
    else:
        g.ok("ignore boundary (svn)", str(len(svn_tracked)) + " paths checked")


def check_test_counts(g, skip_counts):
    script = REPO_ROOT / "tools" / "scripts" / "check_error_mapping.py"
    if skip_counts:
        g.skip("test counts", "explicitly skipped")
        return
    if not script.exists():
        g.skip("test counts", "checker script missing")
        return
    ok, out, err = run([sys.executable, str(script)])
    if ok:
        g.ok("test counts")
    else:
        g.fail("test counts", (out + [err])[-1] if (out or err) else "checker failed")


def check_tag_agreement(g, post_tag, target):
    ok, out, _ = git_lines(["describe", "--tags", "--abbrev=0"])
    tag = out[0].strip() if ok and out else ""
    src = read_source_version()
    if not tag:
        (g.fail if post_tag else g.info)("tag == source version", "no reachable tag")
        return
    tag_v = tag[1:] if tag.startswith("v") else tag
    if tag_v == src:
        g.ok("tag == source version", tag)
    elif post_tag:
        g.fail("tag == source version", "tag=" + tag + " but app/__init__.py=" + src)
    else:
        g.info("tag == source version", "pre-tag stage: latest tag " + tag + ", source " + src)


def main():
    ap = argparse.ArgumentParser(description="pre-release gate")
    ap.add_argument("version", nargs="?", help="target version for this release")
    ap.add_argument("--post-tag", action="store_true", help="verify mode: tag/source agreement becomes a hard check")
    ap.add_argument("--skip-counts", action="store_true", help="skip the test-count check")
    args = ap.parse_args()

    mode = "POST-TAG" if args.post_tag else "PRE-TAG"
    print("=== release ready gate (" + mode + ") ===")
    g = Gate()
    check_branch(g)
    check_version_agreement(g, args.version)
    check_git_clean(g)
    check_svn_clean(g)
    check_ignore_boundary(g)
    check_test_counts(g, args.skip_counts)
    check_tag_agreement(g, args.post_tag, args.version)

    print("")
    if g.fails:
        print("RESULT: FAIL (" + str(g.fails) + " item(s))")
        return 1
    print("RESULT: PASS" + (" (" + str(g.skips) + " skipped)" if g.skips else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
