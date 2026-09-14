"""intro_numpy ノートブック用の自己チェック。

使い方（ノートブック冒頭の準備セル）:
    !wget -q -O mdpcheck.py {RAW}/check/mdpcheck.py
    from mdpcheck import setup
    check = setup("05")

各タスクセルの直後:
    check("q3")
"""

import json
import sys
import urllib.request

import numpy as np

RAW = "https://raw.githubusercontent.com/slp-hu/MediaDataProgramming/main/intro_numpy"

_OK = "\u2705"
_NG = "\u274c"


def _decode(o):
    """build_checks.py が書いた JSON を Python の値に戻す。"""
    if isinstance(o, dict) and "__kind__" in o:
        k = o["__kind__"]
        if k == "ndarray":
            return np.array(o["data"], dtype=o["dtype"])
        if k == "tuple":
            return tuple(_decode(x) for x in o["data"])
        if k == "complex":
            return complex(o["re"], o["im"])
        if k == "opaque":
            return _Opaque(o["repr"])
    if isinstance(o, list):
        return [_decode(x) for x in o]
    return o


class _Opaque:
    """JSON にできなかった値。存在だけを見る。"""

    def __init__(self, text):
        self.text = text


def _shape_of(v):
    a = np.asarray(v)
    return a.shape


def _shape_hint(want, got):
    """形が違うとき、何を直せばよいかを一言で返す。"""
    w, g = _shape_of(want), _shape_of(got)
    if w == g:
        return ""
    if len(w) == 2 and w[0] == 1 and len(g) == 1 and w[1] == g[0]:
        return "行ベクトルは np.array([[...]]) のように [ ] を二重にします。"
    if len(w) == 2 and w[1] == 1 and len(g) == 1 and w[0] == g[0]:
        return "列ベクトルは np.array([[...]]).T のように作ります。"
    if len(w) == 1 and len(g) == 2 and 1 in g and w[0] == max(g):
        return "ここは 1 次元の配列です。[ ] は一重にします。"
    try:
        if int(np.prod(w)) == int(np.prod(g)):
            return "要素は合っていますが、並びの形が違います。"
    except Exception:
        pass
    return ""


def _same(want, got, rtol=1e-6, atol=1e-9, loose=False):
    """(一致したか, 理由) を返す。理由は '形' か '値' か ''。

    loose=True のときは形の違いを問わず、並べた要素だけを比べる。
    """
    if isinstance(want, _Opaque):
        return True, ""
    if isinstance(want, str) or isinstance(got, str):
        return (want == got), ("" if want == got else "値")
    if isinstance(want, tuple) or isinstance(got, tuple):
        try:
            ok = tuple(want) == tuple(got)
        except TypeError:
            ok = False
        return ok, ("" if ok else "値")
    try:
        w = np.asarray(want)
        g = np.asarray(got)
    except Exception:
        return (want == got), ("" if want == got else "値")

    if loose:
        if w.size != g.size:
            return False, "形"
        w, g = w.reshape(-1), g.reshape(-1)
    elif w.shape != g.shape:
        return False, "形"
    if w.dtype.kind in "bUSO" or g.dtype.kind in "bUSO":
        ok = bool(np.all(w == g))
        return ok, ("" if ok else "値")
    try:
        wf = w.astype(float)
        gf = g.astype(float)
    except (TypeError, ValueError):
        ok = bool(np.all(w == g))
        return ok, ("" if ok else "値")
    both_nan = np.isnan(wf) & np.isnan(gf)
    close = np.abs(gf - wf) <= atol + rtol * np.abs(wf)
    ok = bool(np.all(close | both_nan))
    return ok, ("" if ok else "値")


class Checker:
    def __init__(self, spec, nb):
        self.opts = spec.get("__opts__", {})
        self.spec = {k: v for k, v in spec.items() if not k.startswith("__")}
        self.nb = nb

    def __call__(self, task=None):
        env = sys._getframe(1).f_globals
        keys = [task] if task else sorted(self.spec)
        for k in keys:
            self._one(k, env)

    def _one(self, task, env):
        want = self.spec.get(task)
        if want is None:
            print("%s %s というタスクはありません" % (_NG, task))
            return

        loose = self.opts.get(task, {}).get("shape") == "loose"
        msgs = []
        for name, value in want.items():
            if name not in env:
                msgs.append("変数 %s が定義されていません。" % name)
                continue
            w = _decode(value)
            ok, why = _same(w, env[name], loose=loose)
            if ok:
                continue
            if why == "形":
                hint = _shape_hint(w, env[name])
                msgs.append(
                    "変数 %s の形が正しくありません。（正しくは %s、いまは %s）%s"
                    % (name, _shape_of(w), _shape_of(env[name]),
                       ("\n   " + hint) if hint else "")
                )
            else:
                msgs.append("変数 %s の値が正しくありません。" % name)

        if msgs:
            for m in msgs:
                print("%s %s" % (_NG, m))
        else:
            print("%s 正解です。" % _OK)

    def all(self):
        self()


def setup(nb, base=RAW, local=None):
    """期待値を読み込んで check を返す。nb は '05' のような2桁の文字列。"""
    name = "expected_%s.json" % nb
    if local:
        with open(local, encoding="utf-8") as f:
            spec = json.load(f)
    else:
        url = "%s/check/%s" % (base, name)
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                spec = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print("%s 期待値を取得できませんでした（%s）" % (_NG, e))
            spec = {}
    return Checker(spec, nb)
