"""synthetic_v2 귀 확인 페이지를 내 컴퓨터에서 띄운다(python src\\v2\\review_v2.py <태그> → 브라우저가 열림).
'목소리' 탭은 선별 녹음으로 어색한 목소리를 빼고, '문장' 탭은 같은 목소리·문장의 연장·보통·천천히를 나란히 듣고 판정한다.
판정은 누를 때마다 Drive의 screen/voice_review.csv와 <태그>/listen_review.csv에 저장한다.
"""
import csv
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

OUT = __import__("pathlib").Path(r"G:\내 드라이브\tts_dataset\synthetic_v2")
PORT = 8768
LOCK = threading.Lock()  # 빠르게 연달아 누르면 저장이 겹쳐 판정이 사라지는 것을 막는다


def read(path):
    """CSV를 읽는다(없으면 빈 목록)."""
    return list(csv.DictReader(open(path, encoding="utf-8-sig"))) if path.exists() else []


def save(path, key, row):
    """key가 같은 줄을 바꾸거나 더해서 CSV 전체를 다시 쓴다."""
    with LOCK:
        rows = [r for r in read(path) if r[key] != row[key]] + [row]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(rows)


PAGE = """<!doctype html><meta charset="utf-8"><title>합성 v2 확인</title>
<style>body{font-family:'Malgun Gothic',sans-serif;max-width:900px;margin:16px auto;padding:0 16px;background:#fafafa}
.rule{background:#fff7e0;border:1px solid #f0d890;padding:8px 12px;font-size:14px;line-height:1.6}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:10px 0}
.card.done{border-color:#7c7}.card.bad{border-color:#e77}.m{color:#666;font-size:13px}
button{font-size:14px;padding:5px 10px;margin:3px 3px 0 0;cursor:pointer}button.on{outline:3px solid #333}
audio{height:32px;vertical-align:middle}.row{margin:4px 0}.tab{font-size:16px;padding:8px 16px}</style>
<button class="tab" onclick="mode='voice';draw()">목소리</button><button class="tab" onclick="mode='pilot';draw()">시험 생성 문장</button>
<span id="count"></span><div id="rule" class="rule"></div><div id="list"></div>
<script>
let mode='voice',V=[],P=[];
const RULE={voice:"한국어가 자연스럽게 들리면 통과. <b>외국어 억양·로봇·캐릭터·광고 말투·지나치게 연기하는 말투</b>면 빼기. 애매하면 통과(면접 대답처럼 들리는지가 기준).",
 pilot:"<b>연장</b>: 표시 글자를 머뭇거리며 끄는 소리로 들리면 O. 기계음·끊김·글자를 따로따로 읽음·엉뚱한 곳이 늘어나면 X. '구간만'으로 Typecast가 알려 준 연장 위치가 맞는지도 확인.<br><b>보통·천천히</b>: 자연스러우면 O. 천천히인데 특정 글자가 끌려 연장처럼 들리면 X(메모)."};
async function load(){V=await (await fetch('/voices')).json();P=await (await fetch('/pilot')).json();draw()}
const au=(src,a,b)=>`<audio controls preload="none" src="${src}${a!==undefined?'#t='+a+','+b:''}"></audio>`;
function btn(cur,v,f){return `<button class="${cur==v?'on':''}" onclick="${f}">${v}</button>`}
function draw(){rule.innerHTML=RULE[mode];
 if(mode=='voice'){count.textContent=` 확인 ${V.filter(v=>v.verdict).length}/${V.length}, 빼기 ${V.filter(v=>v.verdict=='빼기').length}`;
  list.innerHTML=V.map((v,i)=>`<div class="card ${v.verdict=='통과'?'done':v.verdict=='빼기'?'bad':''}"><b>${i+1}. ${v.voice_name}</b> <span class=m>${v.gender} · ${v.use_cases}</span><div class=row>${au('/screen/'+encodeURIComponent(v.voice_name))}
  ${btn(v.verdict,'통과',`sv(${i},'통과')`)}${btn(v.verdict,'빼기',`sv(${i},'빼기')`)}</div></div>`).join('')}
 else{count.textContent=` 확인 ${P.filter(p=>p.pro&&p.nor&&p.slow).length}/${P.length}`;
  list.innerHTML=P.map((p,i)=>`<div class="card ${p.pro=='X'||p.nor=='X'||p.slow=='X'?'bad':p.pro&&p.nor&&p.slow?'done':''}"><b>${i+1}. ${p.voice_name}</b> <span class=m>${p.sentence_id} ${p.target_type} · 감정 ${p.emotion} ${p.intensity} · 빠르기 ${p.tempo} · 늘인 글자 ${p.k} · 늘어난 ${p.extra_sec}초 · ${p.status}</span>
  <div class=row>연장: ${p.text_pro} ${au('/audio/'+encodeURIComponent(p.f_pro))} <button onclick="span(this,${p.span_s},${p.span_e})">▶ 구간만 (${p.span_s}~${p.span_e}초)</button> ${btn(p.pro,'O',`sp(${i},'pro','O')`)}${btn(p.pro,'X',`sp(${i},'pro','X')`)}</div>
  <div class=row>보통: ${au('/audio/'+encodeURIComponent(p.f_nor))} ${btn(p.nor,'O',`sp(${i},'nor','O')`)}${btn(p.nor,'X',`sp(${i},'nor','X')`)}
   &nbsp; 천천히: ${au('/audio/'+encodeURIComponent(p.f_slow))} ${btn(p.slow,'O',`sp(${i},'slow','O')`)}${btn(p.slow,'X',`sp(${i},'slow','X')`)}</div>
  <input placeholder="메모" value="${p.note||''}" onchange="P[${i}].note=this.value;post('/save_pilot',P[${i}])" style="width:70%"></div>`).join('')}}
function span(b,s,e){const a=b.previousElementSibling;a.currentTime=Math.max(0,s-0.2);a.play();const f=()=>{if(a.currentTime>=e+0.2){a.pause();a.removeEventListener('timeupdate',f)}};a.addEventListener('timeupdate',f)}
async function post(u,d){const r=await fetch(u,{method:'POST',body:JSON.stringify(d)});if(!r.ok)alert('저장 실패: '+await r.text())}
function sv(i,v){V[i].verdict=v;post('/save_voice',V[i]);draw()}
function sp(i,k,v){P[i][k]=v;post('/save_pilot',P[i]);draw()}
load()</script>"""


def pilot_rows(tag):
    """목록 표를 목소리·문장마다 한 줄(연장·보통·천천히 파일)로 묶고, 저장된 판정을 붙인다."""
    by, done = {}, {r["key"]: r for r in read(OUT / tag / "listen_review.csv")}
    for r in read(OUT / tag / "manifest.csv"):
        key = f"{r['voice_name']}__{r['sentence_id']}"
        g = by.setdefault(key, {"key": key, **{k: r[k] for k in ["voice_name", "sentence_id", "target_type", "emotion", "intensity", "tempo"]}})
        g["f_" + {"prolong": "pro", "normal": "nor", "slow": "slow"}[r["kind"]]] = r["file"]
        if r["kind"] == "prolong":
            g.update(text_pro=r["text"], k=r["k"], extra_sec=r["extra_sec"], span_s=r["span_s"] or 0, span_e=r["span_e"] or 0, status=r["status"])
    return [{**g, **{k: done.get(g["key"], {}).get(k, "") for k in ["pro", "nor", "slow", "note"]}}
            for g in by.values() if all(f in g for f in ["f_pro", "f_nor", "f_slow"])]


class Handler(BaseHTTPRequestHandler):
    """페이지·목록·소리를 주고 판정을 저장한다(내 컴퓨터 안에서만)."""

    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def send_audio(self, data):
        """Range 요청을 받아야 브라우저가 재생 위치를 옮길 수 있다('구간만' 재생)."""
        rng = self.headers.get("Range")
        a, b = 0, len(data) - 1
        if rng:
            s, e = rng.split("=")[1].split("-")
            a, b = int(s), int(e) if e else len(data) - 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        self.end_headers()
        self.wfile.write(data[a:b + 1])

    def do_GET(self):
        p = unquote(self.path.split("#")[0])
        name = p.rsplit("/", 1)[-1]
        if p == "/":
            self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif p == "/voices":
            done = {r["voice_id"]: r["verdict"] for r in read(OUT / "screen" / "voice_review.csv")}
            rows = [{**r, "verdict": done.get(r["voice_id"], "")} for r in read(OUT / "screen" / "selected_voices.csv")]
            self.send(json.dumps(rows, ensure_ascii=False).encode(), "application/json")
        elif p == "/pilot":
            self.send(json.dumps(pilot_rows(TAG), ensure_ascii=False).encode(), "application/json")
        elif p.startswith("/screen/") and "/" not in name and ".." not in name:
            self.send_audio((OUT / "screen" / f"{name}.wav").read_bytes())
        elif p.startswith("/audio/") and name.endswith(".wav") and ".." not in name:
            self.send_audio((OUT / TAG / "audio" / name).read_bytes())
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/save_voice":
                save(OUT / "screen" / "voice_review.csv", "voice_id", {k: d[k] for k in ["voice_id", "voice_name", "gender", "verdict"]})
            else:
                save(OUT / TAG / "listen_review.csv", "key", {k: d.get(k, "") for k in ["key", "pro", "nor", "slow", "note"]})
            self.send(b"ok", "text/plain")
        except Exception as e:  # 저장 실패는 화면에 알려서 다시 누르게 한다
            self.send(str(e).encode(), "text/plain", 400)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    TAG = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    url = f"http://127.0.0.1:{PORT}/"
    print("합성 v2 확인 페이지:", url, "(끄려면 Ctrl+C)")
    webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
