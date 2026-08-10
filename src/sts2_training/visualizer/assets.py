from __future__ import annotations

INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STS2 Training — Run Scryer</title>
  <style>
    :root { --ink:#e9e0c8; --muted:#9ca6a2; --panel:rgba(8,16,21,.84); --line:rgba(176,197,185,.19); --cyan:#48d9df; --gold:#e4c36a; --red:#df424d; --green:#6bc468; --amber:#e59d4e; --shadow:0 14px 40px rgba(0,0,0,.45); }
    * { box-sizing:border-box; }
    html,body { width:100%; height:100%; margin:0; overflow:hidden; }
    body { color:var(--ink); font-family:Georgia,'Times New Roman',serif; background:radial-gradient(circle at 50% 35%,rgba(47,91,75,.28),transparent 34%),linear-gradient(rgba(2,12,14,.48),rgba(2,8,12,.88)),repeating-linear-gradient(110deg,#10272a 0 44px,#0d2226 44px 87px,#11292b 87px 132px); }
    body::after { content:''; position:fixed; inset:0; pointer-events:none; box-shadow:inset 0 0 145px rgba(0,0,0,.72); }
    button,select,input { font:inherit; }
    .shell { height:100%; display:grid; grid-template-rows:54px 1fr 76px; position:relative; z-index:1; }
    .topbar { display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:18px; padding:7px 18px; background:linear-gradient(180deg,rgba(17,30,35,.97),rgba(7,17,21,.93)); border-bottom:1px solid #31474b; box-shadow:0 4px 20px rgba(0,0,0,.48); }
    .brand { display:flex; align-items:center; gap:10px; }
    .sigil { width:35px; height:35px; transform:rotate(45deg); border:2px solid var(--cyan); box-shadow:0 0 14px rgba(72,217,223,.4),inset 0 0 12px rgba(72,217,223,.18); background:#102a2f; position:relative; }
    .sigil::after { content:''; position:absolute; inset:8px; border:1px solid var(--gold); }
    .brand strong { display:block; color:#f0dfad; letter-spacing:.12em; font-size:14px; }
    .brand span { color:var(--muted); font:11px ui-monospace,monospace; }
    .top-stats { display:flex; gap:16px; justify-content:center; font-weight:700; text-shadow:0 2px 3px #000; }
    .status-wrap { display:flex; justify-content:end; gap:8px; }
    .pill { padding:5px 9px; border:1px solid var(--line); background:rgba(0,0,0,.28); font:700 10px ui-monospace,monospace; letter-spacing:.09em; }
    .pill.running { border-color:rgba(72,217,223,.58); color:var(--cyan); } .pill.failed { border-color:rgba(223,66,77,.7); color:#ff8690; } .pill.completed { border-color:rgba(107,196,104,.65); color:#a4dfa1; }
    .battle-layout { min-height:0; display:grid; grid-template-columns:244px 1fr 284px; }
    .side { min-height:0; padding:14px 12px; background:linear-gradient(90deg,rgba(4,11,14,.85),rgba(5,12,15,.53)); border-right:1px solid var(--line); overflow:auto; }
    .side.right { border-right:0; border-left:1px solid var(--line); background:linear-gradient(270deg,rgba(4,11,14,.87),rgba(5,12,15,.56)); }
    .panel { background:var(--panel); border:1px solid var(--line); box-shadow:var(--shadow); padding:11px; margin-bottom:10px; }
    .panel h2 { margin:0 0 8px; color:#d4c18c; font-size:12px; letter-spacing:.12em; text-transform:uppercase; }
    .kv { display:grid; grid-template-columns:92px 1fr; gap:5px 8px; font:11px/1.35 ui-monospace,monospace; }
    .kv .k { color:#7f918e; overflow:hidden; text-overflow:ellipsis; } .kv .v { color:#d7dfd9; overflow-wrap:anywhere; }
    .arena { min-width:0; min-height:0; display:grid; grid-template-rows:1fr 205px; overflow:hidden; }
    .stage { position:relative; min-height:0; padding:38px 34px 12px; display:flex; align-items:end; justify-content:space-between; gap:70px; }
    .stage::before { content:''; position:absolute; left:0; right:0; bottom:0; height:43%; background:radial-gradient(ellipse at center,rgba(59,76,53,.28),rgba(4,9,9,.18) 55%,rgba(0,0,0,.45)); }
    .entity { position:relative; z-index:1; width:180px; text-align:center; }
    .portrait { width:116px; height:150px; margin:0 auto 9px; border:1px solid rgba(160,185,173,.22); background:radial-gradient(circle at 50% 33%,rgba(229,195,106,.15),transparent 14%),radial-gradient(ellipse at 50% 58%,rgba(94,117,105,.52),rgba(17,29,28,.68) 58%,rgba(6,10,12,.9)); clip-path:polygon(26% 0,74% 0,91% 22%,84% 100%,16% 100%,9% 22%); display:grid; place-items:center; font-size:42px; color:rgba(225,236,225,.63); box-shadow:inset 0 -18px 30px rgba(0,0,0,.65),0 10px 30px rgba(0,0,0,.45); }
    .entity-name { font-size:13px; margin-bottom:5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .hpbar { height:11px; border:2px solid #2b191b; background:#1a0c0e; box-shadow:0 1px 4px #000; } .hpfill { height:100%; background:linear-gradient(180deg,#ef5962,#bd2535); }
    .hptext { font:700 10px ui-monospace,monospace; margin-top:3px; } .chips { min-height:24px; margin-top:5px; display:flex; flex-wrap:wrap; justify-content:center; gap:4px; }
    .chip { padding:2px 5px; background:rgba(7,12,14,.85); border:1px solid #415354; font:9px ui-monospace,monospace; }
    .intent { position:absolute; top:-28px; left:50%; transform:translateX(-50%); color:#ff9a78; font-weight:700; }
    .enemies { position:relative; z-index:1; display:flex; gap:24px; align-items:end; justify-content:flex-end; min-width:240px; max-width:60%; overflow-x:auto; padding-top:32px; }
    .enemies .entity { width:150px; flex:0 0 150px; } .enemies .portrait { width:104px; height:127px; color:rgba(211,125,96,.72); }
    .empty { color:#6d7c78; font-style:italic; align-self:center; }
    .hand-zone { display:grid; grid-template-columns:86px 1fr 110px; align-items:end; padding:0 18px 12px; background:linear-gradient(180deg,transparent,rgba(0,0,0,.55)); }
    .energy { width:68px; height:68px; display:grid; place-items:center; align-self:center; margin-bottom:15px; background:linear-gradient(145deg,#217b69,#0b3a38); border:3px solid #75d9ae; color:#eaf9ca; font-size:25px; font-weight:700; clip-path:polygon(25% 4%,75% 4%,98% 50%,75% 96%,25% 96%,2% 50%); }
    .hand { height:188px; display:flex; align-items:end; justify-content:center; padding:0 12px; overflow:visible; }
    .card { width:124px; height:176px; flex:0 0 124px; margin-left:-26px; position:relative; overflow:hidden; border:3px solid #556c73; border-radius:12px 12px 18px 18px; background:linear-gradient(160deg,#5c6c76,#263039 21%,#171d22 22% 100%); box-shadow:0 8px 15px rgba(0,0,0,.55); transform-origin:50% 100%; }
    .card:first-child { margin-left:0; } .card.available { border-color:#3cc5cc; } .card.selected { transform:translateY(-16px) scale(1.05)!important; border-color:#f0cf6a; box-shadow:0 0 0 2px #6c5730,0 12px 22px rgba(0,0,0,.62),0 0 22px rgba(228,195,106,.34); z-index:30!important; } .card.speculative.selected { border-color:var(--amber); }
    .card-cost { position:absolute; left:5px; top:3px; width:26px; height:26px; border-radius:50%; background:#4c89a7; border:2px solid #9fd7e3; display:grid; place-items:center; font:700 13px ui-monospace,monospace; }
    .card-name { height:31px; margin:6px 8px 0 27px; text-align:center; font-size:11px; overflow:hidden; } .card-art { height:66px; margin:2px 7px; border:1px solid #56646a; background:radial-gradient(circle at 62% 38%,rgba(230,96,63,.72),transparent 18%),linear-gradient(135deg,#1f5962,#342a37 50%,#10171b); display:grid; place-items:center; } .card-text { padding:6px 8px; font-size:9px; line-height:1.2; text-align:center; max-height:55px; overflow:hidden; }
    .end-turn { align-self:center; margin-bottom:18px; padding:13px 12px; background:#253c3d; border:2px solid #8aa19a; color:#f4e6bd; text-transform:uppercase; }
    .timeline { display:flex; flex-direction:column; gap:4px; max-height:250px; overflow:auto; } .tick { width:100%; text-align:left; border:1px solid transparent; background:rgba(255,255,255,.025); color:#bac5bf; padding:6px; cursor:pointer; font:10px/1.25 ui-monospace,monospace; } .tick.active { border-color:var(--cyan); color:#eefbf9; background:rgba(72,217,223,.09); } .commit { color:var(--gold); } .branch { color:var(--amber); }
    pre { white-space:pre-wrap; overflow-wrap:anywhere; font:10px/1.4 ui-monospace,monospace; color:#a9bbb5; max-height:280px; overflow:auto; }
    .transport { display:grid; grid-template-columns:auto minmax(120px,1fr) auto; align-items:center; gap:14px; padding:10px 16px; background:linear-gradient(180deg,rgba(7,15,18,.93),rgba(3,9,11,.98)); border-top:1px solid #314547; }
    .controls { display:flex; gap:7px; align-items:center; } .ctrl,.start-run { border:1px solid #52666a; background:linear-gradient(#23363a,#142327); color:#e6ddc5; padding:8px 11px; cursor:pointer; } .start-run { border-color:#a5833c; color:#f2d783; min-width:105px; font-weight:700; letter-spacing:.08em; } .start-run:disabled { opacity:.5; cursor:not-allowed; }
    .scrubber { width:100%; accent-color:#49c9cf; } .frame-label { font:11px ui-monospace,monospace; color:#99aaa5; min-width:110px; text-align:right; }
    .toast { position:fixed; left:50%; top:68px; transform:translateX(-50%); z-index:90; padding:8px 12px; background:rgba(43,12,16,.94); border:1px solid #9a3844; color:#ffb6bd; display:none; max-width:72vw; font:11px ui-monospace,monospace; } .toast.show { display:block; } .muted { color:var(--muted); }
    @media (max-width:1050px) { .battle-layout{grid-template-columns:190px 1fr 220px}.side{padding:9px 7px}.card{width:108px;flex-basis:108px;height:164px}.hand-zone{grid-template-columns:68px 1fr 90px} }
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand"><div class="sigil"></div><div><strong>RUN SCRYER</strong><span>STS2 Training visualizer</span></div></div>
    <div class="top-stats"><span>♥ <b id="top-hp">—/—</b></span><span>● <b id="top-gold">—</b></span><span>▟ <b id="top-floor">—</b></span></div>
    <div class="status-wrap"><span id="mode-pill" class="pill">MODE</span><span id="state-pill" class="pill">IDLE</span></div>
  </header>
  <div class="battle-layout">
    <aside class="side"><div class="panel"><h2>Run</h2><div class="kv" id="run-kv"></div></div><div class="panel"><h2>Decision</h2><div class="kv" id="decision-kv"></div></div><div class="panel"><h2>Piles</h2><div class="kv" id="piles-kv"></div></div></aside>
    <main class="arena"><section class="stage"><div class="entity" id="player"></div><div class="enemies" id="enemies"></div></section><section class="hand-zone"><div class="energy" id="energy">—</div><div class="hand" id="hand"></div><button class="end-turn" disabled>End Turn</button></section></main>
    <aside class="side right"><div class="panel"><h2>Selected Action</h2><div id="action-detail" class="muted">No event selected</div></div><div class="panel"><h2>Timeline</h2><div class="timeline" id="timeline"></div></div><div class="panel"><details><summary>Raw event JSON</summary><pre id="raw-json">{}</pre></details></div></aside>
  </div>
  <footer class="transport"><div class="controls"><button id="start-run" class="start-run">START RUN</button><button id="prev" class="ctrl">◀</button><button id="play" class="ctrl">▶</button><button id="next" class="ctrl">▶|</button><select id="speed" class="ctrl"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div><input id="scrubber" class="scrubber" type="range" min="0" max="0" value="0"><div class="frame-label" id="frame-label">0 / 0</div></footer>
</div>
<div id="toast" class="toast"></div>
<script>
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {mode:'replay', lifecycle:'idle', events:[], cursor:-1, frame:-1, playing:false, speed:1, followLive:true, timer:null};
  const text = (value, fallback='—') => value === undefined || value === null || value === '' ? fallback : String(value);
  const esc = value => text(value,'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const compact = (value, limit=88) => { const s = typeof value === 'string' ? value : JSON.stringify(value); return s && s.length > limit ? s.slice(0,limit-1)+'…' : (s || '—'); };
  const pct = (current,max) => { const c=Number(current),m=Number(max); return Number.isFinite(c)&&Number.isFinite(m)&&m>0 ? Math.max(0,Math.min(100,c/m*100)) : 0; };

  function renderEntity(target, entity, kind, index=0) {
    const value = entity || {}; const hp=value.current_hp, maxHp=value.max_hp ?? hp;
    const statuses = Array.isArray(value.statuses) ? value.statuses : [];
    const statusHtml = [value.block ? `<span class="chip">◇ ${esc(value.block)}</span>` : '', ...statuses.slice(0,5).map(s => `<span class="chip">${esc(compact(s,22))}</span>`)].join('');
    const icon = kind === 'player' ? '♙' : ['♞','♜','♝'][index % 3];
    target.innerHTML = `${value.intent !== undefined && value.intent !== null ? `<div class="intent">⚔ ${esc(compact(value.intent,18))}</div>` : ''}<div class="portrait">${icon}</div><div class="entity-name">${esc(value.name || (kind==='player'?'Player':`Enemy ${index+1}`))}</div><div class="hpbar"><div class="hpfill" style="width:${pct(hp,maxHp)}%"></div></div><div class="hptext">${esc(text(hp))} / ${esc(text(maxHp))}</div><div class="chips">${statusHtml}</div>`;
  }

  function renderBattle(event) {
    const frame = event?.frame || {}, player = frame.player || {}, resources = frame.resources || {}, piles = frame.piles || {};
    renderEntity($('player'), player, 'player');
    $('enemies').innerHTML=''; const enemies = Array.isArray(frame.enemies) ? frame.enemies : [];
    if (!enemies.length) $('enemies').innerHTML='<div class="empty">No enemy state in this frame</div>';
    enemies.forEach((enemy,i) => { const node=document.createElement('div'); node.className='entity'; renderEntity(node,enemy,'enemy',i); $('enemies').appendChild(node); });
    $('top-hp').textContent=`${text(player.current_hp)}/${text(player.max_hp)}`; $('top-gold').textContent=text(resources.gold); $('top-floor').textContent=text(resources.floor);
    $('energy').textContent = resources.max_energy === undefined || resources.max_energy === null ? text(resources.energy) : `${text(resources.energy)}/${text(resources.max_energy)}`;
    renderHand(Array.isArray(frame.hand)?frame.hand:[], event);
    setKv('run-kv', [['boundary',frame.boundary],['character',resources.character],['floor',resources.floor],['outcome',frame.outcome],['event count',state.events.length]]);
    setKv('piles-kv', [['draw',piles.draw],['discard',piles.discard],['exhaust',piles.exhaust],['deck',piles.deck]]);
  }

  function renderHand(cards,event) {
    const hand=$('hand'); hand.innerHTML=''; if (!cards.length) { hand.innerHTML='<div class="empty">No card/action entries</div>'; return; }
    const spread=Math.min(7,38/Math.max(cards.length-1,1));
    cards.forEach((card,i) => { const node=document.createElement('div'); const selected=event?.selected_action_id && card.action_id===event.selected_action_id; node.className=`card ${card.is_available!==false?'available':''} ${selected?'selected':''} ${event?.phase==='beam_explore'?'speculative':''}`; const angle=(i-(cards.length-1)/2)*spread, lift=Math.abs(i-(cards.length-1)/2)*1.6; node.style.transform=`rotate(${angle}deg) translateY(${lift}px)`; node.style.zIndex=String(i+1); node.innerHTML=`<div class="card-cost">${esc(card.cost)}</div><div class="card-name">${esc(card.name || card.action_type || card.action_id)}</div><div class="card-art">✦</div><div class="card-text">${esc(compact(card.description,120))}</div>`; hand.appendChild(node); });
  }

  function setKv(id,pairs) { $(id).innerHTML=pairs.map(([k,v])=>`<div class="k">${esc(k)}</div><div class="v">${esc(compact(v,90))}</div>`).join(''); }
  function renderMeta(event) { setKv('decision-kv',[['event',event?.event],['operation',event?.operation],['branch',event?.branch_id],['decision',event?.decision_point_id],['phase',event?.phase],['frame source',event?.frame_source],['logged',event?.logged_at]]); const action=event?.selected_action; $('action-detail').innerHTML=action ? `<div class="kv">${Object.entries(action).map(([k,v])=>`<div class="k">${esc(k)}</div><div class="v">${esc(compact(v,110))}</div>`).join('')}</div>` : '<span class="muted">No selected action in this event</span>'; $('raw-json').textContent=JSON.stringify(event?.raw || {},null,2); }
  function renderTimeline() { const timeline=$('timeline'); timeline.innerHTML=''; const start=Math.max(0,state.events.length-80); state.events.slice(start).forEach((event,offset)=>{ const i=start+offset,button=document.createElement('button'); const cls=event.operation==='commit_action'?'commit':(event.phase==='beam_explore'?'branch':''); button.className=`tick ${i===state.frame?'active':''}`; button.innerHTML=`#${i+1} <span class="${cls}">${esc(event.operation || event.event)}</span> ${esc(compact(event.selected_action_id || event.frame?.outcome || event.branch_id || '',26))}`; button.onclick=()=>{state.playing=false;state.followLive=false;setFrame(i)}; timeline.appendChild(button); }); if(state.followLive)timeline.scrollTop=timeline.scrollHeight; }
  function setFrame(index) { if(!state.events.length){state.frame=-1;$('frame-label').textContent='0 / 0';return} state.frame=Math.max(0,Math.min(index,state.events.length-1)); const event=state.events[state.frame]; renderBattle(event); renderMeta(event); renderTimeline(); $('scrubber').max=String(Math.max(0,state.events.length-1)); $('scrubber').value=String(state.frame); $('frame-label').textContent=`${state.frame+1} / ${state.events.length}`; }
  function appendEvents(events) { if(!Array.isArray(events)||!events.length)return; events.forEach(event=>{state.events.push(event);state.cursor=Math.max(state.cursor,Number(event.index))}); if(state.mode==='live'&&state.followLive)setFrame(state.events.length-1); else if(state.frame<0)setFrame(0); else { $('scrubber').max=String(state.events.length-1); renderTimeline(); } }
  function playTick(){ clearTimeout(state.timer); if(!state.playing||!state.events.length)return; if(state.frame>=state.events.length-1){ if(state.mode==='live'&&state.lifecycle==='running'){state.timer=setTimeout(playTick,300);return} state.playing=false;$('play').textContent='▶';return } setFrame(state.frame+1); state.timer=setTimeout(playTick,Math.max(80,760/state.speed)); }
  function toast(message){const node=$('toast');node.textContent=message;node.classList.add('show');setTimeout(()=>node.classList.remove('show'),4200)}
  async function jsonFetch(url,options){const response=await fetch(url,options);const body=await response.json().catch(()=>({}));if(!response.ok)throw new Error(body.error||`${response.status} ${response.statusText}`);return body}
  async function refreshStatus(){try{const data=await jsonFetch('/api/status');state.mode=data.mode;state.lifecycle=data.state;$('mode-pill').textContent=data.mode.toUpperCase();$('state-pill').textContent=String(data.state||'ready').toUpperCase();$('state-pill').className=`pill ${data.state||''}`;$('start-run').style.display=data.mode==='live'?'':'none';$('start-run').disabled=data.mode!=='live'||data.state!=='idle';if(data.error)toast(data.error)}catch(error){toast(error.message)}}
  async function pollEvents(){try{const data=await jsonFetch(`/api/events?after=${state.cursor}`);appendEvents(data.events||[])}catch(error){toast(error.message)}if(state.mode==='live')setTimeout(pollEvents,350)}
  $('start-run').onclick=async()=>{try{await jsonFetch('/api/live/start',{method:'POST'});state.followLive=true;await refreshStatus()}catch(error){toast(error.message)}};
  $('play').onclick=()=>{state.playing=!state.playing;state.followLive=false;$('play').textContent=state.playing?'Ⅱ':'▶';playTick()}; $('prev').onclick=()=>{state.playing=false;state.followLive=false;$('play').textContent='▶';setFrame(state.frame-1)}; $('next').onclick=()=>{state.playing=false;state.followLive=false;$('play').textContent='▶';setFrame(state.frame+1)}; $('speed').onchange=e=>{state.speed=Number(e.target.value)||1}; $('scrubber').oninput=e=>{state.playing=false;state.followLive=false;setFrame(Number(e.target.value))};
  (async()=>{await refreshStatus();const data=await jsonFetch('/api/events?after=-1').catch(error=>{toast(error.message);return{events:[]}});appendEvents(data.events||[]);if(state.mode==='live')setTimeout(pollEvents,350);setInterval(refreshStatus,900)})();
})();
</script>
</body>
</html>'''
