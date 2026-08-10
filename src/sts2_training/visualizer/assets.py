from __future__ import annotations

INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STS2 Training — Run Scryer</title>
  <style>
    :root {
      --ink:#eee6cf; --muted:#91a09c; --panel:rgba(7,15,19,.88); --panel-soft:rgba(11,24,28,.72);
      --line:rgba(174,199,190,.18); --line-strong:rgba(174,199,190,.32); --cyan:#49d6dc;
      --gold:#e2c06a; --red:#e24b57; --green:#71c56b; --amber:#e39a4d; --violet:#b995e8;
      --shadow:0 16px 42px rgba(0,0,0,.46); --mono:ui-monospace,SFMono-Regular,Consolas,monospace;
    }
    * { box-sizing:border-box; }
    html,body { width:100%; height:100%; margin:0; overflow:hidden; }
    body {
      color:var(--ink); font-family:Georgia,'Times New Roman',serif;
      background:radial-gradient(circle at 50% 28%,rgba(47,91,75,.30),transparent 34%),
        linear-gradient(rgba(2,12,14,.45),rgba(2,8,12,.90)),
        repeating-linear-gradient(110deg,#10272a 0 44px,#0d2226 44px 87px,#11292b 87px 132px);
    }
    body::after { content:''; position:fixed; inset:0; pointer-events:none; box-shadow:inset 0 0 150px rgba(0,0,0,.74); }
    button,select,input { font:inherit; }
    button { border-radius:4px; }
    .shell { height:100%; display:grid; grid-template-rows:58px minmax(0,1fr) 72px; position:relative; z-index:1; }

    .topbar {
      display:grid; grid-template-columns:minmax(210px,1fr) auto minmax(210px,1fr); align-items:center; gap:18px;
      padding:8px 18px; background:linear-gradient(180deg,rgba(17,31,36,.98),rgba(7,17,21,.94));
      border-bottom:1px solid #31474b; box-shadow:0 4px 20px rgba(0,0,0,.48);
    }
    .brand { display:flex; align-items:center; gap:11px; min-width:0; }
    .sigil { width:34px; height:34px; flex:0 0 auto; transform:rotate(45deg); border:2px solid var(--cyan); background:#102a2f; box-shadow:0 0 14px rgba(72,217,223,.38),inset 0 0 12px rgba(72,217,223,.16); position:relative; }
    .sigil::after { content:''; position:absolute; inset:8px; border:1px solid var(--gold); }
    .brand-copy { min-width:0; }
    .brand strong { display:block; color:#f0dfad; letter-spacing:.13em; font-size:14px; white-space:nowrap; }
    .brand span { color:var(--muted); font:10px var(--mono); white-space:nowrap; }
    .top-stats { display:flex; gap:10px; justify-content:center; font-weight:700; text-shadow:0 2px 3px #000; }
    .top-stat { min-width:88px; padding:6px 10px; text-align:center; border:1px solid rgba(255,255,255,.08); background:rgba(0,0,0,.23); font-size:13px; }
    .status-wrap { display:flex; justify-content:end; gap:8px; }
    .pill { padding:6px 9px; border:1px solid var(--line); background:rgba(0,0,0,.28); font:700 10px var(--mono); letter-spacing:.09em; }
    .pill.running { border-color:rgba(72,217,223,.58); color:var(--cyan); }
    .pill.failed { border-color:rgba(223,66,77,.7); color:#ff8690; }
    .pill.completed { border-color:rgba(107,196,104,.65); color:#a4dfa1; }

    .battle-layout { min-height:0; display:grid; grid-template-columns:clamp(210px,17vw,270px) minmax(0,1fr) clamp(238px,19vw,300px); }
    .side { min-height:0; padding:12px 10px; background:linear-gradient(90deg,rgba(4,11,14,.87),rgba(5,12,15,.54)); border-right:1px solid var(--line); overflow:auto; scrollbar-width:thin; }
    .side.right { border-right:0; border-left:1px solid var(--line); background:linear-gradient(270deg,rgba(4,11,14,.89),rgba(5,12,15,.57)); }
    .panel { background:var(--panel); border:1px solid var(--line); box-shadow:var(--shadow); padding:11px; margin-bottom:10px; }
    .panel:last-child { margin-bottom:0; }
    .panel h2 { display:flex; align-items:center; justify-content:space-between; margin:0 0 8px; color:#d7c38d; font-size:11px; letter-spacing:.14em; text-transform:uppercase; }
    .panel h2::after { content:''; width:34px; border-top:1px solid rgba(226,192,106,.28); }
    .kv { display:grid; grid-template-columns:minmax(72px,88px) minmax(0,1fr); gap:5px 8px; font:10px/1.38 var(--mono); }
    .kv .k { color:#7f918e; overflow:hidden; text-overflow:ellipsis; }
    .kv .v { color:#d7dfd9; overflow-wrap:anywhere; }
    details summary { color:#c8b985; cursor:pointer; font-size:11px; }

    .arena { min-width:0; min-height:0; display:grid; grid-template-rows:minmax(270px,1fr) 214px; overflow:hidden; position:relative; }
    .stage {
      position:relative; min-height:0; padding:42px clamp(24px,3vw,48px) 18px;
      display:grid; grid-template-columns:minmax(160px,210px) minmax(260px,1fr); align-items:end; gap:clamp(32px,6vw,90px);
    }
    .stage::before { content:''; position:absolute; inset:0 0 0; background:radial-gradient(ellipse at 54% 72%,rgba(74,88,61,.30),rgba(5,10,11,.12) 54%,rgba(0,0,0,.34)); }
    .stage::after { content:''; position:absolute; left:8%; right:8%; bottom:17px; border-top:1px solid rgba(181,200,174,.13); box-shadow:0 0 26px rgba(115,174,143,.12); }
    .boundary-badge { position:absolute; top:14px; left:50%; transform:translateX(-50%); z-index:3; padding:5px 12px; border:1px solid var(--line); background:rgba(3,10,12,.72); color:#9db0aa; font:10px var(--mono); letter-spacing:.08em; }

    .entity { position:relative; z-index:2; width:100%; text-align:center; min-width:0; }
    .entity-card { max-width:210px; margin:0 auto; padding:10px 11px 9px; background:linear-gradient(180deg,rgba(14,28,31,.74),rgba(5,12,14,.86)); border:1px solid rgba(151,178,168,.16); box-shadow:0 12px 28px rgba(0,0,0,.30); }
    .portrait { width:104px; height:126px; margin:0 auto 8px; border:1px solid rgba(160,185,173,.23); background:radial-gradient(circle at 50% 33%,rgba(229,195,106,.15),transparent 14%),radial-gradient(ellipse at 50% 58%,rgba(94,117,105,.52),rgba(17,29,28,.68) 58%,rgba(6,10,12,.9)); clip-path:polygon(26% 0,74% 0,91% 22%,84% 100%,16% 100%,9% 22%); display:grid; place-items:center; font-size:38px; color:rgba(225,236,225,.63); box-shadow:inset 0 -18px 30px rgba(0,0,0,.65),0 10px 24px rgba(0,0,0,.4); }
    .entity-name { font-size:13px; margin-bottom:6px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .hpbar { height:12px; border:2px solid #2b191b; background:#1a0c0e; box-shadow:0 1px 4px #000; }
    .hpfill { height:100%; background:linear-gradient(180deg,#ef5962,#bd2535); }
    .hptext { display:flex; justify-content:space-between; gap:6px; font:700 10px var(--mono); margin-top:4px; color:#ead6d4; }
    .combat-tags { min-height:22px; margin-top:6px; display:flex; flex-wrap:wrap; justify-content:center; gap:4px; }
    .chip { padding:2px 5px; background:rgba(7,12,14,.88); border:1px solid #415354; color:#c7d1cd; font:9px var(--mono); }
    .chip.block { border-color:rgba(111,181,206,.55); color:#bde4f1; }
    .intent { position:absolute; top:-32px; left:50%; transform:translateX(-50%); color:#ff9a78; font-weight:700; font-size:12px; white-space:nowrap; text-shadow:0 2px 4px #000; }

    .power-strip { margin-top:7px; min-height:28px; display:flex; flex-wrap:wrap; justify-content:center; gap:5px; }
    .power { display:inline-flex; align-items:center; min-width:0; max-width:100%; border:1px solid rgba(185,149,232,.42); background:linear-gradient(180deg,rgba(71,43,91,.58),rgba(29,20,41,.78)); box-shadow:inset 0 0 10px rgba(185,149,232,.08); font:9px var(--mono); }
    .power-rune { width:20px; height:20px; flex:0 0 20px; display:grid; place-items:center; color:#eadcff; background:rgba(185,149,232,.16); border-right:1px solid rgba(185,149,232,.30); font-weight:800; }
    .power-name { min-width:0; padding:3px 5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#d9c9ee; }
    .power-amount { padding:3px 5px; border-left:1px solid rgba(185,149,232,.24); color:#fff0aa; font-weight:800; }
    .power.more { color:#bfb1ce; padding:4px 6px; }

    .enemies { position:relative; z-index:2; display:flex; gap:16px; align-items:end; justify-content:flex-end; min-width:0; overflow-x:auto; padding:34px 2px 0; scrollbar-width:thin; }
    .enemies .entity { width:172px; flex:0 0 172px; }
    .enemies .entity-card { max-width:172px; }
    .enemies .portrait { width:92px; height:106px; color:rgba(211,125,96,.72); }
    .empty { color:#6d7c78; font-style:italic; align-self:center; font-size:12px; }

    .hand-zone { display:grid; grid-template-columns:82px minmax(0,1fr) 104px; align-items:end; gap:8px; padding:0 16px 12px; background:linear-gradient(180deg,rgba(0,0,0,.04),rgba(0,0,0,.58)); border-top:1px solid rgba(255,255,255,.04); }
    .energy-wrap { align-self:center; text-align:center; margin-bottom:12px; }
    .energy { width:66px; height:66px; margin:0 auto; display:grid; place-items:center; background:linear-gradient(145deg,#217b69,#0b3a38); border:3px solid #75d9ae; color:#eaf9ca; font-size:23px; font-weight:700; clip-path:polygon(25% 4%,75% 4%,98% 50%,75% 96%,25% 96%,2% 50%); }
    .energy-label { margin-top:4px; color:#81a99a; font:9px var(--mono); letter-spacing:.08em; }
    .hand { height:195px; min-width:0; display:flex; align-items:end; justify-content:center; padding:0 8px; overflow:visible; }
    .card { width:120px; height:174px; flex:0 0 120px; margin-left:-24px; position:relative; overflow:hidden; border:3px solid #556c73; border-radius:12px 12px 18px 18px; background:linear-gradient(160deg,#5c6c76,#263039 21%,#171d22 22% 100%); box-shadow:0 8px 15px rgba(0,0,0,.55); transform-origin:50% 100%; transition:filter .12s ease; }
    .card:first-child { margin-left:0; }
    .card.available { border-color:#3cc5cc; }
    .card.selected { transform:translateY(-16px) scale(1.05)!important; border-color:#f0cf6a; box-shadow:0 0 0 2px #6c5730,0 12px 22px rgba(0,0,0,.62),0 0 22px rgba(228,195,106,.34); z-index:30!important; }
    .card.speculative.selected { border-color:var(--amber); }
    .card-cost { position:absolute; left:5px; top:3px; width:26px; height:26px; border-radius:50%; background:#4c89a7; border:2px solid #9fd7e3; display:grid; place-items:center; font:700 13px var(--mono); }
    .card-name { height:31px; margin:6px 8px 0 27px; text-align:center; font-size:11px; overflow:hidden; }
    .card-art { height:66px; margin:2px 7px; border:1px solid #56646a; background:radial-gradient(circle at 62% 38%,rgba(230,96,63,.72),transparent 18%),linear-gradient(135deg,#1f5962,#342a37 50%,#10171b); display:grid; place-items:center; }
    .card-text { padding:6px 8px; font-size:9px; line-height:1.2; text-align:center; max-height:55px; overflow:hidden; }
    .end-turn { align-self:center; margin-bottom:14px; padding:12px 10px; background:#253c3d; border:2px solid #8aa19a; color:#f4e6bd; text-transform:uppercase; font-size:10px; letter-spacing:.07em; }

    .action-title { margin-bottom:8px; color:#f1dda0; font-size:12px; font-weight:700; }
    .timeline { display:flex; flex-direction:column; gap:4px; max-height:min(34vh,310px); overflow:auto; scrollbar-width:thin; }
    .tick { width:100%; text-align:left; border:1px solid transparent; background:rgba(255,255,255,.025); color:#bac5bf; padding:6px; cursor:pointer; font:10px/1.25 var(--mono); }
    .tick.active { border-color:var(--cyan); color:#eefbf9; background:rgba(72,217,223,.09); }
    .commit { color:var(--gold); } .branch { color:var(--amber); }
    pre { white-space:pre-wrap; overflow-wrap:anywhere; font:10px/1.4 var(--mono); color:#a9bbb5; max-height:260px; overflow:auto; }

    .transport { display:grid; grid-template-columns:auto minmax(120px,1fr) auto; align-items:center; gap:14px; padding:10px 16px; background:linear-gradient(180deg,rgba(7,15,18,.94),rgba(3,9,11,.99)); border-top:1px solid #314547; }
    .controls { display:flex; gap:7px; align-items:center; }
    .ctrl,.start-run { border:1px solid #52666a; background:linear-gradient(#23363a,#142327); color:#e6ddc5; padding:8px 11px; cursor:pointer; }
    .ctrl:hover,.start-run:hover:not(:disabled) { filter:brightness(1.14); }
    .start-run { border-color:#a5833c; color:#f2d783; min-width:105px; font-weight:700; letter-spacing:.08em; }
    .start-run:disabled { opacity:.5; cursor:not-allowed; }
    .scrubber { width:100%; accent-color:#49c9cf; }
    .frame-label { font:11px var(--mono); color:#99aaa5; min-width:110px; text-align:right; }
    .toast { position:fixed; left:50%; top:68px; transform:translateX(-50%); z-index:90; padding:8px 12px; background:rgba(43,12,16,.94); border:1px solid #9a3844; color:#ffb6bd; display:none; max-width:72vw; font:11px var(--mono); }
    .toast.show { display:block; } .muted { color:var(--muted); }

    @media (max-width:1180px) {
      .battle-layout { grid-template-columns:205px minmax(0,1fr) 238px; }
      .stage { gap:28px; padding-left:22px; padding-right:22px; }
      .card { width:108px; flex-basis:108px; height:164px; margin-left:-22px; }
      .hand-zone { grid-template-columns:68px minmax(0,1fr) 88px; padding-left:9px; padding-right:9px; }
      .energy { width:58px; height:58px; font-size:20px; }
    }
    @media (max-width:900px) {
      .topbar { grid-template-columns:1fr auto; }
      .top-stats { display:none; }
      .battle-layout { grid-template-columns:180px minmax(0,1fr) 205px; }
      .side { padding:8px 6px; }
      .stage { grid-template-columns:145px minmax(220px,1fr); gap:18px; padding-left:12px; padding-right:12px; }
      .entity-card { padding-left:7px; padding-right:7px; }
      .enemies .entity { width:148px; flex-basis:148px; }
      .enemies .entity-card { max-width:148px; }
      .card { width:96px; flex-basis:96px; margin-left:-20px; }
    }
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand"><div class="sigil"></div><div class="brand-copy"><strong>RUN SCRYER</strong><span>STS2 Training visualizer</span></div></div>
    <div class="top-stats">
      <span class="top-stat">♥ <b id="top-hp">—/—</b></span>
      <span class="top-stat">● <b id="top-gold">—</b></span>
      <span class="top-stat">▟ <b id="top-floor">—</b></span>
    </div>
    <div class="status-wrap"><span id="mode-pill" class="pill">MODE</span><span id="state-pill" class="pill">IDLE</span></div>
  </header>

  <div class="battle-layout">
    <aside class="side">
      <div class="panel"><h2>Run</h2><div class="kv" id="run-kv"></div></div>
      <div class="panel"><h2>Piles</h2><div class="kv" id="piles-kv"></div></div>
      <div class="panel"><h2>Decision</h2><div class="kv" id="decision-kv"></div></div>
    </aside>

    <main class="arena">
      <section class="stage">
        <div id="boundary-badge" class="boundary-badge">BOUNDARY —</div>
        <div class="entity" id="player"></div>
        <div class="enemies" id="enemies"></div>
      </section>
      <section class="hand-zone">
        <div class="energy-wrap"><div class="energy" id="energy">—</div><div class="energy-label">ENERGY</div></div>
        <div class="hand" id="hand"></div>
        <button class="end-turn" disabled>End Turn</button>
      </section>
    </main>

    <aside class="side right">
      <div class="panel"><h2>Selected Action</h2><div id="action-detail" class="muted">No event selected</div></div>
      <div class="panel"><h2>Timeline</h2><div class="timeline" id="timeline"></div></div>
      <div class="panel"><details><summary>Raw event JSON</summary><pre id="raw-json">{}</pre></details></div>
    </aside>
  </div>

  <footer class="transport">
    <div class="controls"><button id="start-run" class="start-run">START RUN</button><button id="prev" class="ctrl">◀</button><button id="play" class="ctrl">▶</button><button id="next" class="ctrl">▶|</button><select id="speed" class="ctrl"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div>
    <input id="scrubber" class="scrubber" type="range" min="0" max="0" value="0">
    <div class="frame-label" id="frame-label">0 / 0</div>
  </footer>
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

  function renderPowers(powers) {
    const values = Array.isArray(powers) ? powers : [];
    if (!values.length) return '';
    const visible = values.slice(0,8).map(power => {
      const name = text(power?.name || power?.id, 'Power');
      const rune = esc(name.trim().slice(0,1).toUpperCase() || '✦');
      const amount = power?.amount === undefined || power?.amount === null || power?.amount === '' ? '' : `<span class="power-amount">${esc(power.amount)}</span>`;
      const title = power?.description ? ` title="${esc(compact(power.description,180))}"` : '';
      return `<span class="power"${title}><span class="power-rune">${rune}</span><span class="power-name">${esc(compact(name,28))}</span>${amount}</span>`;
    }).join('');
    const more = values.length > 8 ? `<span class="power more">+${values.length-8}</span>` : '';
    return `<div class="power-strip">${visible}${more}</div>`;
  }

  function renderEntity(target, entity, kind, index=0) {
    const value = entity || {}, hp=value.current_hp, maxHp=value.max_hp ?? hp;
    const statuses = Array.isArray(value.statuses) ? value.statuses : [];
    const tags = [value.block ? `<span class="chip block">◇ ${esc(value.block)}</span>` : '', ...statuses.slice(0,5).map(s => `<span class="chip">${esc(compact(s,22))}</span>`)].join('');
    const icon = kind === 'player' ? '♙' : ['♞','♜','♝'][index % 3];
    const intent = value.intent !== undefined && value.intent !== null ? `<div class="intent">⚔ ${esc(compact(value.intent,18))}</div>` : '';
    target.innerHTML = `${intent}<div class="entity-card"><div class="portrait">${icon}</div><div class="entity-name">${esc(value.name || (kind==='player'?'Player':`Enemy ${index+1}`))}</div><div class="hpbar"><div class="hpfill" style="width:${pct(hp,maxHp)}%"></div></div><div class="hptext"><span>${esc(text(hp))} / ${esc(text(maxHp))}</span><span>${value.block ? `BLK ${esc(value.block)}` : ''}</span></div><div class="combat-tags">${tags}</div>${renderPowers(value.powers)}</div>`;
  }

  function renderBattle(event) {
    const frame = event?.frame || {}, player = frame.player || {}, resources = frame.resources || {}, piles = frame.piles || {};
    renderEntity($('player'), player, 'player');
    $('enemies').innerHTML='';
    const enemies = Array.isArray(frame.enemies) ? frame.enemies : [];
    if (!enemies.length) $('enemies').innerHTML='<div class="empty">No enemy state in this frame</div>';
    enemies.forEach((enemy,i) => { const node=document.createElement('div'); node.className='entity'; renderEntity(node,enemy,'enemy',i); $('enemies').appendChild(node); });
    $('boundary-badge').textContent=`BOUNDARY ${text(frame.boundary)}`;
    $('top-hp').textContent=`${text(player.current_hp)}/${text(player.max_hp)}`;
    $('top-gold').textContent=text(resources.gold);
    $('top-floor').textContent=text(resources.floor);
    $('energy').textContent = resources.max_energy === undefined || resources.max_energy === null ? text(resources.energy) : `${text(resources.energy)}/${text(resources.max_energy)}`;
    renderHand(Array.isArray(frame.hand)?frame.hand:[], event);
    setKv('run-kv', [['boundary',frame.boundary],['character',resources.character],['floor',resources.floor],['outcome',frame.outcome],['event count',state.events.length]]);
    setKv('piles-kv', [['draw',piles.draw],['discard',piles.discard],['exhaust',piles.exhaust],['deck',piles.deck]]);
  }

  function renderHand(cards,event) {
    const hand=$('hand'); hand.innerHTML='';
    if (!cards.length) { hand.innerHTML='<div class="empty">No card/action entries</div>'; return; }
    const spread=Math.min(7,38/Math.max(cards.length-1,1));
    cards.forEach((card,i) => {
      const node=document.createElement('div'), selected=event?.selected_action_id && card.action_id===event.selected_action_id;
      node.className=`card ${card.is_available!==false?'available':''} ${selected?'selected':''} ${event?.phase==='beam_explore'?'speculative':''}`;
      const angle=(i-(cards.length-1)/2)*spread, lift=Math.abs(i-(cards.length-1)/2)*1.6;
      node.style.transform=`rotate(${angle}deg) translateY(${lift}px)`; node.style.zIndex=String(i+1);
      node.innerHTML=`<div class="card-cost">${esc(card.cost)}</div><div class="card-name">${esc(card.name || card.action_type || card.action_id)}</div><div class="card-art">✦</div><div class="card-text">${esc(compact(card.description,120))}</div>`;
      hand.appendChild(node);
    });
  }

  function setKv(id,pairs) { $(id).innerHTML=pairs.map(([k,v])=>`<div class="k">${esc(k)}</div><div class="v">${esc(compact(v,90))}</div>`).join(''); }
  function renderMeta(event) {
    setKv('decision-kv',[['event',event?.event],['operation',event?.operation],['branch',event?.branch_id],['decision',event?.decision_point_id],['phase',event?.phase],['frame source',event?.frame_source],['logged',event?.logged_at]]);
    const action=event?.selected_action;
    $('action-detail').innerHTML=action ? `<div class="action-title">${esc(action.name || action.action_id || 'Action')}</div><div class="kv">${Object.entries(action).filter(([key])=>key!=='name').map(([k,v])=>`<div class="k">${esc(k)}</div><div class="v">${esc(compact(v,110))}</div>`).join('')}</div>` : '<span class="muted">No selected action in this event</span>';
    $('raw-json').textContent=JSON.stringify(event?.raw || {},null,2);
  }
  function renderTimeline() {
    const timeline=$('timeline'); timeline.innerHTML=''; const start=Math.max(0,state.events.length-80);
    state.events.slice(start).forEach((event,offset)=>{
      const i=start+offset,button=document.createElement('button'),cls=event.operation==='commit_action'?'commit':(event.phase==='beam_explore'?'branch':'');
      button.className=`tick ${i===state.frame?'active':''}`;
      button.innerHTML=`#${i+1} <span class="${cls}">${esc(event.operation || event.event)}</span> ${esc(compact(event.selected_action_id || event.frame?.outcome || event.branch_id || '',26))}`;
      button.onclick=()=>{state.playing=false;state.followLive=false;setFrame(i)}; timeline.appendChild(button);
    });
    if(state.followLive)timeline.scrollTop=timeline.scrollHeight;
  }
  function setFrame(index) {
    if(!state.events.length){state.frame=-1;$('frame-label').textContent='0 / 0';return}
    state.frame=Math.max(0,Math.min(index,state.events.length-1)); const event=state.events[state.frame];
    renderBattle(event); renderMeta(event); renderTimeline();
    $('scrubber').max=String(Math.max(0,state.events.length-1)); $('scrubber').value=String(state.frame); $('frame-label').textContent=`${state.frame+1} / ${state.events.length}`;
  }
  function appendEvents(events) {
    if(!Array.isArray(events)||!events.length)return;
    events.forEach(event=>{state.events.push(event);state.cursor=Math.max(state.cursor,Number(event.index))});
    if(state.mode==='live'&&state.followLive)setFrame(state.events.length-1); else if(state.frame<0)setFrame(0); else { $('scrubber').max=String(state.events.length-1); renderTimeline(); }
  }
  function playTick(){ clearTimeout(state.timer); if(!state.playing||!state.events.length)return; if(state.frame>=state.events.length-1){ if(state.mode==='live'&&state.lifecycle==='running'){state.timer=setTimeout(playTick,300);return} state.playing=false;$('play').textContent='▶';return } setFrame(state.frame+1); state.timer=setTimeout(playTick,Math.max(80,760/state.speed)); }
  function toast(message){const node=$('toast');node.textContent=message;node.classList.add('show');setTimeout(()=>node.classList.remove('show'),4200)}
  async function jsonFetch(url,options){const response=await fetch(url,options);const body=await response.json().catch(()=>({}));if(!response.ok)throw new Error(body.error||`${response.status} ${response.statusText}`);return body}
  async function refreshStatus(){try{const data=await jsonFetch('/api/status');state.mode=data.mode;state.lifecycle=data.state;$('mode-pill').textContent=data.mode.toUpperCase();$('state-pill').textContent=String(data.state||'ready').toUpperCase();$('state-pill').className=`pill ${data.state||''}`;$('start-run').style.display=data.mode==='live'?'':'none';$('start-run').disabled=data.mode!=='live'||data.state!=='idle';if(data.error)toast(data.error)}catch(error){toast(error.message)}}
  async function pollEvents(){try{const data=await jsonFetch(`/api/events?after=${state.cursor}`);appendEvents(data.events||[])}catch(error){toast(error.message)}if(state.mode==='live')setTimeout(pollEvents,350)}
  $('start-run').onclick=async()=>{try{await jsonFetch('/api/live/start',{method:'POST'});state.followLive=true;await refreshStatus()}catch(error){toast(error.message)}};
  $('play').onclick=()=>{state.playing=!state.playing;state.followLive=false;$('play').textContent=state.playing?'Ⅱ':'▶';playTick()};
  $('prev').onclick=()=>{state.playing=false;state.followLive=false;$('play').textContent='▶';setFrame(state.frame-1)};
  $('next').onclick=()=>{state.playing=false;state.followLive=false;$('play').textContent='▶';setFrame(state.frame+1)};
  $('speed').onchange=e=>{state.speed=Number(e.target.value)||1};
  $('scrubber').oninput=e=>{state.playing=false;state.followLive=false;setFrame(Number(e.target.value))};
  (async()=>{await refreshStatus();const data=await jsonFetch('/api/events?after=-1').catch(error=>{toast(error.message);return{events:[]}});appendEvents(data.events||[]);if(state.mode==='live')setTimeout(pollEvents,350);setInterval(refreshStatus,900)})();
})();
</script>
</body>
</html>'''

__all__ = ["INDEX_HTML"]
