from __future__ import annotations

from sts2_training.visualizer.assets import INDEX_HTML as _BASE_INDEX_HTML


def _replace_once(html: str, old: str, new: str, *, label: str) -> str:
    count = html.count(old)
    if count != 1:
        raise RuntimeError(f"visualizer HTML patch {label!r} expected one match, found {count}")
    return html.replace(old, new, 1)


def build_index_html() -> str:
    """Apply focused UI changes without duplicating the large embedded HTML asset."""
    html = _BASE_INDEX_HTML

    html = _replace_once(
        html,
        """    .enemies { position:relative; z-index:2; display:flex; gap:14px; align-items:end; justify-content:flex-end; min-width:0; overflow-x:auto; padding:30px 2px 0; scrollbar-width:thin; }
    .enemies .entity { width:164px; flex:0 0 164px; }
    .enemies .entity-card { max-width:164px; }
    .enemies .portrait { width:86px; height:98px; color:rgba(211,125,96,.72); }
""",
        """    .enemies { position:relative; z-index:2; display:grid; grid-template-columns:repeat(var(--enemy-count,1),minmax(0,1fr)); gap:8px; align-items:stretch; justify-content:stretch; min-width:0; overflow:hidden; padding:18px 2px 0; }
    .enemies .entity { width:auto; min-width:0; text-align:left; }
    .enemies .entity-card { width:100%; max-width:none; min-width:0; min-height:154px; height:100%; padding:14px 12px; }
    .enemy-card .entity-name { margin-bottom:12px; color:#f1e5bd; font-size:16px; font-weight:700; white-space:normal; overflow-wrap:anywhere; }
    .enemy-facts { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; margin-bottom:10px; }
    .enemy-fact { padding:7px 7px; border:1px solid rgba(174,199,190,.18); background:rgba(0,0,0,.22); min-width:0; }
    .enemy-fact span,.enemy-section-label { display:block; margin-bottom:3px; color:#7f918e; font:8px var(--mono); letter-spacing:.08em; overflow-wrap:anywhere; }
    .enemy-fact strong { display:block; min-width:0; color:#f0d6c6; font:700 11px/1.35 var(--mono); overflow-wrap:anywhere; }
    .enemy-card .power-strip { justify-content:flex-start; min-height:0; margin-top:0; }
    .enemy-card .power { min-width:0; font-size:9px; }
    .enemy-none { color:#697a76; font:10px var(--mono); }
    .enemies.many-enemies { gap:5px; }
    .enemies.many-enemies .entity-card { padding:10px 7px; }
    .enemies.many-enemies .enemy-card .entity-name { margin-bottom:7px; font-size:12px; }
    .enemies.many-enemies .enemy-facts { grid-template-columns:1fr; gap:3px; margin-bottom:7px; }
    .enemies.many-enemies .enemy-fact { padding:3px 5px; display:grid; grid-template-columns:minmax(46px,auto) minmax(0,1fr); gap:5px; align-items:center; }
    .enemies.many-enemies .enemy-fact span { margin-bottom:0; }
    .enemies.many-enemies .enemy-fact strong { text-align:right; font-size:10px; }
    .enemies.many-enemies .enemy-card .power { font-size:8px; }
""",
        label="enemy layout",
    )

    html = _replace_once(
        html,
        """    const tags = [value.block ? `<span class=\"chip block\">◇ BLOCK ${esc(value.block)}</span>` : '', ...statuses.slice(0,5).map(s => `<span class=\"chip\">${esc(compact(s,22))}</span>`)].join('');
""",
        """    const blockTag = value.block === undefined || value.block === null || value.block === '' ? '' : `<span class=\"chip block\">◇ BLOCK ${esc(value.block)}</span>`;
    const tags = [blockTag, ...statuses.slice(0,5).map(s => `<span class=\"chip\">${esc(compact(s,22))}</span>`)].join('');
""",
        label="player block tag",
    )

    html = _replace_once(
        html,
        """    target.innerHTML = `${intent}<div class=\"entity-card\"><div class=\"portrait\">${icon}</div><div class=\"entity-name\">${esc(value.name || (kind==='player'?'Player':`Enemy ${index+1}`))}</div><div class=\"hpbar\"><div class=\"hpfill\" style=\"width:${pct(hp,maxHp)}%\"></div></div><div class=\"hptext\">${esc(hpText(value))}</div><div class=\"combat-tags\">${tags}</div>${renderPowers(value.powers)}</div>`;
""",
        """    if (kind === 'enemy') {
      const intentMove = value.intent === undefined || value.intent === null || value.intent === '' ? '—' : compact(value.intent,40);
      const block = value.block === undefined || value.block === null || value.block === '' ? '0' : value.block;
      const powers = renderPowers(value.powers) || '<div class=\"enemy-none\">—</div>';
      target.innerHTML = `<div class=\"entity-card enemy-card\"><div class=\"entity-name\">${esc(value.name || 'Enemy')}</div><div class=\"enemy-facts\"><div class=\"enemy-fact\"><span>HP</span><strong>${esc(hpText(value))}</strong></div><div class=\"enemy-fact\"><span>INTENT MOVE</span><strong>${esc(intentMove)}</strong></div><div class=\"enemy-fact\"><span>BLOCK</span><strong>${esc(block)}</strong></div></div><span class=\"enemy-section-label\">POWER</span>${powers}</div>`;
      return;
    }
    target.innerHTML = `${intent}<div class=\"entity-card\"><div class=\"portrait\">${icon}</div><div class=\"entity-name\">${esc(value.name || 'Player')}</div><div class=\"hpbar\"><div class=\"hpfill\" style=\"width:${pct(hp,maxHp)}%\"></div></div><div class=\"hptext\">${esc(hpText(value))}</div><div class=\"combat-tags\">${tags}</div>${renderPowers(value.powers)}</div>`;
""",
        label="enemy renderer",
    )

    html = _replace_once(
        html,
        """    const enemies = Array.isArray(frame.enemies) ? frame.enemies : [];
    if (!enemies.length) $('enemies').innerHTML='<div class=\"empty\">No enemy state in this frame</div>';
""",
        """    const enemies = Array.isArray(frame.enemies) ? frame.enemies : [];
    $('enemies').style.setProperty('--enemy-count', String(Math.max(1,enemies.length)));
    $('enemies').classList.toggle('many-enemies', enemies.length >= 3);
    if (!enemies.length) $('enemies').innerHTML='<div class=\"empty\">No enemy state in this frame</div>';
""",
        label="enemy count layout",
    )

    html = _replace_once(
        html,
        """      const type=choice.action_type ? `<span class=\"choice-type\">${esc(choice.action_type)}</span>` : '';
      const cost=choice.cost === undefined || choice.cost === null || choice.cost === '' ? '' : `<span class=\"choice-cost\">COST ${esc(choice.cost)}</span>`;
      const summary=choice.summary || choice.description || '';
      const details=(Array.isArray(choice.details)?choice.details:[]).slice(0,3).map(detail=>`<span class=\"choice-detail\">${esc(detail.label)}: ${esc(detail.value)}</span>`).join('');
      node.innerHTML=`<span class=\"choice-index\">${index+1}</span><div class=\"choice-name\">${esc(choice.name || choice.action_id || `Choice ${index+1}`)}</div><div class=\"choice-meta\">${type}${cost}</div>${summary?`<div class=\"choice-summary\">${esc(compact(summary,100))}</div>`:''}${details?`<div class=\"choice-details\">${details}</div>`:''}`;
""",
        """      const cost=choice.cost === undefined || choice.cost === null || choice.cost === '' ? '' : `<span class=\"choice-cost\">COST ${esc(choice.cost)}</span>`;
      const actionType=String(choice.action_type || '').toLowerCase();
      const isCard=Boolean(choice.card_id) || actionType.includes('card');
      const label=isCard ? (choice.card_id || choice.name || choice.action_id || `Choice ${index+1}`) : (choice.name || choice.summary || choice.action_id || `Choice ${index+1}`);
      if (isCard) {
        node.innerHTML=`<span class=\"choice-index\">${index+1}</span><div class=\"choice-name\">${esc(label)}</div>${cost?`<div class=\"choice-meta\">${cost}</div>`:''}`;
      } else {
        const summary=choice.summary && choice.summary !== label ? choice.summary : '';
        const details=(Array.isArray(choice.details)?choice.details:[]).slice(0,3).map(detail=>`<span class=\"choice-detail\">${esc(detail.label)}: ${esc(detail.value)}</span>`).join('');
        node.innerHTML=`<span class=\"choice-index\">${index+1}</span><div class=\"choice-name\">${esc(label)}</div>${summary?`<div class=\"choice-summary\">${esc(compact(summary,100))}</div>`:''}${details?`<div class=\"choice-details\">${details}</div>`:''}`;
      }
""",
        label="choice renderer",
    )

    html = _replace_once(
        html,
        "$('start-run').disabled=data.mode!=='live'||data.state!=='idle';",
        "$('start-run').disabled=data.mode!=='live'||!['idle','completed','failed'].includes(data.state);",
        label="restart button state",
    )
    html = _replace_once(
        html,
        "$('start-run').onclick=async()=>{try{await jsonFetch('/api/live/start',{method:'POST'});state.followLive=true;await refreshStatus()}catch(error){toast(error.message)}};",
        "$('start-run').onclick=async()=>{try{await jsonFetch('/api/live/start',{method:'POST'});state.events=[];state.cursor=-1;state.frame=-1;state.playing=false;state.followLive=true;$('timeline').innerHTML='';$('choices').innerHTML='';$('hand').innerHTML='';$('enemies').innerHTML='';$('player').innerHTML='';$('frame-label').textContent='0 / 0';$('scrubber').max='0';$('scrubber').value='0';await refreshStatus()}catch(error){toast(error.message)}};",
        label="restart client state",
    )
    return html


INDEX_HTML = build_index_html()

__all__ = ["INDEX_HTML", "build_index_html"]
