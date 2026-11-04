const $ = (id) => document.getElementById(id);
const escapeHTML = (text) => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data, run, stepIndex = 0, timer = null, peaksVisible = false;
const selected = new Set(), solved = new Set();
let revealed = false;

function pairs(groups) {
  const result = new Set();
  for (const group of groups) for (let i=0;i<group.length;i++) for(let j=i+1;j<group.length;j++) result.add([group[i],group[j]].sort().join('|'));
  return result;
}
function quality(partition) {
  const predicted = pairs(partition), gold = pairs(data.puzzle.groups.map(g => g.words));
  const correct = [...predicted].filter(p => gold.has(p)).length;
  return predicted.size ? 2*correct/(predicted.size+gold.size) : 0;
}
function groupStatus(group) {
  if (data.puzzle.groups.some(g => g.words.length === group.length && group.every(w=>g.words.includes(w)))) return 'exact';
  if (data.puzzle.groups.some(g => group.every(w=>g.words.includes(w)))) return 'partial';
  return 'mixed';
}
function chartMarkup(values, current, width, height, compact=false) {
  const pad = compact ? {l:9,r:12,t:14,b:19} : {l:31,r:16,t:17,b:29};
  const x=i=>pad.l+i/(values.length-1)*(width-pad.l-pad.r);
  const y=v=>pad.t+(1-v)*(height-pad.t-pad.b);
  const line=values.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
  const area=`${line} L${x(values.length-1)},${y(0)} L${x(0)},${y(0)} Z`;
  const grid=[0,.5,1].map(v=>`<line class="chart-grid" x1="${pad.l}" x2="${width-pad.r}" y1="${y(v)}" y2="${y(v)}"/>${compact?'':`<text class="chart-label" x="${pad.l-8}" y="${y(v)+3}" text-anchor="end">${v.toFixed(1)}</text>`}`).join('');
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${compact?'Illustrative':'Constructed example'} partition quality trajectory. Current quality ${values[current].toFixed(3)}. Horizontal axis: ordered trajectory states; vertical axis: pairwise F1 from zero to one."><title>Partition quality over trajectory states</title>${grid}<path class="chart-area" d="${area}"/><path class="chart-line" d="${line}"/><line class="chart-cursor" x1="${x(current)}" x2="${x(current)}" y1="${pad.t}" y2="${y(0)}"/><circle class="chart-dot" cx="${x(current)}" cy="${y(values[current])}" r="5"/>${compact?`<text class="chart-label" x="${x(3)}" y="${y(1)-5}" text-anchor="middle">best state</text><text class="chart-label" x="${width-pad.r}" y="${height-3}" text-anchor="end">final answer</text>`:`<text class="chart-label" x="${pad.l}" y="${height-7}">START</text><text class="chart-label" x="${width-pad.r}" y="${height-7}" text-anchor="end">FINAL STATE</text>`}</svg>`;
}
function renderCharts() {
  if (!run) return;
  const box=$('trajectory-chart');
  box.innerHTML=chartMarkup(run.steps.map(s=>quality(s.partition)),stepIndex,Math.max(220,box.clientWidth),box.clientHeight);
  const hero=$('hero-chart'), demo=data.runs.find(r=>r.id==='regression');
  hero.innerHTML=chartMarkup(demo.steps.map(s=>quality(s.partition)),demo.steps.length-1,Math.max(230,hero.clientWidth),105,true);
}
function stop() {
  if(timer) clearInterval(timer);
  timer=null; $('play').textContent='▶'; $('play').setAttribute('aria-label','Play trajectory');
}
function renderRun() {
  const state=run.steps[stepIndex], values=run.steps.map(s=>quality(s.partition)), peak=Math.max(...values);
  $('run-type').textContent=run.regime;
  $('run-title').textContent=run.title;
  $('run-id').textContent=`DEMO / ${run.id.toUpperCase()}`;
  $('state-position').textContent=`STATE ${String(stepIndex+1).padStart(2,'0')} / ${run.steps.length}`;
  const assigned=new Set(state.partition.flat());
  const rows=state.partition.map(group=>({group,status:groupStatus(group)}));
  const remaining=data.puzzle.words.filter(w=>!assigned.has(w));
  for(let i=0;i<remaining.length;i+=4) rows.push({group:remaining.slice(i,i+4),status:'unassigned'});
  $('partition-board').innerHTML=rows.map(({group,status})=>`<div class="partition-row ${status}" style="grid-template-columns:repeat(${Math.max(4,group.length)},1fr) 40px" aria-label="${status} group: ${escapeHTML(group.join(', '))}">${group.map(w=>`<span class="partition-word">${escapeHTML(w)}</span>`).join('')}${Array.from({length:Math.max(0,4-group.length)},()=>'<span class="partition-word empty"></span>').join('')}<span class="row-status">${{exact:'4/4',partial:`${group.length}/4`,mixed:'MIX',unassigned:'—'}[status]}</span></div>`).join('');
  $('current-quality').textContent=values[stepIndex].toFixed(3);
  $('peak-quality').textContent=peak.toFixed(3);
  $('regret-quality').textContent=(peak-values.at(-1)).toFixed(3);
  $('step-slider').max=run.steps.length-1;
  $('step-slider').value=stepIndex;
  $('step-slider').setAttribute('aria-valuetext',`State ${stepIndex+1}: ${state.title}`);
  $('step-readout').textContent=`${String(stepIndex+1).padStart(2,'0')} / ${String(run.steps.length).padStart(2,'0')}`;
  $('event-type').textContent=state.kind==='submission'?'OFFICIAL SUBMISSION':values[stepIndex]===peak?'BEST STATE':'REASONING STATE';
  $('event-title').textContent=state.title;
  $('event-description').textContent=state.description;
  $('action-detail').replaceChildren();
  if(state.action){
    const line=document.createElement('span'); line.textContent=`GUESS ${state.action_key}: ${state.action.join(' · ')}`;
    const status=document.createElement('span');status.className='feedback-tag';status.textContent=state.feedback;
    $('action-detail').append(line,status);
  }
  $('history-panel').hidden=!run.steps.some(s=>s.action);
  const history=run.steps.slice(0,stepIndex+1).filter(s=>s.action);
  $('action-history').innerHTML=history.length?history.map(s=>`<div class="history-action ${s.feedback==='REPEATED'?'repeated':''}"><strong>GUESS ${escapeHTML(s.action_key)}</strong> / ${escapeHTML(s.feedback)}<br>${s.action.map(escapeHTML).join(' · ')}</div>`).join(''):'<span class="fineprint">No official guesses submitted yet.</span>';
  renderCharts();
}
function chooseRun(id) {
  stop();run=data.runs.find(r=>r.id===id);stepIndex=0;
  document.querySelectorAll('[data-run]').forEach(button=>{const active=button.dataset.run===id;button.setAttribute('aria-selected',active);button.tabIndex=active?0:-1;});
  $('replay-panel').setAttribute('aria-labelledby',`tab-${id}`);renderRun();
}
function renderPuzzle() {
  const words=revealed?data.puzzle.groups.flatMap(g=>g.words):data.puzzle.words;
  $('puzzle-board').replaceChildren();
  for(const word of words){
    const index=data.puzzle.groups.findIndex(g=>g.words.includes(word));
    const locked=solved.has(index)||revealed;
    const b=document.createElement('button');b.type='button';b.className=`word-tile${locked?` solved group-${index}`:''}`;
    b.textContent=word;b.setAttribute('aria-pressed',selected.has(word));b.disabled=locked;
    if(locked)b.setAttribute('aria-label',`${word}, ${data.puzzle.groups[index].category}`);
    b.addEventListener('click',()=>{if(selected.has(word)) selected.delete(word);else if(selected.size<4)selected.add(word);renderPuzzle();});
    $('puzzle-board').append(b);
  }
  $('puzzle-submit').disabled=selected.size!==4||revealed;
  $('puzzle-reveal').textContent=revealed||solved.size===4?'Reset puzzle ↻':'Reveal groups ↗';
}
function renderComparison() {
  $('comparison-chart').innerHTML=data.comparison.models.map(m=>`<div class="compare-row"><div class="compare-name"><span>${escapeHTML(m.name)}</span><span class="compare-value">${m.final.toFixed(3)}${peaksVisible?` → ${(m.final+m.regret).toFixed(3)}`:''}</span></div><div class="compare-track" role="img" aria-label="${escapeHTML(m.name)}: mean final ${m.final.toFixed(3)}${peaksVisible?`, mean peak ${(m.final+m.regret).toFixed(3)}, mean regret ${m.regret.toFixed(3)}`:''}"><span class="compare-final" style="width:${m.final*100}%"></span><span class="compare-loss" style="width:${peaksVisible?m.regret*100:0}%"></span></div><div class="compare-regret">${peaksVisible?`MEAN REGRET ${m.regret.toFixed(3)}`:'FINAL PARTITION QUALITY'}</div></div>`).join('');
  $('reveal-peaks').innerHTML=peaksVisible?'Show final scores only <span aria-hidden="true">↶</span>':'Reveal what was lost <span aria-hidden="true">↗</span>';
  $('reveal-peaks').setAttribute('aria-pressed',peaksVisible);
  document.querySelector('.peak-legend').hidden=!peaksVisible;
  $('comparison-insight').textContent=peaksVisible?'Gemma retains almost all of its peak quality. Qwen3 and GPT-OSS reach higher peaks, then lose more before the final answer.':'The endpoints tell a remarkably similar story.';
}
const methodDetails=[
 ['DETERMINISTIC EXTRACTION','Lexical matching identifies spans in a chain-of-thought reasoning trace containing at least two puzzle words. This stage finds candidate word sets without deciding whether the model accepts them.'],
 ['AUTOMATED STANCE CLASSIFICATION','A classifier labels candidates as propose, refine_add, refine_pivot, reject, commit, enumerate, or mention. Only hypothesis-bearing labels update the partition. Human validation remains an important next step.'],
 ['DETERMINISTIC STATE UPDATES','Hypothesis-bearing groups update an accumulated partition with last-writer-wins reassignment. Parsed submissions anchor official action states. Different reconstruction policies must retain explicit provenance.'],
 ['TRAJECTORY-LEVEL ANALYSIS','M1–M4 describe revision, error propagation, structural drift, and discovery versus preservation. Multi-turn boundaries additionally expose reasoning gain, commitment loss, action change, and returns to earlier guesses.']
];
async function init() {
  const response=await fetch('data/showcase.json');if(!response.ok)throw new Error(`Data unavailable (${response.status})`);data=await response.json();
  $('hero-tiles').innerHTML=data.puzzle.groups.flatMap(g=>g.words).map(w=>`<span class="hero-tile">${escapeHTML(w)}</span>`).join('');
  renderPuzzle();renderComparison();chooseRun('regression');
  document.querySelectorAll('[data-run]').forEach((button,i,list)=>{
    button.addEventListener('click',()=>chooseRun(button.dataset.run));
    button.addEventListener('keydown',event=>{let next;if(event.key==='ArrowRight')next=(i+1)%list.length;else if(event.key==='ArrowLeft')next=(i+list.length-1)%list.length;else if(event.key==='Home')next=0;else if(event.key==='End')next=list.length-1;if(next!==undefined){event.preventDefault();chooseRun(list[next].dataset.run);list[next].focus();}});
  });
  $('step-slider').addEventListener('input',event=>{stop();stepIndex=Number(event.target.value);renderRun();});
  $('play').addEventListener('click',()=>{if(timer){stop();return;}if(stepIndex===run.steps.length-1)stepIndex=0;renderRun();$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','Pause trajectory');timer=setInterval(()=>{stepIndex++;renderRun();if(stepIndex===run.steps.length-1)stop();},1500);});
  $('jump-peak').addEventListener('click',()=>{stop();const values=run.steps.map(s=>quality(s.partition));stepIndex=values.indexOf(Math.max(...values));renderRun();});
  $('jump-action').addEventListener('click',()=>{stop();const next=run.steps.findIndex((s,i)=>i>stepIndex&&s.kind==='submission');stepIndex=next<0?run.steps.findIndex(s=>s.kind==='submission'):next;renderRun();});
  $('puzzle-submit').addEventListener('click',()=>{const index=data.puzzle.groups.findIndex(g=>g.words.every(w=>selected.has(w)));if(index>=0){solved.add(index);$('puzzle-message').textContent=solved.size===4?'All four connections found. Nicely done.':`Correct: ${data.puzzle.groups[index].category}.`;}else{const best=Math.max(...data.puzzle.groups.map(g=>g.words.filter(w=>selected.has(w)).length));$('puzzle-message').textContent=best===3?'One word away. Try a different combination.':'Not quite. Try a different connection.';}selected.clear();renderPuzzle();});
  $('puzzle-reveal').addEventListener('click',()=>{if(revealed||solved.size===4){revealed=false;solved.clear();$('puzzle-message').textContent='Find four words that belong together.';}else{revealed=true;$('puzzle-message').textContent=data.puzzle.groups.map(g=>g.category).join(' · ');}selected.clear();renderPuzzle();});
  $('reveal-peaks').addEventListener('click',()=>{peaksVisible=!peaksVisible;renderComparison();});
  document.querySelectorAll('[data-method]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-method]').forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});const detail=methodDetails[Number(button.dataset.method)];$('method-label').textContent=detail[0];$('method-description').textContent=detail[1];}));
  new ResizeObserver(renderCharts).observe($('trajectory-chart'));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
}
init().catch(error=>{console.error(error);$('run-title').textContent='The replay data could not be loaded.';$('event-description').textContent='Please reload the page to try again.';});
