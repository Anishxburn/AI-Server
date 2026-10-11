(() => {
  const byId = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const history = [];
  let busy = false;

  async function api(path, payload) {
    const response = await fetch('/debug/sandbox' + path, payload === undefined ? {} : {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload),
    });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); }
    catch {
      const detail = raw.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 180);
      throw new Error(`Sandbox API returned HTTP ${response.status} with a non-JSON response${detail ? `: ${detail}` : '.'}`);
    }
    if (!response.ok) throw new Error(data.error || 'Sandbox request failed.');
    return data;
  }
  async function submitAndWait(payload) {
    const accepted = await api('', payload);
    if (!accepted.job_id) return accepted;
    for (let attempt = 0; attempt < 600; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const state = await api(`/jobs/${encodeURIComponent(accepted.job_id)}`);
      if (state.status === 'completed') return state.result;
      if (state.status === 'failed') throw new Error(state.error || 'Sandbox model generation failed.');
      setStatus(state.phase || (state.status === 'queued' ? 'Queued for the model...' : 'The model is working...'));
    }
    throw new Error('The model is still working. Check Traces for the sandbox job status.');
  }
  function setStatus(message, error = false) {
    byId('sandbox-status').textContent = message;
    byId('sandbox-status').dataset.error = String(error);
  }
  function setBusy(value) {
    busy = value;
    byId('sandbox-send').disabled = value;
    byId('sandbox-input').disabled = value;
    byId('sandbox-clear').disabled = value;
    document.querySelectorAll('.sandbox-prompt').forEach(button => { button.disabled = value; });
    byId('sandbox-status').setAttribute('aria-busy', String(value));
  }
  function appendMessage(role, content, model = '') {
    const intro = byId('sandbox-chat').querySelector('.sandbox-intro');
    if (intro) intro.remove();
    const node = document.createElement('div');
    node.className = `sandbox-message ${role}`;
    node.innerHTML = `<small>${role === 'user' ? 'You' : `AI sandbox${model ? ` · ${esc(model)}` : ''}`}</small>${esc(content)}`;
    byId('sandbox-chat').appendChild(node);
    byId('sandbox-chat').scrollTop = byId('sandbox-chat').scrollHeight;
  }
  function renderOutput(result) {
    const charts = (result.charts || []).map((chart, index) => `<section><h3>${esc(chart.title)}</h3>
      <div class="sandbox-chart"><canvas id="sandbox-chart-${index}" aria-label="${esc(chart.title)}" role="img"></canvas></div>
      <details><summary>Chart specification</summary><pre>${esc(JSON.stringify(chart, null, 2))}</pre></details></section>`).join('');
    const report = result.report ? `<section><h3>Report preview</h3><pre>${esc(JSON.stringify(result.report, null, 2))}</pre></section>` : '';
    const evidence = `<details><summary>Data decision and evidence</summary><pre>${esc(JSON.stringify({data_accessed:result.data_accessed,data_decision:result.data_decision,dataset:result.dataset,evidence:result.evidence,formulas:result.formula_reference,mcp_called:result.mcp_called}, null, 2))}</pre></details>`;
    const root = byId('sandbox-output');
    root.hidden = false;
    root.innerHTML = (charts || report ? `<h3>Generated output</h3>${charts}${report}` : '') + evidence;
    (result.charts || []).forEach((chart, index) => {
      const canvas = byId(`sandbox-chart-${index}`);
      if (canvas && window.renderDashboardChart) window.renderDashboardChart(canvas, chart);
    });
  }
  async function send(message) {
    if (busy || !message.trim()) return;
    setBusy(true);
    byId('sandbox-output').hidden = true;
    appendMessage('user', message);
    history.push({role:'user', content:message});
    setStatus('Asking the LLM whether this question needs sample readings...');
    try {
      const result = await submitAndWait({message, history:history.slice(0, -1).slice(-12)});
      appendMessage('assistant', result.answer, result.model);
      history.push({role:'assistant', content:result.answer});
      renderOutput(result);
      setStatus(result.data_accessed
        ? `Completed with ${result.model}. Sample data used: ${result.dataset.categories_loaded.join(', ')}. MCP called: no.`
        : `Completed with ${result.model}. No sample readings were loaded. MCP called: no.`);
    } catch (error) {
      appendMessage('assistant', error.message);
      history.push({role:'assistant', content:error.message});
      setStatus(error.message, true);
    } finally {
      setBusy(false);
      byId('sandbox-input').focus();
    }
  }
  byId('sandbox-form').addEventListener('submit', event => {
    event.preventDefault();
    const input = byId('sandbox-input');
    const message = input.value.trim();
    input.value = '';
    send(message);
  });
  byId('sandbox-input').addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); byId('sandbox-form').requestSubmit(); }
  });
  byId('sandbox-clear').addEventListener('click', () => {
    history.length = 0;
    byId('sandbox-chat').innerHTML = '<div class="sandbox-intro">Ask a question about the sample EMS data, or ask the model a general question. Follow-up questions use this chat\'s recent messages.</div>';
    byId('sandbox-output').hidden = true;
    byId('sandbox-output').innerHTML = '';
    setStatus('Sandbox conversation cleared.');
  });
  document.querySelectorAll('.sandbox-prompt').forEach(button => button.addEventListener('click', () => send(button.dataset.prompt || '')));

  window.SandboxChat = {
    async refresh() {
      try {
        const data = await api('', undefined);
        const runtime = data.ollama || {};
        const runtimeLabel = runtime.status === 'online'
          ? `Ollama online Â· ${runtime.model_installed ? 'chat model available' : 'configured model not installed'} (${runtime.model})`
          : `Ollama offline Â· ${runtime.error || 'cannot connect'}`;
        byId('sandbox-runtime').textContent = runtimeLabel;
        byId('sandbox-runtime').dataset.error = String(runtime.status !== 'online' || !runtime.model_installed);
        byId('sandbox-dataset').innerHTML = `<strong>${esc(data.dataset.dataset_name)}</strong><br>${esc(data.dataset.site.name)} / Site ${esc(data.dataset.site.site_id)}<br><em>Synthetic only. Readings enter a chat prompt only when relevant.</em>`;
        /*
        if (false) byId('sandbox-facts').innerHTML = [
          ['7-day energy', `${Number(data.dataset.daily_energy.reduce((sum,row) => sum + row.value, 0)).toLocaleString()} kWh`],
          ['Peak demand', `${peak.value} kW`],
          ['Highest consumer', `${esc(top.name)} · ${Number(top.energy_kwh).toLocaleString()} kWh`],
          ['Model', esc(data.model)],
        ].map(([label,value]) => `<div class="sandbox-fact"><span>${label}</span><strong>${value}</strong></div>`).join('');
        */
        byId('sandbox-facts').innerHTML = `<div class="sandbox-fact"><span>Available categories</span><strong>${data.dataset.available_categories.map(esc).join(', ')}</strong></div><div class="sandbox-fact"><span>Model</span><strong>${esc(data.model)}</strong></div>`;
        byId('sandbox-formulas').innerHTML = data.formulas.map(item => `<div class="sandbox-formula"><strong>${esc(item.name)}</strong><code>${esc(item.formula)}</code>${esc(item.needs)}</div>`).join('');
      } catch (error) { setStatus(error.message, true); }
    },
  };
})();
