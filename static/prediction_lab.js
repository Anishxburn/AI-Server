(() => {
  const byId = id => document.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const number = value => value === null || value === undefined ? 'Unavailable' : Number(value).toLocaleString(undefined, {maximumFractionDigits: 2});
  const charts = new Map();
  let dataset = null, currentRun = null, busy = false, initialized = false;
  const colors = ['#087f69', '#d57725', '#2277b5', '#bf4768', '#8b6ca8'];

  window.renderDashboardChart = (canvas, spec) => {
    charts.forEach((chart, node) => { if (!node.isConnected) { chart.destroy(); charts.delete(node); } });
    if (charts.has(canvas)) charts.get(canvas).destroy();
    canvas.style.height = '260px';
    canvas.height = 260;
    const type = spec.type === 'donut' ? 'doughnut' : spec.type;
    const datasets = spec.series.map((series, index) => ({
      label: series.name || series.label || 'Value', data: series.data,
      borderColor: colors[index % colors.length],
      backgroundColor: type === 'doughnut' ? colors : colors[index % colors.length] + '33',
      borderWidth: 2, pointRadius: spec.labels.length > 50 ? 0 : 2, spanGaps: false,
      borderDash: series.dashed ? [6, 4] : [], fill: false,
    }));
    const chart = new Chart(canvas, {type, data: {labels: spec.labels, datasets}, options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: {mode: 'index', intersect: false},
      plugins: {legend: {display: datasets.length > 1 || type === 'doughnut', position: 'bottom'}},
      scales: type === 'doughnut' ? {} : {
        x: {ticks: {maxTicksLimit: 8, maxRotation: 25}, grid: {display: false}},
        y: {beginAtZero: true, title: {display: true, text: spec.series[0]?.unit || 'kWh'}},
      },
    }});
    charts.set(canvas, chart);
  };
  window.addEventListener('resize', () => charts.forEach(chart => chart.resize()));

  async function api(path, payload) {
    const response = await fetch('/debug/predictions' + path, payload === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed.');
    return data;
  }
  function notice(message, error = false) {
    byId('lab-notice').hidden = !message;
    byId('lab-notice').textContent = message;
    byId('lab-notice').dataset.error = String(error);
  }
  function setBusy(value) {
    busy = value;
    document.querySelectorAll('#prediction-workspace button, #prediction-workspace select, #prediction-workspace input').forEach(control => { control.disabled = value; });
    byId('lab-run').disabled = value || !byId('lab-dataset').value;
    byId('lab-extract').disabled = value || !byId('lab-site').value;
    byId('lab-download').disabled = value || !currentRun;
  }
  async function action(fn) {
    if (busy) return;
    setBusy(true);
    try { await fn(); } catch (error) { notice(error.message, true); }
    finally { setBusy(false); }
  }
  async function job(path, payload) {
    const pending = await api(path, payload);
    for (let attempt = 0; attempt < 360; attempt++) {
      const state = await api('/jobs?id=' + encodeURIComponent(pending.job_id));
      if (state.status === 'completed') return state.result;
      if (state.status === 'failed') throw new Error(state.error);
      notice(state.status === 'queued' ? 'Queued...' : 'Processing dataset...');
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error('The job is still running. Refresh saved datasets or runs shortly.');
  }
  async function refreshLists(selectedId) {
    const [datasets, runs] = await Promise.all([api('/datasets'), api('/runs')]);
    const selected = selectedId || byId('lab-dataset').value;
    byId('lab-dataset').innerHTML = '<option value="">Select a dataset</option>' + datasets.items.map(item =>
      `<option value="${escape(item.id)}">${escape(item.name)} (${item.quality.observed_days} days)</option>`).join('');
    byId('lab-dataset').value = selected;
    byId('lab-runs').innerHTML = '<option value="">Select a run</option>' + runs.items.map(item =>
      `<option value="${escape(item.id)}">${escape(item.dataset_name)} / ${item.horizon}d / ${escape(item.created_at.slice(0, 16))}</option>`).join('');
    setBusy(busy);
  }
  function showDataset(data, run = null) {
    dataset = data; currentRun = run;
    byId('lab-title').textContent = data.name;
    const source = data.source === 'demo_synthetic' ? 'SYNTHETIC SAMPLE' : data.source;
    byId('lab-provenance').textContent = `${source} | ${data.rows[0].date} to ${data.rows.at(-1).date} | Asia/Kuala_Lumpur`;
    const stats = run ? [
      ['Forecast total', number(run.forecast_total) + ' kWh'],
      [`vs previous ${run.horizon} days`, run.change_percent === null ? 'Unavailable' : number(run.change_percent) + '%'],
      ['Backtest MAE', number(run.models[run.method].mae) + ' kWh'],
      ['History coverage', number(data.quality.coverage_percent) + '%'],
    ] : [['Complete days', data.quality.observed_days], ['History coverage', number(data.quality.coverage_percent) + '%'],
      ['Missing days', data.quality.missing_days.length], ['Latest reading age', data.quality.latest_age_days + ' days']];
    byId('lab-metrics').innerHTML = stats.map(([label, value]) => `<div class="lab-stat"><span>${escape(label)}</span><strong>${escape(value)}</strong></div>`).join('');
    byId('lab-empty').hidden = true; byId('lab-chart-region').hidden = false; byId('lab-readings').hidden = false;
    const future = run?.forecast || [];
    const labels = data.rows.map(row => row.date).concat(future.map(row => row.date));
    const history = data.rows.map(row => row.value);
    const series = [{name: 'Historical energy', unit: 'kWh', data: history.concat(future.map(() => null))}];
    if (run) {
      const padding = history.map(() => null);
      series.push({name: 'Forecast', unit: 'kWh', dashed: true, data: padding.concat(future.map(row => row.value))});
      series.push({name: 'Lower error band', unit: 'kWh', dashed: true, data: padding.concat(future.map(row => row.lower))});
      series.push({name: 'Upper error band', unit: 'kWh', dashed: true, data: padding.concat(future.map(row => row.upper))});
    }
    renderDashboardChart(byId('lab-chart'), {type: 'line', labels, series});
    byId('lab-band-note').textContent = run ? `${run.method_name}. ${run.band_label}. Forecast begins after ${data.rows.at(-1).date}.${data.quality.latest_age_days > 1 ? ' History is stale; forecast dates may already be in the past.' : ''}` : '';
    byId('lab-models').hidden = !run;
    if (run) byId('lab-model-rows').innerHTML = Object.entries(run.models).map(([key, model]) =>
      `<tr><td>${escape(model.name)}</td><td>${number(model.mae)}</td><td>${model.wape_percent === null ? 'Unavailable' : number(model.wape_percent) + '%'}</td><td>${model.folds}</td><td class="lab-selected">${key === run.method ? 'Selected' : ''}</td></tr>`).join('');
    byId('lab-reading-rows').innerHTML = data.rows.map(row => `<tr><td>${escape(row.date)}</td><td>Historical</td><td>${number(row.value)}</td><td></td></tr>`).concat(
      future.map(row => `<tr><td>${escape(row.date)}</td><td class="lab-selected">Forecast</td><td>${number(row.value)}</td><td>${number(row.lower)} - ${number(row.upper)}</td></tr>`)).join('');
    setBusy(busy);
  }
  byId('lab-sample').addEventListener('click', () => action(async () => {
    const result = await api('/demo', {}); await refreshLists(result.id); showDataset(result); notice('Synthetic sample loaded.');
  }));
  byId('lab-import-form').addEventListener('submit', event => { event.preventDefault(); action(async () => {
    const file = byId('lab-file').files[0];
    if (!file || file.size > 1900000) throw new Error('Choose a CSV smaller than 1.9 MB.');
    const result = await api('/import', {csv: await file.text(), name: byId('lab-name').value,
      unit: byId('lab-unit').value, reading_kind: byId('lab-reading-kind').value,
      site_id: byId('lab-import-site').value ? Number(byId('lab-import-site').value) : null});
    await refreshLists(result.id); showDataset(result); notice('Dataset imported.');
  }); });
  byId('lab-extract-form').addEventListener('submit', event => { event.preventDefault(); action(async () => {
    const result = await job('/extract', {site_id: Number(byId('lab-site').value), history_days: Number(byId('lab-history').value), device_id: byId('lab-device').value || null});
    await refreshLists(result.id); showDataset(result); notice('Historical extraction completed.');
  }); });
  byId('lab-forecast-form').addEventListener('submit', event => { event.preventDefault(); action(async () => {
    const result = await job('/forecast', {dataset_id: byId('lab-dataset').value, horizon: Number(byId('lab-horizon').value), method: byId('lab-method').value});
    const data = await api('/dataset?id=' + encodeURIComponent(result.dataset_id));
    showDataset(data, result); await refreshLists(data.id); byId('lab-runs').value = result.id;
    notice(result.source === 'demo_synthetic' ? 'Sample forecast completed using synthetic data.' : 'Prediction completed.');
  }); });
  byId('lab-dataset').addEventListener('change', () => action(async () => {
    if (byId('lab-dataset').value) showDataset(await api('/dataset?id=' + encodeURIComponent(byId('lab-dataset').value)));
    else { currentRun = null; dataset = null; byId('lab-title').textContent = 'Energy prediction'; byId('lab-provenance').textContent = ''; byId('lab-metrics').innerHTML = ''; byId('lab-empty').hidden = false; ['lab-chart-region', 'lab-models', 'lab-readings'].forEach(id => { byId(id).hidden = true; }); }
  }));
  byId('lab-runs').addEventListener('change', () => action(async () => {
    if (!byId('lab-runs').value) return;
    const run = await api('/run?id=' + encodeURIComponent(byId('lab-runs').value));
    showDataset(await api('/dataset?id=' + encodeURIComponent(run.dataset_id)), run); byId('lab-dataset').value = run.dataset_id;
  }));
  byId('lab-download').addEventListener('click', () => {
    if (!currentRun) return;
    const csv = 'date,forecast_kwh,lower_error_band_kwh,upper_error_band_kwh\n' + currentRun.forecast.map(row => [row.date, row.value, row.lower, row.upper].join(',')).join('\n');
    const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
    link.download = 'energy-forecast-' + currentRun.id + '.csv'; link.click(); URL.revokeObjectURL(link.href);
  });
  window.PredictionLab = {async show() {
    try {
      if (!initialized) {
        const config = await api('');
        byId('lab-source-state').textContent = config.source_configured ? 'Configured export connection' : 'Bulk export source not configured';
        byId('lab-site').innerHTML = '<option value="">Select a site</option>' + (config.source_configured ? config.sites : []).map(site => `<option value="${escape(site.id)}">${escape(site.name || site.id)}</option>`).join('');
        byId('lab-site').addEventListener('change', () => setBusy(busy));
        initialized = true;
      }
      await refreshLists(); charts.forEach(chart => chart.resize());
    } catch (error) { notice(error.message, true); }
  }};
  if (window.lucide) lucide.createIcons();
  if (location.hash === '#predictions') document.querySelector('[data-view=predictions]').click();
})();
