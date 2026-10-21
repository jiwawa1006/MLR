// app.js - Controller for Multiple Linear Regression Explorer

(function () {
  'use strict';

  // Global State
  let currentDatasetId = 'amse_housing';
  let currentAlgo = 'forward';
  let currentK = 192;
  let rmseChart = null;
  let r2Chart = null;
  let showAllKButtons = false;
  let showAllEquationTerms = false;
  let showAllCalcFeatures = false;
  let currentZoomRange = 'all'; // 'all' | 'early' | 'sweet'

  const ALGO_COLORS = {
    lasso: '#4f46e5',
    forward: '#f97316',
    backward: '#06b6d4',
    rfe: '#10b981',
    elastic_net: '#ec4899',
    mutual_information: '#8b5cf6',
    variance_threshold: '#64748b',
  };

  // Wait for DOM & Data with resilient polling
  function tryInitApp() {
    if (window.DATASETS_DATA && Object.keys(window.DATASETS_DATA).length > 0) {
      initApp();
    } else {
      let attempts = 0;
      const interval = setInterval(() => {
        attempts++;
        if (window.DATASETS_DATA && Object.keys(window.DATASETS_DATA).length > 0) {
          clearInterval(interval);
          initApp();
        } else if (attempts > 200) { // 10 seconds timeout
          clearInterval(interval);
          console.error('Timeout waiting for window.DATASETS_DATA.');
        }
      }, 50);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', tryInitApp);
  } else {
    tryInitApp();
  }

  function initApp() {
    if (!window.DATASETS_DATA || Object.keys(window.DATASETS_DATA).length === 0) {
      console.error('No dataset data found in window.DATASETS_DATA.');
      return;
    }

    populateDatasetDropdown();
    setupGlobalEventListeners();
    loadDataset(currentDatasetId);

    const loader = document.getElementById('appLoader');
    if (loader) {
      loader.classList.add('fade-out');
      setTimeout(() => {
        if (loader.parentNode) loader.parentNode.removeChild(loader);
      }, 400);
    }
  }

  // Populate Dropdown Menu
  function populateDatasetDropdown() {
    const select = document.getElementById('datasetSelect');
    if (!select) return;
    select.innerHTML = '';

    const datasets = window.DATASETS_DATA;
    const datasetIds = Object.keys(datasets);

    // Prefer amse_housing first if present, otherwise first available
    if (datasets['amse_housing']) {
      currentDatasetId = 'amse_housing';
    } else {
      currentDatasetId = datasetIds[0];
    }

    // Sort dropdown so amse_housing is the very first option
    const sortedIds = [...datasetIds].sort((a, b) => {
      if (a === 'amse_housing') return -1;
      if (b === 'amse_housing') return 1;
      return a.localeCompare(b);
    });

    sortedIds.forEach((id) => {
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = datasets[id].name || id;
      if (id === currentDatasetId) {
        opt.selected = true;
      }
      select.appendChild(opt);
    });

    select.addEventListener('change', (e) => {
      loadDataset(e.target.value);
    });
  }

  function setupGlobalEventListeners() {
    // Static plot drawer toggle
    const toggleBtn = document.getElementById('togglePlotsBtn');
    const drawerToggle = document.getElementById('plotDrawerToggle');
    const drawerBody = document.getElementById('plotDrawerBody');
    const drawerIcon = document.getElementById('plotDrawerIcon');

    if (drawerToggle && drawerBody) {
      const toggleDrawer = () => {
        const isHidden = drawerBody.style.display === 'none';
        drawerBody.style.display = isHidden ? 'block' : 'none';
        if (drawerIcon) {
          drawerIcon.className = isHidden ? 'fa-solid fa-chevron-up' : 'fa-solid fa-chevron-down';
        }
        if (toggleBtn) {
          toggleBtn.querySelector('span').textContent = isHidden ? 'Hide Figures' : 'Show Figures';
        }
      };

      drawerToggle.addEventListener('click', toggleDrawer);
      if (toggleBtn) {
        toggleBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          toggleDrawer();
        });
      }
    }

    // Stepper & Slider Global Listeners
    const slider = document.getElementById('kRangeSlider');
    const numInput = document.getElementById('kNumberInput');
    const stepDec = document.getElementById('kStepDec');
    const stepInc = document.getElementById('kStepInc');

    if (slider) {
      slider.addEventListener('input', (e) => {
        setK(parseInt(e.target.value, 10));
      });
    }
    if (numInput) {
      numInput.addEventListener('change', (e) => {
        setK(parseInt(e.target.value, 10));
      });
    }
    if (stepDec) {
      stepDec.addEventListener('click', () => {
        setK(currentK - 1);
      });
    }
    if (stepInc) {
      stepInc.addEventListener('click', () => {
        setK(currentK + 1);
      });
    }
  }

  // Load / Switch Dataset
  function loadDataset(datasetId) {
    const ds = window.DATASETS_DATA[datasetId];
    if (!ds) return;

    currentDatasetId = datasetId;
    showAllKButtons = false;
    showAllEquationTerms = false;
    showAllCalcFeatures = false;

    // Reset default algo and k to sweet spot
    const availableAlgos = Object.keys(ds.methods);
    if (!availableAlgos.includes(currentAlgo)) {
      currentAlgo = availableAlgos[0] || 'lasso';
    }

    const sweet = ds.sweet_spot || { k: 2, method: currentAlgo };
    currentK = sweet.k || 2;
    if (availableAlgos.includes(sweet.method)) {
      currentAlgo = sweet.method;
    }

    // Update Header & Breadcrumb
    const bc = document.getElementById('breadcrumbDataset');
    if (bc) bc.textContent = ds.name;
    const ht = document.getElementById('headerTitle');
    if (ht) ht.textContent = `${ds.name} MLR Explorer`;

    // Update Hero Banner
    const hTitle = document.getElementById('heroTitle');
    if (hTitle) hTitle.textContent = `${ds.name} - Feature Progression & Model Parsimony`;
    const hDesc = document.getElementById('heroDesc');
    if (hDesc) hDesc.textContent = ds.description;
    const hName = document.getElementById('heroDatasetName');
    if (hName) hName.textContent = ds.name;
    const hSamples = document.getElementById('heroSamples');
    if (hSamples) hSamples.textContent = `${ds.total_samples} (${ds.train_samples} Train / ${ds.test_samples} Test)`;
    const hPredictors = document.getElementById('heroPredictors');
    if (hPredictors) hPredictors.textContent = `${ds.n_encoded_features} Encoded Features`;

    const hSweet = document.getElementById('heroSweetSpot');
    if (hSweet) {
      hSweet.textContent = `k=${sweet.k} via ${sweet.method.toUpperCase()} (${formatNumber(sweet.test_rmse, ds.unit)})`;
    }

    // Render Algorithm Tabs
    renderAlgorithmTabs(ds);

    // Update UI for current selection
    updateAlgorithmSelection();
  }

  function renderAlgorithmTabs(ds) {
    const tabsContainer = document.getElementById('algoTabs');
    if (!tabsContainer) return;
    tabsContainer.innerHTML = '';

    const methods = ds.methods;
    Object.keys(methods).forEach((mKey) => {
      const mObj = methods[mKey];
      const btn = document.createElement('button');
      btn.className = `tab-btn ${mKey === currentAlgo ? 'active' : ''}`;
      btn.dataset.algo = mKey;
      btn.innerHTML = `
        <i class="fa-solid fa-code-branch"></i>
        <span>${mObj.name}</span>
        <span class="badge-pill">${mObj.badge}</span>
      `;
      btn.addEventListener('click', () => {
        document.querySelectorAll('#algoTabs .tab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        currentAlgo = mKey;
        updateAlgorithmSelection();
      });
      tabsContainer.appendChild(btn);
    });

    // "Compare All Methods" tab
    const allBtn = document.createElement('button');
    allBtn.className = `tab-btn ${currentAlgo === 'ALL' ? 'active' : ''}`;
    allBtn.dataset.algo = 'ALL';
    allBtn.style.borderLeft = '2px solid var(--primary)';
    allBtn.innerHTML = `
      <i class="fa-solid fa-layer-group" style="color: #fbbf24;"></i>
      <span>Compare All 7 Methods</span>
      <span class="badge-pill">Overview</span>
    `;
    allBtn.addEventListener('click', () => {
      document.querySelectorAll('#algoTabs .tab-btn').forEach((b) => b.classList.remove('active'));
      allBtn.classList.add('active');
      currentAlgo = 'ALL';
      updateAlgorithmSelection();
    });
    tabsContainer.appendChild(allBtn);
  }

  function updateAlgorithmSelection() {
    const ds = window.DATASETS_DATA[currentDatasetId];
    if (!ds) return;

    renderKButtons(ds);
    renderDashboardMetrics(ds);
    renderActivePredictors(ds);
    renderCharts(ds);
    renderEquation(ds);
    renderCoefficientsTable(ds);
    renderLiveCalculator(ds);
    renderOutliersTable(ds);
    updateStaticPlots(ds);
  }

  function renderKButtons(ds) {
    const kContainer = document.getElementById('kSelectorGroup');
    const presetsContainer = document.getElementById('kPresetsContainer');
    const slider = document.getElementById('kRangeSlider');
    const numInput = document.getElementById('kNumberInput');
    const maxLabel = document.getElementById('kMaxLabel');
    const controlLabel = document.getElementById('kControlLabel');

    if (kContainer) kContainer.innerHTML = '';
    if (presetsContainer) presetsContainer.innerHTML = '';

    const methodData = currentAlgo === 'ALL'
      ? (ds.methods['lasso'] ? ds.methods['lasso'].data : Object.values(ds.methods)[0].data)
      : (ds.methods[currentAlgo] ? ds.methods[currentAlgo].data : Object.values(ds.methods)[0].data);

    if (!methodData || methodData.length === 0) return;

    const sweetK = ds.sweet_spot.k || 2;
    const maxK = methodData.length;

    // Constrain currentK
    if (currentK > maxK) currentK = maxK;
    if (currentK < 1) currentK = 1;

    // Update Label & Stepper / Slider
    if (controlLabel) {
      controlLabel.innerHTML = `<i class="fa-solid fa-list-ol"></i> 2. Number of Selected Features (k = 1 to ${maxK})`;
    }
    if (maxLabel) maxLabel.textContent = `/ ${maxK}`;

    if (slider) {
      slider.min = 1;
      slider.max = maxK;
      slider.value = currentK;
    }
    if (numInput) {
      numInput.min = 1;
      numInput.max = maxK;
      numInput.value = currentK;
    }

    // Milestone Presets
    if (presetsContainer) {
      const presets = [];

      // 1. Sweet Spot
      presets.push({ label: `<i class="fa-solid fa-star"></i> Sweet Spot (k=${sweetK})`, k: sweetK, isSweet: true });

      // 2. Minimal model (k=1)
      if (sweetK !== 1) {
        presets.push({ label: 'k=1 (Min)', k: 1 });
      }

      // 3. Intermediate milestones
      const milestones = [2, 3, 5, 10, 20, 50, 100];
      milestones.forEach((m) => {
        if (m < maxK && m !== sweetK && m !== 1 && !presets.some(p => p.k === m)) {
          if (maxK > 16 && (m === 5 || m === 10 || (maxK > 30 && m === 20) || (maxK > 100 && m === 50))) {
            presets.push({ label: `k=${m}`, k: m });
          } else if (maxK <= 16) {
            presets.push({ label: `k=${m}`, k: m });
          }
        }
      });

      // 4. Full model (k=maxK)
      if (maxK > 1 && !presets.some(p => p.k === maxK)) {
        presets.push({ label: `k=${maxK} (Full)`, k: maxK });
      }

      presets.sort((a, b) => a.k - b.k);

      presets.forEach((p) => {
        const chip = document.createElement('button');
        let cls = 'k-preset-chip';
        if (p.isSweet) cls += ' sweet';
        if (p.k === currentK) cls += ' active';
        chip.className = cls;
        chip.innerHTML = p.label;
        chip.addEventListener('click', () => {
          setK(p.k);
        });
        presetsContainer.appendChild(chip);
      });
    }

    // Individual K Buttons
    if (kContainer) {
      if (maxK <= 16 || showAllKButtons) {
        methodData.forEach((row) => {
          const k = row.k;
          const btn = document.createElement('button');
          const isSweet = k === sweetK;
          const isOverfit = k === maxK && maxK >= 4;

          let extraClass = '';
          if (isSweet) extraClass += ' sweet-spot';
          if (isOverfit && !isSweet) extraClass += ' overfitting';
          if (k === currentK) extraClass += ' active';

          btn.className = `k-btn${extraClass}`;
          btn.dataset.k = k;

          let labelExtra = '';
          if (isSweet) {
            labelExtra = ' ★';
          }

          btn.innerHTML = `<span>k = ${k}${labelExtra}</span>`;
          btn.addEventListener('click', () => {
            setK(k);
          });
          kContainer.appendChild(btn);
        });

        if (maxK > 16) {
          const toggleBtn = document.createElement('button');
          toggleBtn.className = 'btn btn-secondary btn-sm';
          toggleBtn.innerHTML = '<i class="fa-solid fa-chevron-up"></i> Collapse Individual Buttons';
          toggleBtn.addEventListener('click', () => {
            showAllKButtons = false;
            renderKButtons(ds);
          });
          kContainer.appendChild(toggleBtn);
        }
      } else {
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'btn btn-secondary btn-sm';
        toggleBtn.innerHTML = `<i class="fa-solid fa-list-ol"></i> Show All ${maxK} Individual Buttons`;
        toggleBtn.addEventListener('click', () => {
          showAllKButtons = true;
          renderKButtons(ds);
        });
        kContainer.appendChild(toggleBtn);
      }
    }
  }

  function setK(newK) {
    const ds = window.DATASETS_DATA[currentDatasetId];
    if (!ds) return;

    const methodData = currentAlgo === 'ALL'
      ? (ds.methods['lasso'] ? ds.methods['lasso'].data : Object.values(ds.methods)[0].data)
      : (ds.methods[currentAlgo] ? ds.methods[currentAlgo].data : Object.values(ds.methods)[0].data);
    const maxK = methodData ? methodData.length : 5;

    let k = parseInt(newK, 10);
    if (isNaN(k)) k = 1;
    if (k < 1) k = 1;
    if (k > maxK) k = maxK;

    currentK = k;

    // Update Slider & Number Input
    const slider = document.getElementById('kRangeSlider');
    const numInput = document.getElementById('kNumberInput');
    if (slider) slider.value = currentK;
    if (numInput) numInput.value = currentK;

    // Update preset chips
    document.querySelectorAll('#kPresetsContainer .k-preset-chip').forEach((c) => {
      const isMatch = c.textContent.includes(`k=${currentK}`);
      c.classList.toggle('active', isMatch);
    });

    // Update buttons
    document.querySelectorAll('#kSelectorGroup .k-btn').forEach((b) => {
      b.classList.toggle('active', parseInt(b.dataset.k, 10) === currentK);
    });

    updateKSelection();
  }

  function updateKSelection() {
    const ds = window.DATASETS_DATA[currentDatasetId];
    if (!ds) return;

    renderDashboardMetrics(ds);
    renderActivePredictors(ds);
    renderCharts(ds);
    renderEquation(ds);
    renderCoefficientsTable(ds);
    renderLiveCalculator(ds);
    renderOutliersTable(ds);
  }

  // Normal CDF approximation for p-value calculation
  function normalCdf(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989422804014327 * Math.exp(-x * x / 2);
    const prob = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x > 0 ? 1 - prob : prob;
  }

  // Normalize row: reconstruct features, coefficients, and outliers from compact schema
  function normalizeRow(row, ds) {
    if (!row) return row;

    // 1. Reconstruct features from indices
    if (!row.features && row.fi && ds && ds.features) {
      row.features = row.fi.map((idx) => ds.features[idx] || `Feat_${idx}`);
    }

    // 2. Reconstruct coefficients from compact pairs [coef, stderr]
    if (!row.coefficients && row.c) {
      const feats = row.features || [];
      const tCrit = 1.96;
      row.coefficients = row.c.map(([coef, stderr], i) => {
        const feature = i === 0 ? 'Intercept (β₀)' : (feats[i - 1] || `Var_${i}`);
        const tstat = Math.abs(stderr) > 1e-6 ? coef / stderr : 0;
        const pval = 2 * (1 - normalCdf(Math.abs(tstat)));
        return {
          feature,
          coef,
          stderr,
          tstat: Math.round(tstat * 1000) / 1000,
          pvalue: Math.max(0, Math.min(1, Math.round(pval * 100000) / 100000)),
          ci_lower: Math.round((coef - tCrit * stderr) * 100) / 100,
          ci_upper: Math.round((coef + tCrit * stderr) * 100) / 100,
        };
      });
    }

    // 3. Reconstruct outliers from compact triples [index, actual, predicted]
    if (!row.outliers && row.o) {
      row.outliers = row.o.map(([origId, actual, predicted], idx) => {
        const residual = Math.round((actual - predicted) * 100) / 100;
        const abs_error = Math.abs(residual);
        const pct_error = Math.round((abs_error / Math.max(Math.abs(actual), 1e-4) * 100) * 100) / 100;
        let status = 'normal';
        if (pct_error > 15) status = 'alert';
        else if (pct_error > 8) status = 'warning';
        return {
          rank: idx + 1,
          index: origId,
          actual,
          predicted,
          residual,
          abs_error,
          pct_error,
          status,
        };
      });
    }

    return row;
  }

  function getActiveRowData(ds) {
    const methodObj = currentAlgo === 'ALL'
      ? (ds.methods['lasso'] || Object.values(ds.methods)[0])
      : ds.methods[currentAlgo];

    if (!methodObj || !methodObj.data) return null;
    const rawRow = methodObj.data.find((d) => d.k === currentK) || methodObj.data[0];
    return normalizeRow(rawRow, ds);
  }

  function renderDashboardMetrics(ds) {
    const row = getActiveRowData(ds);
    if (!row) return;

    const unit = ds.unit || '';
    const isSweet = row.k === ds.sweet_spot.k;

    // RMSE
    document.getElementById('valRMSE').textContent = formatNumber(row.test_rmse, unit);
    const rmseBadge = document.getElementById('rmseBadge');
    if (isSweet) {
      rmseBadge.innerHTML = '<i class="fa-solid fa-award"></i> Optimal';
      rmseBadge.className = 'badge-tag badge-green';
    } else {
      rmseBadge.innerHTML = `k=${row.k}`;
      rmseBadge.className = 'badge-tag badge-slate';
    }

    // R2
    document.getElementById('valR2').textContent = row.test_r2.toFixed(4);
    const pctFit = Math.max(0, row.test_r2 * 100).toFixed(1);
    document.getElementById('r2Badge').innerHTML = `<i class="fa-solid fa-circle-check"></i> ${pctFit}% Fit`;
    document.getElementById('r2VarianceNote').textContent = `Explains ${pctFit}% of ${ds.target} variance`;

    // Adj R2
    document.getElementById('valAdjR2').textContent = row.adj_r2.toFixed(4);

    // MAE
    document.getElementById('valMAE').textContent = formatNumber(row.mae, unit);

    // CV RMSE
    document.getElementById('valCVRMSE').textContent = formatNumber(row.cv_rmse, unit);

    // F-Stat
    const fVal = row.f_stat ? row.f_stat.toFixed(1) : 'N/A';
    const fP = row.f_pvalue < 0.001 ? 'p < 0.001' : `p = ${row.f_pvalue.toFixed(3)}`;
    document.getElementById('fStatBadge').textContent = `F = ${fVal} (${fP})`;
  }

  function renderActivePredictors(ds) {
    const row = getActiveRowData(ds);
    if (!row) return;

    document.getElementById('featuresCount').textContent = row.features.length;
    const chipsBox = document.getElementById('featuresListChips');
    chipsBox.innerHTML = '';

    row.features.forEach((feat) => {
      const chip = document.createElement('span');
      chip.className = 'feature-chip';
      chip.textContent = feat;
      chipsBox.appendChild(chip);
    });

    const isSweet = row.k === ds.sweet_spot.k;
    const parsimony = document.getElementById('parsimonyNotice');
    if (isSweet) {
      parsimony.className = 'badge-tag badge-green parsimony-badge';
      parsimony.innerHTML = `<i class="fa-solid fa-star"></i> Occam's Razor: Optimal sweet spot (k=${row.k}) achieves lowest test error without overfitting.`;
    } else if (row.k > ds.sweet_spot.k) {
      parsimony.className = 'badge-tag badge-slate parsimony-badge';
      parsimony.innerHTML = `<i class="fa-solid fa-triangle-exclamation" style="color: var(--accent-amber);"></i> Model complexity increased: adding extra variables increases variance and test error.`;
    } else {
      parsimony.className = 'badge-tag badge-slate parsimony-badge';
      parsimony.innerHTML = `<i class="fa-solid fa-info-circle"></i> Under-parameterized model (k=${row.k}): higher bias.`;
    }
  }

  // Vertical Guide Lines Plugin for Chart.js
  const verticalGuidePlugin = {
    id: 'verticalGuidePlugin',
    afterDatasetsDraw(chart) {
      const { ctx, chartArea, scales: { x } } = chart;
      if (!x || !chartArea || !chart.config.options.customGuides) return;

      const { sweetK, currentK, displayData } = chart.config.options.customGuides;
      const { top, bottom } = chartArea;

      // 1. Draw Sweet Spot Guide (Emerald dashed)
      const sweetIdx = displayData.findIndex((d) => d.k === sweetK);
      if (sweetIdx !== -1) {
        const xPos = x.getPixelForTick(sweetIdx);
        if (xPos >= chartArea.left && xPos <= chartArea.right) {
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.lineWidth = 1.8;
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)'; // Emerald
          ctx.beginPath();
          ctx.moveTo(xPos, top);
          ctx.lineTo(xPos, bottom);
          ctx.stroke();

          ctx.fillStyle = '#059669';
          ctx.font = 'bold 10px Inter, sans-serif';
          ctx.fillText(`★ k=${sweetK}`, xPos + 4, top + 14);
          ctx.restore();
        }
      }

      // 2. Draw Active K Guide (Rose dashed)
      const activeIdx = displayData.findIndex((d) => d.k === currentK);
      if (activeIdx !== -1 && activeIdx !== sweetIdx) {
        const xPos = x.getPixelForTick(activeIdx);
        if (xPos >= chartArea.left && xPos <= chartArea.right) {
          ctx.save();
          ctx.setLineDash([3, 3]);
          ctx.lineWidth = 1.8;
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.8)'; // Rose
          ctx.beginPath();
          ctx.moveTo(xPos, top);
          ctx.lineTo(xPos, bottom);
          ctx.stroke();

          ctx.fillStyle = '#e11d48';
          ctx.font = 'bold 10px Inter, sans-serif';
          ctx.fillText(`● k=${currentK}`, xPos + 4, bottom - 6);
          ctx.restore();
        }
      }
    },
  };

  // Render Chart.js Visualizations
  function renderCharts(ds) {
    const ctxRMSE = document.getElementById('rmseChart').getContext('2d');
    const ctxR2 = document.getElementById('r2Chart').getContext('2d');

    if (rmseChart) rmseChart.destroy();
    if (r2Chart) r2Chart.destroy();

    const isCompareAll = currentAlgo === 'ALL';
    const methodObj = isCompareAll
      ? (ds.methods['lasso'] || Object.values(ds.methods)[0])
      : ds.methods[currentAlgo];
    const fullMethodData = methodObj ? methodObj.data : [];
    const totalK = fullMethodData.length;
    const sweetK = ds.sweet_spot.k || 2;
    const isDense = totalK > 20;

    // Badges
    document.getElementById('rmseChartBadge').textContent = isCompareAll ? 'Comparing 7 Methods' : ds.methods[currentAlgo].name;
    document.getElementById('r2ChartBadge').textContent = isCompareAll ? 'Comparing 7 Methods' : ds.methods[currentAlgo].name;

    // View Range Toolbar setup
    const toolbarRow = document.getElementById('chartToolbarRow');
    const zoomMaxKEl = document.getElementById('zoomMaxK');
    const zoomSweetKEl = document.getElementById('zoomSweetK');

    if (!isCompareAll && totalK > 25) {
      if (toolbarRow) toolbarRow.style.display = 'flex';
      if (zoomMaxKEl) zoomMaxKEl.textContent = totalK;
      if (zoomSweetKEl) zoomSweetKEl.textContent = sweetK;

      document.querySelectorAll('#chartZoomPills .zoom-pill').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.range === currentZoomRange);
        if (!btn._hasZoomListener) {
          btn._hasZoomListener = true;
          btn.addEventListener('click', () => {
            currentZoomRange = btn.dataset.range;
            document.querySelectorAll('#chartZoomPills .zoom-pill').forEach((b) => b.classList.toggle('active', b.dataset.range === currentZoomRange));
            renderCharts(ds);
          });
        }
      });
    } else {
      if (toolbarRow) toolbarRow.style.display = 'none';
      currentZoomRange = 'all';
    }

    // Apply Focus Range Slicing
    let displayData = fullMethodData;
    if (!isCompareAll && currentZoomRange === 'early') {
      const cutoff = Math.min(totalK, 40);
      displayData = fullMethodData.slice(0, cutoff);
    } else if (!isCompareAll && currentZoomRange === 'sweet') {
      const minK = Math.max(1, sweetK - 25);
      const maxK = Math.min(totalK, sweetK + 25);
      displayData = fullMethodData.filter((d) => d.k >= minK && d.k <= maxK);
    }

    // Stat Strips update
    const sweetRow = fullMethodData.find((d) => d.k === sweetK) || fullMethodData[0];
    const activeRow = fullMethodData.find((d) => d.k === currentK) || fullMethodData[0];

    const rmseSweetText = document.getElementById('rmseSweetText');
    if (rmseSweetText && sweetRow) rmseSweetText.textContent = `k=${sweetK} (${formatNumber(sweetRow.test_rmse, ds.unit)})`;
    const rmseActiveText = document.getElementById('rmseActiveText');
    if (rmseActiveText && activeRow) rmseActiveText.textContent = `k=${activeRow.k} (${formatNumber(activeRow.test_rmse, ds.unit)})`;

    const r2SweetText = document.getElementById('r2SweetText');
    if (r2SweetText && sweetRow) r2SweetText.textContent = `k=${sweetK} (${sweetRow.test_r2.toFixed(4)})`;
    const r2ActiveText = document.getElementById('r2ActiveText');
    if (r2ActiveText && activeRow) r2ActiveText.textContent = `k=${activeRow.k} (${activeRow.test_r2.toFixed(4)})`;

    if (!isCompareAll) {
      const trainRMSE = displayData.map((d) => d.train_rmse);
      const testRMSE = displayData.map((d) => d.test_rmse);
      const trainR2 = displayData.map((d) => d.train_r2);
      const testR2 = displayData.map((d) => d.test_r2);
      const labels = displayData.map((d) => `k=${d.k}`);

      // Point styling: smooth continuous line with NO cluttered dots on normal steps when dense
      const trainRadii = displayData.map(() => (isDense ? 0 : 3.5));
      const testRadii = displayData.map((d) => {
        if (d.k === currentK) return 8; // vibrant prominent selected marker
        if (d.k === sweetK) return 7; // optimal sweet spot star
        return isDense ? 0 : 4;
      });

      const testPointStyles = displayData.map((d) => {
        if (d.k === sweetK && d.k !== currentK) return 'star';
        return 'circle';
      });

      const testPointColors = displayData.map((d) => {
        if (d.k === currentK) return '#f43f5e'; // vibrant Rose
        if (d.k === sweetK) return '#10b981'; // Emerald
        return '#4f46e5';
      });

      const testBorderColors = displayData.map((d) => {
        if (d.k === currentK || d.k === sweetK) return '#ffffff';
        return '#4f46e5';
      });

      const testBorderWidths = displayData.map((d) => {
        if (d.k === currentK || d.k === sweetK) return 2.5;
        return 1;
      });

      // Chart Options
      const chartOpts = (yTitle, unit) => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        customGuides: { sweetK, currentK, displayData },
        interaction: {
          mode: 'index',
          intersect: false,
        },
        onClick: (event, elements) => {
          if (elements && elements.length > 0) {
            const clickedIdx = elements[0].index;
            if (displayData[clickedIdx]) {
              setK(displayData[clickedIdx].k);
            }
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              boxWidth: 14,
              font: { family: "'Inter', sans-serif", size: 11.5, weight: '600' },
            },
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.92)',
            titleFont: { family: "'Outfit', sans-serif", size: 13, weight: '700' },
            bodyFont: { family: "'Fira Code', monospace", size: 12 },
            padding: 10,
            cornerRadius: 8,
            callbacks: {
              title: function (items) {
                if (!items.length) return '';
                const idx = items[0].dataIndex;
                const item = displayData[idx];
                const sweetBadge = item.k === sweetK ? ' ★ Optimal Sweet Spot' : '';
                return `k = ${item.k} Features${sweetBadge}`;
              },
              label: function (context) {
                const val = context.raw;
                return ` ${context.dataset.label}: ${typeof val === 'number' ? val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 }) : val} ${unit}`;
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Number of Selected Features (k)',
              font: { family: "'Outfit', sans-serif", size: 12, weight: '600' },
              color: '#64748b',
              padding: { top: 6 },
            },
            grid: { color: 'rgba(226, 232, 240, 0.5)' },
            ticks: {
              font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
              color: '#64748b',
              autoSkip: true,
              maxTicksLimit: isDense ? 12 : 20,
              maxRotation: 0,
              minRotation: 0,
            },
          },
          y: {
            title: {
              display: true,
              text: yTitle,
              font: { family: "'Outfit', sans-serif", size: 12, weight: '600' },
              color: '#64748b',
            },
            grid: { color: 'rgba(226, 232, 240, 0.6)' },
            ticks: {
              font: { family: "'Inter', sans-serif", size: 11 },
              color: '#64748b',
            },
          },
        },
      });

      // RMSE Chart
      rmseChart = new Chart(ctxRMSE, {
        type: 'line',
        plugins: [verticalGuidePlugin],
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Train RMSE',
              data: trainRMSE,
              borderColor: '#94a3b8',
              backgroundColor: 'rgba(148, 163, 184, 0.08)',
              borderDash: [5, 5],
              borderWidth: 1.8,
              pointRadius: trainRadii,
              pointHoverRadius: 6,
              hitRadius: 8,
              tension: 0.15,
            },
            {
              label: 'Test RMSE',
              data: testRMSE,
              borderColor: '#4f46e5',
              backgroundColor: 'rgba(79, 70, 229, 0.08)',
              borderWidth: 2.2,
              pointRadius: testRadii,
              pointStyle: testPointStyles,
              pointBackgroundColor: testPointColors,
              pointBorderColor: testBorderColors,
              pointBorderWidth: testBorderWidths,
              pointHoverRadius: 8,
              hitRadius: 10,
              tension: 0.15,
              fill: true,
            },
          ],
        },
        options: chartOpts('RMSE', ds.unit),
      });

      // R2 Chart
      r2Chart = new Chart(ctxR2, {
        type: 'line',
        plugins: [verticalGuidePlugin],
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Train R²',
              data: trainR2,
              borderColor: '#94a3b8',
              backgroundColor: 'rgba(148, 163, 184, 0.08)',
              borderDash: [5, 5],
              borderWidth: 1.8,
              pointRadius: trainRadii,
              pointHoverRadius: 6,
              hitRadius: 8,
              tension: 0.15,
            },
            {
              label: 'Test R²',
              data: testR2,
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.08)',
              borderWidth: 2.2,
              pointRadius: testRadii,
              pointStyle: testPointStyles,
              pointBackgroundColor: testPointColors,
              pointBorderColor: testBorderColors,
              pointBorderWidth: testBorderWidths,
              pointHoverRadius: 8,
              hitRadius: 10,
              tension: 0.15,
              fill: true,
            },
          ],
        },
        options: chartOpts('R² Score', ''),
      });
    } else {
      // Compare All 7 Methods Overlay
      const maxPts = Math.max(...Object.values(ds.methods).map((m) => m.data.length));
      const compLabels = Array.from({ length: maxPts }, (_, i) => `k=${i + 1}`);
      const compDense = maxPts > 20;

      const rmseDatasets = [];
      const r2Datasets = [];

      Object.keys(ds.methods).forEach((mKey) => {
        const m = ds.methods[mKey];
        const color = ALGO_COLORS[mKey] || '#64748b';
        rmseDatasets.push({
          label: m.name,
          data: m.data.map((d) => d.test_rmse),
          borderColor: color,
          backgroundColor: 'transparent',
          pointRadius: compDense ? 0 : 3,
          pointHoverRadius: 6,
          hitRadius: 8,
          borderWidth: 2,
          tension: 0.15,
        });
        r2Datasets.push({
          label: m.name,
          data: m.data.map((d) => d.test_r2),
          borderColor: color,
          backgroundColor: 'transparent',
          pointRadius: compDense ? 0 : 3,
          pointHoverRadius: 6,
          hitRadius: 8,
          borderWidth: 2,
          tension: 0.15,
        });
      });

      const compOpts = (yTitle, unit) => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 250 },
        interaction: {
          mode: 'index',
          intersect: false,
        },
        onClick: (event, elements) => {
          if (elements && elements.length > 0) {
            setK(elements[0].index + 1);
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              boxWidth: 14,
              font: { family: "'Inter', sans-serif", size: 11, weight: '600' },
            },
          },
          tooltip: {
            backgroundColor: 'rgba(15, 23, 42, 0.92)',
            titleFont: { family: "'Outfit', sans-serif", size: 13, weight: '700' },
            bodyFont: { family: "'Fira Code', monospace", size: 12 },
            padding: 10,
            cornerRadius: 8,
            callbacks: {
              title: function (items) {
                if (!items.length) return '';
                return `k = ${items[0].dataIndex + 1} Features`;
              },
              label: function (context) {
                const val = context.raw;
                return ` ${context.dataset.label}: ${typeof val === 'number' ? val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 }) : val} ${unit}`;
              },
            },
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Number of Features (k)',
              font: { family: "'Outfit', sans-serif", size: 12, weight: '600' },
              color: '#64748b',
              padding: { top: 6 },
            },
            grid: { color: 'rgba(226, 232, 240, 0.5)' },
            ticks: {
              font: { family: "'Inter', sans-serif", size: 11, weight: '500' },
              color: '#64748b',
              autoSkip: true,
              maxTicksLimit: compDense ? 12 : 20,
              maxRotation: 0,
            },
          },
          y: {
            title: {
              display: true,
              text: yTitle,
              font: { family: "'Outfit', sans-serif", size: 12, weight: '600' },
              color: '#64748b',
            },
            grid: { color: 'rgba(226, 232, 240, 0.6)' },
            ticks: {
              font: { family: "'Inter', sans-serif", size: 11 },
              color: '#64748b',
            },
          },
        },
      });

      rmseChart = new Chart(ctxRMSE, {
        type: 'line',
        data: {
          labels: compLabels,
          datasets: rmseDatasets,
        },
        options: compOpts('Test RMSE (Comparison)', ds.unit),
      });

      r2Chart = new Chart(ctxR2, {
        type: 'line',
        data: {
          labels: compLabels,
          datasets: r2Datasets,
        },
        options: compOpts('Test R² (Comparison)', ''),
      });
    }
  }

  // Render Fitted Mathematical Equation Box
  function renderEquation(ds) {
    const row = getActiveRowData(ds);
    if (!row || !row.coefficients) return;

    const coefs = row.coefficients;
    const intercept = coefs.find((c) => c.feature.includes('Intercept')) || coefs[0];
    const slopeCoefs = coefs.filter((c) => !c.feature.includes('Intercept'));

    const expandRow = document.getElementById('equationExpandRow');
    const toggleBtn = document.getElementById('equationToggleBtn');
    const toggleText = document.getElementById('equationToggleText');

    let termsToRender = slopeCoefs;
    const isTruncated = slopeCoefs.length > 8 && !showAllEquationTerms;

    if (isTruncated) {
      termsToRender = slopeCoefs.slice(0, 8);
      if (expandRow) expandRow.style.display = 'block';
      if (toggleText) toggleText.textContent = `Show all ${slopeCoefs.length} terms (+${slopeCoefs.length - 8} more)`;
    } else {
      if (slopeCoefs.length > 8) {
        if (expandRow) expandRow.style.display = 'block';
        if (toggleText) toggleText.textContent = 'Collapse to top 8 terms';
      } else {
        if (expandRow) expandRow.style.display = 'none';
      }
    }

    let formulaHtml = `<span class="equation-var">${ds.target}</span> = <span class="equation-coef">${intercept.coef.toFixed(4)}</span>`;

    termsToRender.forEach((c) => {
      const sign = c.coef >= 0 ? '+' : '-';
      const absVal = Math.abs(c.coef).toFixed(4);
      formulaHtml += ` ${sign} <span class="equation-coef">${absVal}</span> × (<span class="equation-var">${c.feature}</span>)`;
    });

    if (isTruncated) {
      formulaHtml += ` <span style="color: #a5b4fc; font-style: italic;">+ ... (${slopeCoefs.length - 8} more predictors)</span>`;
    }

    document.getElementById('equationBox').innerHTML = formulaHtml;

    if (toggleBtn && !toggleBtn._hasListener) {
      toggleBtn._hasListener = true;
      toggleBtn.addEventListener('click', () => {
        showAllEquationTerms = !showAllEquationTerms;
        renderEquation(ds);
      });
    }
  }

  // Render Coefficients & Statistical Significance Table
  function renderCoefficientsTable(ds) {
    const row = getActiveRowData(ds);
    const tbody = document.getElementById('coefTableBody');
    tbody.innerHTML = '';

    if (!row || !row.coefficients) return;

    row.coefficients.forEach((c) => {
      const tr = document.createElement('tr');

      let star = 'NS';
      let starColor = 'var(--slate-400)';
      if (c.pvalue < 0.001) {
        star = '***';
        starColor = 'var(--accent-emerald)';
      } else if (c.pvalue < 0.01) {
        star = '**';
        starColor = 'var(--accent-emerald)';
      } else if (c.pvalue < 0.05) {
        star = '*';
        starColor = 'var(--accent-amber)';
      }

      const pStr = c.pvalue < 0.0001 ? '< 0.0001' : c.pvalue.toFixed(4);

      tr.innerHTML = `
        <td style="font-weight: 600; color: var(--slate-900);">${c.feature}</td>
        <td class="mono">${c.coef.toFixed(4)}</td>
        <td class="mono">${c.stderr.toFixed(4)}</td>
        <td class="mono">${c.tstat.toFixed(3)}</td>
        <td class="mono" style="color: ${c.pvalue < 0.05 ? 'var(--primary)' : 'inherit'}; font-weight: ${c.pvalue < 0.05 ? '700' : 'normal'};">${pStr}</td>
        <td class="mono" style="font-size: 11.5px; color: var(--slate-500);">[${c.ci_lower.toFixed(2)}, ${c.ci_upper.toFixed(2)}]</td>
        <td><span style="font-weight: 700; color: ${starColor};">${star}</span></td>
      `;

      tbody.appendChild(tr);
    });
  }

  // Render Interactive Live Predictor Calculator
  function renderLiveCalculator(ds) {
    const row = getActiveRowData(ds);
    const container = document.getElementById('calculatorInputsContainer');
    const expandRow = document.getElementById('calcExpandRow');
    const toggleBtn = document.getElementById('calcToggleAllBtn');
    const toggleText = document.getElementById('calcToggleText');
    const resetBtn = document.getElementById('resetCalcBtn');

    container.innerHTML = '';

    document.getElementById('predTargetLabel').textContent = `${ds.target} (${ds.unit || ''})`;

    if (!row || !row.features) return;

    const isTruncated = row.features.length > 8 && !showAllCalcFeatures;

    if (isTruncated) {
      if (expandRow) expandRow.style.display = 'block';
      if (toggleText) toggleText.textContent = `Show all ${row.features.length} features (+${row.features.length - 8} more)`;
    } else {
      if (row.features.length > 8) {
        if (expandRow) expandRow.style.display = 'block';
        if (toggleText) toggleText.textContent = 'Collapse to top 8 features';
      } else {
        if (expandRow) expandRow.style.display = 'none';
      }
    }

    row.features.forEach((feat, idx) => {
      const stats = (ds.feature_stats && ds.feature_stats[feat]) || {
        min: 0,
        max: 100,
        step: 1,
        default: 50,
      };

      const isHidden = isTruncated && idx >= 8;
      const group = document.createElement('div');
      group.className = 'input-group';
      if (isHidden) group.style.display = 'none';

      const safeId = feat.replace(/[^a-zA-Z0-9]/g, '_');
      const inputId = `calc_num_${safeId}`;
      const sliderId = `calc_range_${safeId}`;
      const labelId = `calc_val_${safeId}`;

      group.innerHTML = `
        <div class="input-label-row">
          <span title="${feat}" style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${feat}</span>
          <span id="${labelId}" style="color: var(--primary); font-family: var(--font-mono); font-size: 12px;">${stats.default.toFixed(2)}</span>
        </div>
        <input type="number" id="${inputId}" class="input-field" value="${stats.default}" min="${stats.min}" max="${stats.max}" step="${stats.step}">
        <input type="range" id="${sliderId}" class="range-slider" value="${stats.default}" min="${stats.min}" max="${stats.max}" step="${stats.step}">
      `;

      container.appendChild(group);

      const numInput = group.querySelector(`#${inputId}`);
      const rangeInput = group.querySelector(`#${sliderId}`);
      const valLabel = group.querySelector(`#${labelId}`);

      numInput.addEventListener('input', () => {
        rangeInput.value = numInput.value;
        valLabel.textContent = Number(numInput.value).toFixed(2);
        calculatePrediction(ds, row);
      });

      rangeInput.addEventListener('input', () => {
        numInput.value = rangeInput.value;
        valLabel.textContent = Number(rangeInput.value).toFixed(2);
        calculatePrediction(ds, row);
      });
    });

    if (toggleBtn && !toggleBtn._hasListener) {
      toggleBtn._hasListener = true;
      toggleBtn.addEventListener('click', () => {
        showAllCalcFeatures = !showAllCalcFeatures;
        renderLiveCalculator(ds);
      });
    }

    if (resetBtn && !resetBtn._hasListener) {
      resetBtn._hasListener = true;
      resetBtn.addEventListener('click', () => {
        row.features.forEach((feat) => {
          const stats = (ds.feature_stats && ds.feature_stats[feat]);
          if (!stats) return;
          const safeId = feat.replace(/[^a-zA-Z0-9]/g, '_');
          const numInput = document.getElementById(`calc_num_${safeId}`);
          const rangeInput = document.getElementById(`calc_range_${safeId}`);
          const valLabel = document.getElementById(`calc_val_${safeId}`);
          if (numInput) numInput.value = stats.default;
          if (rangeInput) rangeInput.value = stats.default;
          if (valLabel) valLabel.textContent = Number(stats.default).toFixed(2);
        });
        calculatePrediction(ds, row);
      });
    }

    calculatePrediction(ds, row);
  }

  function calculatePrediction(ds, row) {
    if (!row || !row.coefficients) return;

    const coefs = row.coefficients;
    const interceptObj = coefs.find((c) => c.feature.includes('Intercept')) || coefs[0];
    let yHat = interceptObj.coef;

    row.features.forEach((feat) => {
      const inputId = `calc_num_${feat.replace(/[^a-zA-Z0-9]/g, '_')}`;
      const el = document.getElementById(inputId);
      const val = el ? parseFloat(el.value) || 0 : 0;

      const coefObj = coefs.find((c) => c.feature === feat);
      if (coefObj) {
        yHat += coefObj.coef * val;
      }
    });

    document.getElementById('predOutputValue').textContent = formatNumber(yHat, ds.unit);
  }

  // Render Outliers Diagnostics Table
  function renderOutliersTable(ds) {
    const row = getActiveRowData(ds);
    const tbody = document.getElementById('outliersTableBody');
    tbody.innerHTML = '';

    if (!row || !row.outliers) return;

    row.outliers.forEach((item) => {
      const tr = document.createElement('tr');

      let dotClass = 'dot-green';
      let statusText = 'Normal';

      if (item.status === 'alert' || item.pct_error > 15) {
        dotClass = 'dot-red';
        statusText = 'High Residual';
      } else if (item.status === 'warning' || item.pct_error > 8) {
        dotClass = 'dot-amber';
        statusText = 'Moderate';
      }

      tr.innerHTML = `
        <td class="mono" style="font-weight: 700;">#${item.rank}</td>
        <td class="mono">Sample [${item.index}]</td>
        <td class="mono">${formatNumber(item.actual, ds.unit)}</td>
        <td class="mono">${formatNumber(item.predicted, ds.unit)}</td>
        <td class="mono" style="color: ${item.residual < 0 ? 'var(--accent-rose)' : 'var(--accent-emerald)'}; font-weight: 600;">
          ${item.residual >= 0 ? '+' : ''}${formatNumber(item.residual, ds.unit)}
        </td>
        <td class="mono">${formatNumber(item.abs_error, ds.unit)}</td>
        <td class="mono">${item.pct_error.toFixed(2)}%</td>
        <td>
          <span class="status-dot ${dotClass}"></span>
          <span style="font-size: 12px; font-weight: 600;">${statusText}</span>
        </td>
      `;

      tbody.appendChild(tr);
    });
  }

  // Update Static High-Res Figures
  function updateStaticPlots(ds) {
    const algoKey = currentAlgo === 'ALL' ? 'lasso' : currentAlgo;
    const rmsePath = `fig/${ds.id}/${algoKey}.png`;
    const r2Path = `fig/${ds.id}/${algoKey}_r2.png`;

    const imgRMSE = document.getElementById('staticPlotRMSE');
    const linkRMSE = document.getElementById('staticPlotRMSELink');
    const imgR2 = document.getElementById('staticPlotR2');
    const linkR2 = document.getElementById('staticPlotR2Link');

    if (imgRMSE) imgRMSE.src = rmsePath;
    if (linkRMSE) linkRMSE.href = rmsePath;

    if (imgR2) imgR2.src = r2Path;
    if (linkR2) linkR2.href = r2Path;
  }

  function formatNumber(num, unit) {
    if (typeof num !== 'number' || isNaN(num)) return 'N/A';
    const isCurrency = unit === '$';
    const prefix = isCurrency ? '$' : '';
    const suffix = !isCurrency && unit ? ` ${unit}` : '';

    return `${prefix}${num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}${suffix}`;
  }

})();
